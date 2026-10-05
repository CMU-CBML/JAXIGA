"""CAD sampling and refitting for GmshMesh (setup stage, outside JAX tracing)."""

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from itertools import permutations, product
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from jaxiga.geometry.gmsh_geometry import (
    sample_parameters, spline_basis, spline_quality, tensor_grid, tensor_solve,
)
from jaxiga.geometry.nurbs import Patch, evaluate_patch


def _project(gmsh, entity_dim, tag, points, plane_z=0.0):
    dim = points.shape[1]
    xyz = np.zeros((len(points), 3))
    xyz[:, :dim] = points
    if dim == 2:
        xyz[:, 2] = plane_z
    projected, _ = gmsh.model.getClosestPoint(entity_dim, tag, xyz.ravel())
    result = np.asarray(projected).reshape(-1, 3)[:, :dim]
    if result.shape != points.shape or not np.isfinite(result).all():
        raise ValueError(f"CAD projection failed on entity ({entity_dim}, {tag})")
    return result


def cad_error(gmsh, mesh):
    """Maximum sampled distance of spline curves/faces to their CAD entities.

    Samples include off-node points on every knot span. This is an accuracy
    indicator, not a Hausdorff-distance certificate.
    """
    if mesh.cell_type == "mixed":
        from jaxiga.geometry.simplex import SimplexPatch
        from jaxiga.geometry.gmsh_simplex import cad_error as simplex_error
        return max((simplex_error if isinstance(p,SimplexPatch) else cad_error)(
            gmsh,replace(mesh,patches=[p],_cad_constraints=(c,)))
            for p,c in zip(mesh,mesh._cad_constraints))
    error = 0.0
    for patch, constraints in zip(mesh, mesh._cad_constraints):
        for entity_dim, tag, fixed in constraints:
            axes = sample_parameters(patch, max(11, 3 * max(patch.degree) + 2))
            for d, end in fixed:
                axes[d] = [end]
            parameters = tensor_grid(axes)
            points = evaluate_patch(patch, parameters)
            exact = _project(gmsh, entity_dim, tag, points, mesh._plane_z)
            error = max(error, float(np.linalg.norm(points - exact, axis=1).max()))
    return error


def _check_interfaces(mesh, patches):
    """Validate known shared faces, so a crack cannot disappear from topology."""
    corners = tensor_grid([[0, 1]] * patches[0].dim)
    faces = defaultdict(list)
    for i, tags in enumerate(mesh._corner_tags):
        for d in range(patches[i].dim):
            for end in (0, 1):
                key = tuple(sorted(tags[corners[:, d] == end]))
                faces[key].append((i, d, end))
    for owners in faces.values():
        if len(owners) != 2:
            continue
        (a, da, ea), (b, db, eb) = owners
        pa, pb = patches[a], patches[b]
        fa = [d for d in range(pa.dim) if d != da]
        fb = [d for d in range(pb.dim) if d != db]
        ca = corners[corners[:, da] == ea]
        cb = corners[corners[:, db] == eb]
        ta = mesh._corner_tags[a][corners[:, da] == ea]
        tb = mesh._corner_tags[b][corners[:, db] == eb]
        for perm in permutations(fb):
            for flips in product((False, True), repeat=len(fa)):
                mapped = np.array([cb[np.flatnonzero(tb == tag)[0]] for tag in ta])
                if not all(np.array_equal(ca[:, x], 1 - mapped[:, y] if flip else mapped[:, y])
                           for x, y, flip in zip(fa, perm, flips)):
                    continue
                for x, y, flip in zip(fa, perm, flips):
                    knots = np.asarray(pb.knots[y])
                    if flip:
                        knots = 1 - knots[::-1]
                    if (pa.degree[x] != pb.degree[y] or len(pa.knots[x]) != len(knots)
                            or not np.allclose(pa.knots[x], knots, rtol=0, atol=1e-12)):
                        raise ValueError("refinement gives incompatible knot vectors on a shared face")
                axes = sample_parameters(pa)
                axes[da] = [ea]
                qa = tensor_grid(axes)
                qb = np.full_like(qa, eb, dtype=float)
                for x, y, flip in zip(fa, perm, flips):
                    qb[:, y] = 1 - qa[:, x] if flip else qa[:, x]
                xa = evaluate_patch(pa, qa)
                xb = evaluate_patch(pb, qb)
                scale = max(1.0, np.max(np.abs(xa)))
                if not np.allclose(xa, xb, rtol=0, atol=1e-10 * scale):
                    raise ValueError("CAD refitting would open a crack on a shared face")
                break
            else:
                continue
            break
        else:
            raise ValueError("cannot orient a shared mesh face")


def update_geometry(mesh, patches, to_cad):
    """Interpolate CAD-projected Greville samples in the refined spline space.

    Control points are obtained by solving a collocation system. They are not
    projected onto CAD. Unclassified interior samples retain their old geometry.
    Curves take precedence over surfaces at CAD edges in 3D.
    """
    from jaxiga.geometry.gmsh import GmshImportError

    _check_interfaces(mesh, patches)
    if not to_cad or mesh.geometry_representation == "rational_cad":
        return replace(mesh, patches=patches)
    if mesh._source_digest is None:
        raise GmshImportError("CAD-aware refinement requires .geo or STEP input; use to_cad=False for .msh")
    source = Path(mesh.source)
    if not source.is_file() or sha256(source.read_bytes()).hexdigest() != mesh._source_digest:
        raise GmshImportError("source CAD file is missing or changed; import it again before refitting")
    import gmsh
    if gmsh.isInitialized():
        raise RuntimeError("CAD-aware refinement requires its own Gmsh session; finalize the active session first")
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(mesh.source)
        updated, qualities, bounds = [], [], []
        for patch, constraints in zip(patches, mesh._cad_constraints):
            if not np.allclose(patch.weights, 1, rtol=0, atol=1e-14):
                raise GmshImportError("CAD refitting currently requires polynomial patches")
            greville, matrices = [], []
            for knots, degree, count in zip(patch.knots, patch.degree, patch.n_cp_per_dir):
                points = np.array([np.mean(knots[i + 1:i + degree + 1]) for i in range(count)])
                if np.any(np.diff(points) <= 0):
                    raise GmshImportError("CAD refitting requires distinct Greville points; reduce knot multiplicity")
                greville.append(points)
                matrices.append(BSpline(knots, np.eye(count), degree)(points))
            parameters = tensor_grid(greville)
            values = spline_basis(patch, parameters) @ np.asarray(patch.ctrl_pts)
            vertices = np.all((parameters == 0) | (parameters == 1), axis=1)
            for entity_dim, tag, fixed in constraints:
                mask = np.all([parameters[:, d] == end for d, end in fixed], axis=0)
                # Original mesh vertices already lie on CAD. Reprojecting them
                # can introduce CAD-kernel endpoint noise and separate traces.
                mask &= ~vertices
                if mask.any():
                    values[mask] = _project(gmsh, entity_dim, tag, values[mask], mesh._plane_z)
            control = tensor_solve(matrices, values)
            result = Patch.create(patch.knots, patch.degree, control, labels=patch.labels)
            try:
                quality, bound = spline_quality(result)
            except ValueError as exc:
                raise GmshImportError(str(exc)) from exc
            if quality < mesh._min_quality:
                raise GmshImportError(f"CAD refitting produced scaled Jacobian {quality:g}, below {mesh._min_quality:g}")
            updated.append(result)
            qualities.append(quality)
            bounds.append(bound)
        _check_interfaces(mesh, updated)
        result = replace(mesh, patches=updated, scaled_jacobians=np.asarray(qualities),
                         jacobian_lower_bounds=np.asarray(bounds))
        result.geometry_error = cad_error(gmsh, result)
        return result
    finally:
        gmsh.finalize()
