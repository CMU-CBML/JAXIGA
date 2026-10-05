"""Exact conic and aligned cylindrical CAD reconstruction in homogeneous form.

This is deliberately a supported-family reconstruction, not a NURBS fit to an
arbitrary STEP surface. Unsupported CAD entities fail explicitly.
"""

from dataclasses import replace
from itertools import product

import numpy as np

from jaxiga.geometry.gmsh_geometry import (
    _positive_bound,
    sample_parameters,
    spline_basis,
    tensor_grid,
    tensor_solve,
)
from jaxiga.geometry.mixed import from_homogeneous, homogeneous
from jaxiga.geometry.nurbs import evaluate_patch
from jaxiga.geometry.simplex import SimplexPatch
from jaxiga.space.bernstein import bernstein_basis


def rational_quality(patch):
    """Bernstein lower bound on detJ's homogeneous polynomial numerator."""
    H = homogeneous(patch)
    dim = patch.dim

    def data(points):
        values = spline_basis(patch, points) @ H
        derivative = np.stack(
            [spline_basis(patch, points, d) @ H for d in range(dim)], axis=1
        )
        W = values[:, -1]
        jac = (
            derivative[:, :, :dim] * W[:, None, None]
            - derivative[:, :, -1:] * values[:, None, :dim]
        ) / W[:, None, None] ** 2
        augmented = np.concatenate([derivative, values[:, None, :]], axis=1)
        return jac, np.linalg.det(augmented)

    J, _ = data(tensor_grid(sample_parameters(patch)))
    det = np.linalg.det(J)
    scaled = det / np.maximum(np.linalg.norm(J, axis=2).prod(axis=1), 1e-300)
    if not np.isfinite(scaled).all() or det.min() <= 0:
        raise ValueError(
            "rational reconstruction produced inverted or degenerate geometry"
        )
    degrees = [(dim + 1) * p - 1 for p in patch.degree]
    matrices = [bernstein_basis(np.linspace(-1, 1, p + 1), p) for p in degrees]
    spans = [list(zip(np.unique(k)[:-1], np.unique(k)[1:])) for k in patch.knots]
    bounds = []
    for box in product(*spans):
        points = tensor_grid(
            [np.linspace(a, b, p + 1) for (a, b), p in zip(box, degrees)]
        )
        _, numerator = data(points)
        coefficients = tensor_solve(matrices, numerator[:, None]).reshape(
            tuple(p + 1 for p in degrees), order="F"
        )
        bounds.append(
            _positive_bound(coefficients)
            / float(np.max(patch.weights)) ** (dim + 1)
            / 2**dim
        )
    return float(scaled.min()), min(bounds)


def _quadratic_arc(a, m, b, ta, tb):
    t = np.linalg.lstsq(np.column_stack([ta, -tb]), b - a, rcond=None)[0]
    control = a + t[0] * ta
    scale = max(np.linalg.norm(a - b), 1.0)
    if np.linalg.norm(control - (b + t[1] * tb)) > 1e-9 * scale:
        raise ValueError("CAD conic endpoint tangents do not intersect")
    delta = control - m
    w = np.dot(2 * m - a - b, delta) / (2 * np.dot(delta, delta))
    if not np.isfinite(w) or not 0 < w <= 1 + 1e-9:
        raise ValueError("conic arc must be shorter than a half turn; use a finer mesh")
    result = np.column_stack(
        [np.array([a, control, b]) * np.array([1, w, 1])[:, None], [1, w, 1]]
    )
    test = (result[0] + 2 * result[1] + result[2]) / 4
    if np.linalg.norm(test[:-1] / test[-1] - m) > 1e-9 * scale:
        raise ValueError("CAD curve is not a supported quadratic conic arc")
    return result


def _cad_arc(gmsh, tag, a, b, plane_z):
    dim = len(a)
    xyz = np.zeros((2, 3))
    xyz[:, :dim] = [a, b]
    if dim == 2:
        xyz[:, 2] = plane_z
    parameters = np.asarray(gmsh.model.getParametrization(1, tag, xyz.ravel()))
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    parameters = np.clip(parameters, float(lo[0]), float(hi[0]))
    # OCC inverse parametrization can miss a trimmed endpoint by ~1e-8.
    # Refine in parameter space; do not move shared mesh vertices.
    for _ in range(5):
        values = np.asarray(gmsh.model.getValue(1, tag, parameters)).reshape(2, 3)
        derivative = np.asarray(gmsh.model.getDerivative(1, tag, parameters)).reshape(
            2, 3
        )
        parameters += np.sum((xyz - values) * derivative, axis=1) / np.sum(
            derivative**2, axis=1
        )
        parameters = np.clip(parameters, float(lo[0]), float(hi[0]))
    delta = (parameters[1] - parameters[0] + np.pi) % (2 * np.pi) - np.pi
    middle = parameters[0] + delta / 2
    m = np.asarray(gmsh.model.getValue(1, tag, [middle]))[:dim]
    tangents = np.asarray(gmsh.model.getDerivative(1, tag, parameters)).reshape(2, 3)[
        :, :dim
    ]
    return _quadratic_arc(a, m, b, *tangents)


def _circle_arc(a, m, b):
    v = np.array([m - a, b - a])
    center = a + np.linalg.solve(2 * (v @ v.T), np.sum(v * v, axis=1)) @ v
    normal = np.cross(v[0], v[1])
    normal /= np.linalg.norm(normal)
    return _quadratic_arc(
        a, m, b, np.cross(normal, a - center), np.cross(normal, b - center)
    )


def _elevated_curve(h, degree):
    x = np.linspace(-1, 1, degree + 1)
    return np.linalg.solve(bernstein_basis(x, degree), bernstein_basis(x, 2) @ h)


def _curve_indices(patch, fixed):
    if isinstance(patch, SimplexPatch):
        a, b = fixed
        alpha = patch.multi_indices
        ids = np.flatnonzero(
            np.all(
                alpha[:, [k for k in range(patch.dim + 1) if k not in fixed]] == 0,
                axis=1,
            )
        )
        return ids[np.argsort(alpha[ids, b])], patch.degree[0]
    free = [d for d in range(patch.dim) if d not in dict(fixed)]
    grid = tensor_grid([np.arange(n) for n in patch.n_cp_per_dir])
    mask = np.ones(patch.n_cp, dtype=bool)
    for d, end in fixed:
        mask &= grid[:, d] == end * (patch.n_cp_per_dir[d] - 1)
    ids = np.flatnonzero(mask)
    return ids[np.argsort(grid[ids, free[0]])], patch.degree[free[0]]


def _cylinder_face(gmsh, patch, tag, fixed, plane_z):
    from jaxiga.geometry.gmsh_cad import _project

    if isinstance(patch, SimplexPatch):
        raise ValueError(
            "exact cylindrical reconstruction needs aligned quad faces; reconstruct hexes then split_to_simplices"
        )
    d, end = fixed[0]
    free = [i for i in range(3) if i != d]
    uv = tensor_grid([[0, 1], [0, 1]])
    refs = np.full((4, 3), end, dtype=float)
    refs[:, free] = uv
    xyz = evaluate_patch(patch, refs)
    for arc in (0, 1):
        generator = 1 - arc
        a, b = xyz[0], xyz[1 if arc == 0 else 2]
        shift = xyz[2 if generator == 1 else 1] - a
        if not np.allclose(xyz[3] - b, shift, atol=1e-9, rtol=1e-9):
            continue
        # Generators must be straight segments on the CAD cylinder.
        line = np.array([a + shift / 2, b + shift / 2])
        if not np.allclose(
            _project(gmsh, 2, tag, line, plane_z), line, atol=1e-9, rtol=1e-9
        ):
            continue
        middle = _project(gmsh, 2, tag, ((a + b) / 2)[None, :], plane_z)[0]
        h = _elevated_curve(_circle_arc(a, middle, b), patch.degree[free[arc]])
        grid = tensor_grid([np.arange(n) for n in patch.n_cp_per_dir]).astype(int)
        ids = np.flatnonzero(grid[:, d] == end * (patch.n_cp_per_dir[d] - 1))
        result = homogeneous(patch)
        for i in ids:
            numerator = h[grid[i, free[arc]]].copy()
            t = grid[i, free[generator]] / patch.degree[free[generator]]
            numerator[:3] += t * shift * numerator[-1]
            result[i] = numerator
        return result
    raise ValueError(
        "cylinder faces must follow circular arcs and straight axial generators; use a transfinite hex layout"
    )


def reconstruct(gmsh, mesh):
    """Return rational patches reproducing supported CAD entities exactly."""
    from jaxiga.geometry.gmsh import GmshImportError, _MeshingFailure
    from jaxiga.geometry.gmsh_geometry import spline_quality
    from jaxiga.geometry.gmsh_simplex import simplex_quality
    from jaxiga.geometry.multipatch import compute_topology

    patches = [p.elevate(max(2, max(p.degree))) for p in mesh]
    entity_types = {}
    for constraints in mesh._cad_constraints:
        for dim, tag, _ in constraints:
            kind = gmsh.model.getType(dim, tag)
            entity_types[dim, tag] = kind
            allowed = {1: {"Line", "Circle", "Ellipse"}, 2: {"Plane", "Cylinder"}}[dim]
            if kind not in allowed:
                raise GmshImportError(
                    f"rational CAD reconstruction does not support {kind} entity ({dim}, {tag})"
                )
    try:
        for i, (patch, constraints) in enumerate(zip(patches, mesh._cad_constraints)):
            # Surfaces first, then CAD edge curves (the authoritative trims).
            for dim, tag, fixed in sorted(constraints, reverse=True):
                kind = entity_types[dim, tag]
                if dim == 2 and kind == "Cylinder":
                    patch = from_homogeneous(
                        patch, _cylinder_face(gmsh, patch, tag, fixed, mesh._plane_z)
                    )
                elif dim == 1 and kind in {"Circle", "Ellipse"}:
                    ids, degree = _curve_indices(patch, fixed)
                    xyz = np.asarray(patch.ctrl_pts)
                    curve = _cad_arc(
                        gmsh, tag, xyz[ids[0]], xyz[ids[-1]], mesh._plane_z
                    )
                    h = homogeneous(patch)
                    h[ids] = _elevated_curve(curve, degree)
                    patch = from_homogeneous(patch, h)
            patches[i] = patch
        if mesh.cell_type == "mixed":
            from jaxiga.geometry.mixed import conform_mixed

            patches = conform_mixed(patches)
        quality = [
            simplex_quality(p) if isinstance(p, SimplexPatch) else spline_quality(p)
            for p in patches
        ]
        if min(q for q, _ in quality) < mesh._min_quality:
            raise ValueError(
                "rational reconstruction fails the requested scaled-Jacobian threshold"
            )
        if mesh.cell_type == "mixed":
            from jaxiga.space.mixed import build_mixed_space

            build_mixed_space(patches, 1)
        else:
            compute_topology(patches)
    except ValueError as exc:
        raise _MeshingFailure(str(exc)) from exc
    result = replace(
        mesh,
        patches=patches,
        scaled_jacobians=np.array([q for q, _ in quality]),
        jacobian_lower_bounds=np.array([b for _, b in quality]),
        geometry_representation="rational_cad",
        reconstruction_report=tuple(
            (dim, tag, kind) for (dim, tag), kind in sorted(entity_types.items())
        ),
    )
    from jaxiga.geometry.gmsh_cad import cad_error
    from jaxiga.geometry.gmsh_simplex import cad_error as simplex_error

    result.geometry_error = (
        simplex_error if result.cell_type == "simplex" else cad_error
    )(gmsh, result)
    if result.geometry_error > 1e-7 * max(
        1,
        float(
            np.ptp(
                np.concatenate([np.asarray(p.ctrl_pts) for p in patches]), axis=0
            ).max()
        ),
    ):
        raise GmshImportError(
            f"rational CAD validation failed: sampled error {result.geometry_error:g}"
        )
    return result
