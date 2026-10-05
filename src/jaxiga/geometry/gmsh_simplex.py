"""Gmsh simplex conversion and independent Bernstein Jacobian checks."""

from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np

from jaxiga.geometry.simplex import SimplexPatch, evaluate_simplex
from jaxiga.space.simplex import bernstein_simplex, indices, lattice


@lru_cache(None)
def _det_rule(dim, degree):
    points = lattice(dim, degree)
    B = bernstein_simplex(points, degree)[0]
    return points, np.linalg.inv(B)


def simplex_quality(patch):
    """Sample distortion and certify a positive geometry determinant.

    Polynomial determinants have total degree d*(p-1); rational maps use the
    homogeneous numerator of degree (d+1)*p-d and positive denominator bounds.
    Bernstein coefficients bound these over the entire simplex; ambiguous
    bounds trigger longest-edge bisection. No Gmsh/OpenMP routines are involved.
    """
    dim, p = patch.dim, patch.degree[0]
    control = np.asarray(patch.ctrl_pts)
    weights = np.asarray(patch.weights)
    rational = not np.allclose(weights,1,rtol=0,atol=1e-14)
    H = np.column_stack([control*weights[:,None],weights])

    def jac(points, numerator=False):
        B,dB,_ = bernstein_simplex(points,p)
        if not rational:
            result = np.einsum("qdi,ia->qda",dB,control)
            return np.linalg.det(result) if numerator else result
        values = B@H
        deriv = np.einsum("qdi,ia->qda",dB,H)
        if numerator:
            return np.linalg.det(np.concatenate([deriv,values[:,None,:]],axis=1))
        W=values[:,-1,None,None]
        return (deriv[:,:,:dim]*W-deriv[:,:,-1:]*values[:,None,:dim])/W**2

    J = jac(lattice(dim, max(4, 2 * p + 2)))
    determinants = np.linalg.det(J)
    scaled = determinants / np.maximum(np.prod(np.linalg.norm(J, axis=2), axis=1), 1e-300)
    if not np.isfinite(J).all() or np.min(determinants) <= 0:
        raise ValueError("inverted or degenerate simplex geometry")
    nodes, inverse = _det_rule(dim, (dim+1)*p-dim if rational else dim*(p-1))
    bary = np.column_stack([1 - nodes.sum(axis=1), nodes])

    def bound(vertices, depth=0):
        samples = jac(bary @ vertices, numerator=True)
        coeff = inverse @ samples
        margin = 1e-11 * np.max(np.abs(coeff))
        lower = float(coeff.min() - margin)
        if lower > 0:
            return lower
        if samples.min() <= 0 or depth >= 12:
            raise ValueError("simplex Jacobian is nonpositive or cannot be certified positive")
        a, b = max(combinations(range(dim + 1), 2),
                   key=lambda pair: np.linalg.norm(vertices[pair[0]] - vertices[pair[1]]))
        midpoint = (vertices[a] + vertices[b]) / 2
        left, right = vertices.copy(), vertices.copy()
        left[a], right[b] = midpoint, midpoint
        return min(bound(left, depth + 1), bound(right, depth + 1))

    return float(scaled.min()), bound(np.vstack([np.zeros(dim), np.eye(dim)]))/float(weights.max())**(dim+1)


def convert_simplex(gmsh, dim, source, min_quality):
    from jaxiga.geometry.gmsh import GmshImportError, GmshMesh, _elements, _MeshingFailure

    blocks = _elements(gmsh, dim)
    if not blocks:
        raise _MeshingFailure(f"no {dim}D elements were generated")
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(xyz).reshape(-1, 3)
    if dim == 2 and np.ptp(xyz[:, 2]) > 1e-10 * max(float(np.ptp(xyz, axis=0).max()), 1e-30):
        raise GmshImportError("2D meshes must lie in a plane parallel to XY; shells are unsupported")
    node_lookup = dict(zip(map(int, tags), xyz[:, :dim]))
    patches, element_tags, corners, quality, lower, traces = [], [], [], [], [], []
    for kind, ids, conn in blocks:
        name, _, p, count, ref, primary = gmsh.model.mesh.getElementProperties(kind)
        family = "Triangle" if dim == 2 else "Tetrahedron"
        if not name.startswith(family) or primary != dim + 1 or count != comb(p + dim, dim):
            raise GmshImportError("cell_type='simplex' requires complete triangles/tetrahedra of one degree")
        if not 1 <= p <= 4:
            raise GmshImportError("simplex Gmsh geometry supports degrees 1 through 4")
        reference = np.asarray(ref).reshape(count, dim)
        transform = np.linalg.inv(bernstein_simplex(reference, p)[0])
        alpha = indices(dim, p)
        permutation = [alpha.index((a[1], a[0], *a[2:])) for a in alpha]
        for tag, cell in zip(ids, conn.reshape(-1, count)):
            if len(set(cell)) != count:
                raise GmshImportError("a simplex cell has repeated node tags")
            try:
                points = transform @ np.array([node_lookup[int(t)] for t in cell])
            except KeyError as exc:
                raise GmshImportError(f"missing node {exc}") from exc
            patch = SimplexPatch.create(p, points)
            primary_tags = cell[:primary].copy()
            bary = np.column_stack([1 - reference.sum(axis=1), reference])
            J = bernstein_simplex(np.full((1, dim), 1 / (dim + 1)), p)[1][0] @ points
            if np.linalg.det(J) < 0:
                patch = SimplexPatch.create(p, points[permutation])
                primary_tags[[0, 1]] = primary_tags[[1, 0]]
                bary[:, [0, 1]] = bary[:, [1, 0]]
            try:
                q, bound = simplex_quality(patch)
            except ValueError as exc:
                raise _MeshingFailure(f"element {tag}: {exc}") from exc
            if q < min_quality:
                raise _MeshingFailure(f"element {tag}: scaled Jacobian {q:g} < {min_quality:g}")
            patches.append(patch)
            element_tags.append(int(tag))
            corners.append(primary_tags)
            quality.append(q)
            lower.append(bound)
            traces.append([tuple(sorted(cell[np.isclose(bary[:, s], 0)])) for s in range(dim + 1)])
    if len({p.degree for p in patches}) != 1:
        raise GmshImportError("mixed simplex degrees are unsupported; elevate to a common degree")
    if len({tuple(sorted(c)) for c in corners}) != len(corners):
        raise GmshImportError("duplicate simplex cells")
    faces, face_nodes, supports = defaultdict(list), {}, defaultdict(list)
    for e, cell in enumerate(corners):
        for side in range(dim + 1):
            key = tuple(sorted(np.delete(cell, side)))
            faces[key].append((e, f"f{side}"))
            if key in face_nodes and face_nodes[key] != traces[e][side]:
                raise GmshImportError("nonconforming high-order nodes on a shared simplex face")
            face_nodes[key] = traces[e][side]
        for count in range(2, dim + 1):
            for local in combinations(range(dim + 1), count):
                supports[tuple(sorted(cell[list(local)]))].append((e, local))
    if any(len(v) > 2 for v in faces.values()):
        raise GmshImportError("non-manifold simplex mesh")
    physical_names, cell_groups, boundary_groups = {}, defaultdict(set), defaultdict(set)
    labels = [{} for _ in patches]
    index = {t: i for i, t in enumerate(element_tags)}
    for group_dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(group_dim, tag) or f"physical_{group_dim}_{tag}"
        physical_names[group_dim, tag] = name
        for entity in gmsh.model.getEntitiesForPhysicalGroup(group_dim, tag):
            for kind, ids, conn in _elements(gmsh, group_dim, int(entity)):
                if group_dim == dim:
                    cell_groups[name].update(index[int(t)] for t in ids)
                elif group_dim == dim - 1:
                    _, _, _, count, _, primary = gmsh.model.mesh.getElementProperties(kind)
                    for face in conn.reshape(-1, count)[:, :primary]:
                        key = tuple(sorted(face))
                        if key not in faces:
                            raise GmshImportError(f"physical boundary {name!r} does not match a simplex face")
                        for e, side in faces[key]:
                            if side in labels[e] and labels[e][side] != name:
                                raise GmshImportError("overlapping physical boundary groups")
                            labels[e][side] = name
                            boundary_groups[name].add((e, side))
    patches = [p.with_labels(**label) for p, label in zip(patches, labels)]
    mesh = GmshMesh(patches, np.asarray(element_tags),
                    {k: tuple(sorted(v)) for k, v in cell_groups.items()},
                    {k: tuple(sorted(v)) for k, v in boundary_groups.items()}, physical_names,
                    np.asarray(quality), np.asarray(lower), str(source), gmsh.__version__)
    mesh._corner_tags = np.asarray(corners)
    mesh._min_quality = min_quality
    mesh._plane_z = float(xyz[0, 2]) if dim == 2 else 0.0
    constraints = [set() for _ in patches]
    for edim, entity in gmsh.model.getEntities():
        if 0 < edim < dim:
            for kind, _, conn in _elements(gmsh, edim, entity):
                _, _, _, count, _, primary = gmsh.model.mesh.getElementProperties(kind)
                for cell in conn.reshape(-1, count)[:, :primary]:
                    for e, local in supports.get(tuple(sorted(cell)), ()):
                        constraints[e].add((edim, entity, local))
    mesh._cad_constraints = tuple(tuple(sorted(c)) for c in constraints)
    return mesh


def cad_error(gmsh, mesh):
    from jaxiga.geometry.gmsh_cad import _project

    error = 0.0
    for patch, constraints in zip(mesh, mesh._cad_constraints):
        for dim, entity, local in constraints:
            points = lattice(dim, max(10, 3 * patch.degree[0] + 1))
            bary = np.zeros((len(points), patch.dim + 1))
            bary[:, local] = np.column_stack([1 - points.sum(axis=1), points])
            xyz = evaluate_simplex(patch, bary[:, 1:])
            closest = _project(gmsh, dim, entity, xyz, plane_z=mesh._plane_z)
            error = max(error, float(np.linalg.norm(xyz - closest, axis=1).max()))
    return error
