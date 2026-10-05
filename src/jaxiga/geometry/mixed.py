"""Bezier cell restrictions and facet matching for conforming mixed meshes."""

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.spatial import cKDTree

from jaxiga.geometry.nurbs import Patch, side_name
from jaxiga.geometry.simplex import SimplexPatch
from jaxiga.geometry.gmsh_geometry import spline_basis, tensor_grid
from jaxiga.space.simplex import bernstein_simplex, lattice


def homogeneous(patch):
    w = np.asarray(patch.weights)
    return np.column_stack([np.asarray(patch.ctrl_pts) * w[:, None], w])


def polynomial_basis(patch, points):
    if isinstance(patch, SimplexPatch):
        return bernstein_simplex(points, patch.degree)[0]
    return spline_basis(patch, points)


def from_homogeneous(patch, control):
    if not np.isfinite(control).all() or np.any(control[:, -1] <= 0):
        raise ValueError("reconstruction requires finite positive rational weights")
    xyz, weights = control[:, :-1] / control[:, -1:], control[:, -1]
    if isinstance(patch, SimplexPatch):
        return SimplexPatch.create(patch.degree[0], xyz, weights, patch.label_map)
    return Patch.create(patch.knots, patch.degree, xyz, weights, patch.label_map)


def split_to_simplices(patch):
    """Restrict a single Bezier quad/hex to two triangles/six tetrahedra.

    Homogeneous polynomial restriction preserves the rational geometry exactly.
    The simplex degree is the sum of tensor degrees. No CAD projection is used.
    Adjacent cells must choose compatible face diagonals when both are split.
    """
    if not isinstance(patch, Patch) or patch.dim not in (2, 3):
        raise ValueError("split_to_simplices requires a Bezier quad or hex")
    require_bezier(patch)
    dim, degree = patch.dim, sum(patch.degree)
    nodes = lattice(dim, degree)
    B = bernstein_simplex(nodes, degree)[0]
    bary = np.column_stack([1 - nodes.sum(axis=1), nodes])
    result = []
    for order in permutations(range(dim)):
        corners = [np.zeros(dim)]
        for axis in order:
            vertex = corners[-1].copy()
            vertex[axis] = 1
            corners.append(vertex)
        corners = np.asarray(corners)
        if np.linalg.det(corners[1:] - corners[0]) < 0:
            corners[[0, 1]] = corners[[1, 0]]
        control = np.linalg.solve(
            B, polynomial_basis(patch, bary @ corners) @ homogeneous(patch)
        )
        labels = {}
        for side in range(dim + 1):
            face = np.delete(corners, side, axis=0)
            for d in range(dim):
                for end in (0, 1):
                    label = patch.label_map.get(side_name(d, end))
                    if label is not None and np.all(face[:, d] == end):
                        labels[f"f{side}"] = label
        result.append(
            SimplexPatch.create(
                degree, control[:, :-1] / control[:, -1:], control[:, -1], labels
            )
        )
    return result


def require_bezier(patch):
    if not isinstance(patch, SimplexPatch) and any(
        len(set(k)) != 2 for k in patch.knots
    ):
        raise ValueError(
            "mixed spaces currently require single-span Bezier cells; use mesh cells before knot insertion"
        )


def corners(patch):
    if isinstance(patch, SimplexPatch):
        return np.vstack([np.zeros(patch.dim), np.eye(patch.dim)]), patch.vertex_indices
    ref = tensor_grid([[0, 1]] * patch.dim)
    stride = np.r_[1, np.cumprod(patch.n_cp_per_dir[:-1])]
    ids = np.sum(
        ref.astype(int) * (np.asarray(patch.n_cp_per_dir) - 1) * stride, axis=1
    )
    return ref, ids.astype(int)


@dataclass
class Facet:
    cell: int
    side: str
    code: int
    vertices: tuple
    reference: np.ndarray
    local: np.ndarray


def facets(patches, tol=1e-9):
    refs, ids = zip(*(corners(p) for p in patches))
    xyz = np.concatenate([np.asarray(p.ctrl_pts)[i] for p, i in zip(patches, ids)])
    parent = np.arange(len(xyz))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(xyz).query_pairs(tol):
        parent[root(b)] = root(a)
    vertex_ids = np.array([root(i) for i in range(len(xyz))])
    out, offset = [], 0
    from jaxiga.geometry.multipatch import side_indices

    for e, (patch, ref) in enumerate(zip(patches, refs)):
        vertices = vertex_ids[offset : offset + len(ref)]
        offset += len(ref)
        if isinstance(patch, SimplexPatch):
            for s in range(patch.dim + 1):
                mask = np.arange(patch.dim + 1) != s
                out.append(
                    Facet(
                        e,
                        f"f{s}",
                        s,
                        tuple(vertices[mask]),
                        ref[mask],
                        np.flatnonzero(patch.multi_indices[:, s] == 0),
                    )
                )
        else:
            for d in range(patch.dim):
                for end in (0, 1):
                    mask = ref[:, d] == end
                    side = side_name(d, end)
                    out.append(
                        Facet(
                            e,
                            side,
                            2 * d + end,
                            tuple(vertices[mask]),
                            ref[mask],
                            side_indices(patch, side),
                        )
                    )
    return out


def interfaces(patches):
    """Whole matching facets, plus a quad face covered by two triangle faces."""
    records = facets(patches)
    groups = {}
    for i, f in enumerate(records):
        groups.setdefault(tuple(sorted(f.vertices)), []).append(i)
    pairs, used = [], set()
    for owners in groups.values():
        if len(owners) > 2:
            raise ValueError("non-manifold mixed mesh face")
        if len(owners) == 2:
            a, b = owners
            pairs.append((records[a], records[b]))
            used.update(owners)
    if patches[0].dim == 3:
        for i, face in enumerate(records):
            if i in used or len(face.vertices) != 4:
                continue
            triangles = [
                j
                for j, g in enumerate(records)
                if j not in used
                and len(g.vertices) == 3
                and set(g.vertices) < set(face.vertices)
            ]
            if not triangles:
                continue
            if len(triangles) != 2 or set().union(
                *(set(records[j].vertices) for j in triangles)
            ) != set(face.vertices):
                raise ValueError(
                    "a hex face must meet exactly two triangles covering the full face"
                )
            # The common edge must be a diagonal, not an edge of the quad.
            common = set(records[triangles[0]].vertices) & set(
                records[triangles[1]].vertices
            )
            uv = [face.reference[face.vertices.index(v)] for v in common]
            if len(uv) != 2 or np.count_nonzero(uv[0] != uv[1]) != 2:
                raise ValueError("triangles overlap or do not partition the quad face")
            for j in triangles:
                pairs.append((face, records[j]))
            used.update([i, *triangles])
    return pairs, [f for i, f in enumerate(records) if i not in used]


def trace_transfer(master, fm, slave, fs):
    """Homogeneous coefficients on a slave face from a master face."""
    if isinstance(slave, SimplexPatch):
        q = lattice(slave.dim - 1, slave.degree[0])
        bary = np.column_stack([1 - q.sum(axis=1), q])
        # face_points orders barycentric vertices by their local vertex index.
        slave_ref = bary @ fs.reference
        target_corners = np.array(
            [fm.reference[fm.vertices.index(v)] for v in fs.vertices]
        )
        master_ref = bary @ target_corners
    else:
        axis = "uvw".index(fs.side[0])
        free = [d for d in range(slave.dim) if d != axis]
        q = tensor_grid([np.linspace(0, 1, slave.degree[d] + 1) for d in free])
        slave_ref = np.full((len(q), slave.dim), int(fs.side[1]), dtype=float)
        slave_ref[:, free] = q
        # Multilinear corner coordinates also handle rotations/reflections.
        shape = np.prod(
            np.where(
                fs.reference[None, :, free] == 1, q[:, None, :], 1 - q[:, None, :]
            ),
            axis=-1,
        )
        target_corners = np.array(
            [fm.reference[fm.vertices.index(v)] for v in fs.vertices]
        )
        master_ref = shape @ target_corners
    B = polynomial_basis(slave, slave_ref)[:, fs.local]
    T = np.linalg.solve(B, polynomial_basis(master, master_ref))
    T[np.abs(T) < 2e-12] = 0
    return T


def ordered_interfaces(patches):
    pairs, boundary = interfaces(patches)
    ordered = []
    for a, b in pairs:
        pa, pb = patches[a.cell], patches[b.cell]
        # Quad-face masters are necessary at hex/tet transitions. Otherwise
        # use the lower degree trace, with cell index as a deterministic tie.
        if len(a.vertices) < len(b.vertices) or (
            len(a.vertices) == len(b.vertices)
            and (max(pa.degree), a.cell) > (max(pb.degree), b.cell)
        ):
            a, b = b, a
        ordered.append((a, b))
    return ordered, boundary


def conform_mixed(patches):
    """Elevate cells and reparameterize slave traces to match their masters.

    This changes imported simplex geometry when a bilinear/curved hex trace
    differs from the original tetrahedral trace. Callers must validate the new
    cell Jacobians. FunctionSpace itself never changes the supplied geometry.
    """
    patches = list(patches)
    pairs, _ = ordered_interfaces(patches)
    required = [max(p.degree) for p in patches]
    for a, b in pairs:
        if len(a.vertices) == 4 and len(b.vertices) == 3:
            d = "uvw".index(a.side[0])
            required[b.cell] = max(
                required[b.cell],
                sum(p for k, p in enumerate(patches[a.cell].degree) if k != d),
            )
    patches = [
        p.elevate(q) if max(p.degree) < q else p for p, q in zip(patches, required)
    ]
    pairs, _ = ordered_interfaces(patches)
    for a, b in pairs:
        T = trace_transfer(patches[a.cell], a, patches[b.cell], b)
        h = homogeneous(patches[b.cell])
        h[b.local] = T @ homogeneous(patches[a.cell])
        patches[b.cell] = from_homogeneous(patches[b.cell], h)
    return patches


def outward_normal(patch, face):
    ref = face.reference.mean(axis=0)[None, :]
    step = 1e-6
    from jaxiga.space.bernstein import bernstein_tensor

    def value(r):
        B = (
            bernstein_simplex(r, patch.degree)[0]
            if isinstance(patch, SimplexPatch)
            else bernstein_tensor(2 * r - 1, patch.degree)[0]
        )
        h = B @ homogeneous(patch)
        return h[:, :-1] / h[:, -1:]

    J = np.stack(
        [
            (
                value(ref + np.eye(patch.dim)[d] * step)[0]
                - value(ref - np.eye(patch.dim)[d] * step)[0]
            )
            / (2 * step)
            for d in range(patch.dim)
        ]
    )
    if isinstance(patch, SimplexPatch):
        nr = np.ones(patch.dim) if face.code == 0 else -np.eye(patch.dim)[face.code - 1]
    else:
        nr = np.eye(patch.dim)[face.code // 2] * (2 * (face.code % 2) - 1)
    normal = np.linalg.solve(J, nr)
    return normal / np.linalg.norm(normal)
