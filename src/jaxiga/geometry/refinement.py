"""Uniform, shape-preserving subdivision of rational Bezier cell meshes."""

from dataclasses import replace
from functools import lru_cache
from itertools import product

import numpy as np

from jaxiga.geometry.gmsh_geometry import _split_coefficients
from jaxiga.geometry.mixed import from_homogeneous, homogeneous, require_bezier
from jaxiga.geometry.nurbs import Patch, evaluate_patch, side_name
from jaxiga.geometry.simplex import SimplexPatch
from jaxiga.space.simplex import indices


def bezier_cells(patches):
    """Extract exact rational Bezier geometry on each tensor knot span.

    Returns cells in FunctionSpace element order with inherited exterior labels.
    Building a new space on these cells gives C0 intercell continuity; using
    them only for visualization does not change the original analysis space.
    """
    from jaxiga.space.function_space import _patch_connectivity

    patches = [patches] if isinstance(patches, Patch) else list(patches)
    result = []
    for patch in patches:
        if not isinstance(patch, Patch):
            raise ValueError("bezier_cells requires tensor patches")
        ien, boxes, operators, _ = _patch_connectivity(patch)
        H = homogeneous(patch)
        knots = [[0] * (p + 1) + [1] * (p + 1) for p in patch.degree]
        for ids, box, op in zip(ien, boxes, operators):
            local = op.T @ H[ids]
            labels = {
                side: label
                for side, label in patch.label_map.items()
                if np.isclose(
                    box["uvw".index(side[0]) + int(side[1]) * patch.dim], int(side[1])
                )
            }
            result.append(
                Patch.create(
                    knots,
                    patch.degree,
                    local[:, :-1] / local[:, -1:],
                    local[:, -1],
                    labels,
                )
            )
    return result


@lru_cache(4)
def _simplex_children(dim, diagonal=0):
    """Oriented midpoint subdivision; every boundary triangle is red-refined."""
    v = np.vstack([np.zeros(dim), np.eye(dim)])
    m = {
        (i, j): (v[i] + v[j]) / 2 for i in range(dim + 1) for j in range(i + 1, dim + 1)
    }
    children = [
        np.array([v[i], *[m[min(i, j), max(i, j)] for j in range(dim + 1) if j != i]])
        for i in range(dim + 1)
    ]
    if dim == 2:
        children.append(np.array([m[0, 1], m[0, 2], m[1, 2]]))
    else:
        # Split the central octahedron along an opposite-midpoint diagonal.
        # Its choice affects only the interior, never neighboring face splits.
        a, b, c, d = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)][diagonal]
        midpoint = lambda i, j: m[min(i, j), max(i, j)]
        ring = [midpoint(a, c), midpoint(a, d), midpoint(b, d), midpoint(b, c)]
        children.extend(
            np.array([midpoint(a, b), midpoint(c, d), ring[i], ring[(i + 1) % 4]])
            for i in range(4)
        )
    for child in children:
        if np.linalg.det(child[1:] - child[0]) < 0:
            child[[0, 1]] = child[[1, 0]]
        child.setflags(write=False)
    return tuple(children)


def _simplex_diagonal(patch):
    if patch.dim == 2:
        return 0
    # Choosing a fixed diagonal can progressively elongate tetrahedra under
    # repeated refinement. Compare the three diagonals in physical space.
    vertices = np.vstack([np.zeros(3), np.eye(3)])
    pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
    points = evaluate_patch(
        patch, np.array([(vertices[i] + vertices[j]) / 2 for i, j in pairs])
    )
    lengths = np.linalg.norm(points[::2] - points[1::2], axis=1)
    # Keep choices deterministic under floating-point ties.
    return int(np.flatnonzero(lengths <= lengths.min() * (1 + 1e-12))[0])


@lru_cache(32)
def _simplex_operators(dim, degree, diagonal=0):
    """Positive Bernstein restriction matrices from the multivariate blossom.

    Repeated barycentric de Casteljau steps avoid a high-degree interpolation
    solve. Each row is a convex combination of the parent control points.
    """
    operators = []
    for vertices in _simplex_children(dim, diagonal):
        bary = np.column_stack([1 - vertices.sum(axis=1), vertices])

        @lru_cache(None)
        def net(sequence):
            if not sequence:
                return np.eye(len(indices(dim, degree)))
            old = net(sequence[:-1])
            q = degree - len(sequence)
            lookup = {alpha: i for i, alpha in enumerate(indices(dim, q + 1))}
            cols = [
                [
                    lookup[tuple(a + (k == j) for k, a in enumerate(beta))]
                    for j in range(dim + 1)
                ]
                for beta in indices(dim, q)
            ]
            return np.einsum("j,ijk->ik", bary[sequence[-1]], old[cols])

        operator = np.array(
            [
                net(tuple(j for j, n in enumerate(alpha) for _ in range(n)))[0]
                for alpha in indices(dim, degree)
            ]
        )
        operator.setflags(write=False)
        operators.append(operator)
        net.cache_clear()
    return tuple(operators)


def _subdivide(patch):
    H = homogeneous(patch)
    if isinstance(patch, SimplexPatch):
        diagonal = _simplex_diagonal(patch)
        for vertices, op in zip(
            _simplex_children(patch.dim, diagonal),
            _simplex_operators(patch.dim, patch.degree[0], diagonal),
        ):
            bary = np.column_stack([1 - vertices.sum(axis=1), vertices])
            labels = {}
            for child_face in range(patch.dim + 1):
                face = np.delete(bary, child_face, axis=0)
                for parent_face in range(patch.dim + 1):
                    label = patch.label_map.get(f"f{parent_face}")
                    if label is not None and np.all(face[:, parent_face] == 0):
                        labels[f"f{child_face}"] = label
            yield replace(
                from_homogeneous(patch, op @ H), labels=tuple(sorted(labels.items()))
            )
    else:
        nets = [H.reshape((*patch.n_cp_per_dir, H.shape[-1]), order="F")]
        for axis in range(patch.dim):
            nets = [child for net in nets for child in _split_coefficients(net, axis)]
        for offset, net in zip(product((0, 1), repeat=patch.dim), nets):
            labels = {
                side_name(d, end): label
                for d, end in enumerate(offset)
                if (label := patch.label_map.get(side_name(d, end))) is not None
            }
            child = from_homogeneous(patch, net.reshape(H.shape, order="F"))
            yield replace(child, labels=tuple(sorted(labels.items())))


def refine_cells(patches, n=1):
    """Uniformly subdivide a 2D/3D Bezier cell mesh, preserving its geometry.

    Each level splits quads/triangles into four cells and hexes/tetrahedra into
    eight. Families, degrees, rational geometry and face labels are preserved.
    Shared edges and faces receive matching midpoint subdivisions; rebuild the
    FunctionSpace to reconstruct its C0 coupling and solve on the new mesh.

    Accepts one Patch/SimplexPatch or a sequence (including GmshMesh). Returns
    a flat list of patches, grouped by parent. Gmsh source tags, cell groups
    and quality diagnostics are not copied to this list. No CAD access or
    projection is needed. For polynomial geometry this preserves the existing
    CAD approximation, rather than improving it.

    All input cells must have the same dimension and be single-span Bezier
    patches with compatible interfaces. Refine the entire mesh: local marking,
    hanging interfaces and automatic solution transfer are not implemented.
    Tensor Patch.refine(), in contrast, inserts knots into a single patch.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a non-negative integer")
    cells = [patches] if isinstance(patches, (Patch, SimplexPatch)) else list(patches)
    if not cells:
        raise ValueError("at least one cell is required")
    dim = getattr(cells[0], "dim", None)
    if dim not in (2, 3) or any(
        not isinstance(p, (Patch, SimplexPatch)) or p.dim != dim or p.dim_phys != dim
        for p in cells
    ):
        raise ValueError("refine_cells requires 2D or 3D volume cells of one dimension")
    for patch in cells:
        require_bezier(patch)
        if not np.isfinite(homogeneous(patch)).all() or np.any(
            np.asarray(patch.weights) <= 0
        ):
            raise ValueError(
                "refinement requires finite control points and positive weights"
            )
    for _ in range(n):
        cells = [child for parent in cells for child in _subdivide(parent)]
    return cells
