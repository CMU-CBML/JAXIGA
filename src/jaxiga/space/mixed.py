"""C0 mixed quad/triangle and hex/tetrahedron spaces through trace constraints.

A homogeneous constraint elimination expresses local coefficients in terms of
independent global coefficients. Element extraction includes that map; the
existing rational evaluator and assembly kernels therefore need no multipliers.
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from jaxiga.geometry.mixed import (
    corners,
    homogeneous,
    ordered_interfaces,
    outward_normal,
    require_bezier,
    trace_transfer,
)
from jaxiga.geometry.simplex import SimplexPatch
from jaxiga.geometry.multipatch import Interface, Topology
from jaxiga.geometry.gmsh_geometry import tensor_grid, tensor_solve
from jaxiga.space.bernstein import bernstein_basis


def _eliminate(n, rows):
    pivots = {}
    for coefficients in rows:
        row = {i: float(v) for i, v in coefficients.items() if abs(v) > 1e-11}
        while row:
            pivot = max(row)
            value = row[pivot]
            if pivot not in pivots:
                pivots[pivot] = {i: -v / value for i, v in row.items() if i != pivot}
                break
            row.pop(pivot)
            for i, v in pivots[pivot].items():
                row[i] = row.get(i, 0) + value * v
                if abs(row[i]) < 1e-11:
                    row.pop(i)
    free = np.array([i for i in range(n) if i not in pivots])
    columns = {i: k for k, i in enumerate(free)}
    expanded = []
    rr, cc, vv = [], [], []
    for i in range(n):
        if i in columns:
            row = {columns[i]: 1.0}
        else:
            row = {}
            for j, v in pivots[i].items():
                for k, w in expanded[j].items():
                    row[k] = row.get(k, 0) + v * w
            row = {k: v for k, v in row.items() if abs(v) > 1e-11}
        expanded.append(row)
        for k, v in row.items():
            rr.append(i)
            cc.append(k)
            vv.append(v)
    return free, csr_matrix((vv, (rr, cc)), shape=(n, len(free)))


@lru_cache(64)
def _embedding(dim, degree, source_degree, simplex):
    from jaxiga.space.simplex import bernstein_simplex
    from jaxiga.space.bernstein import bernstein_tensor

    nodes = np.linspace(0, 1, degree + 1)
    pts = tensor_grid([nodes] * dim)
    B = (
        bernstein_simplex(pts, source_degree)[0]
        if simplex
        else bernstein_tensor(2 * pts - 1, source_degree)[0]
    )
    matrix = bernstein_basis(2 * nodes - 1, degree)
    return tensor_solve([matrix] * dim, B).T


def build_mixed_space(patches, vec):
    from jaxiga.space.function_space import (
        BoundarySet,
        StaticArray,
        StaticPatches,
        _new_space,
    )

    dim = patches[0].dim
    if any(p.dim != dim for p in patches):
        raise ValueError("mixed cells must have the same dimension")
    for p in patches:
        require_bezier(p)
    offsets = np.r_[0, np.cumsum([p.n_cp for p in patches])]
    pairs, boundary = ordered_interfaces(patches)
    rows = []
    interfaces = []
    for a, b in pairs:
        master, slave = patches[a.cell], patches[b.cell]
        if len(a.vertices) == 4 and len(b.vertices) == 3:
            axis = "uvw".index(a.side[0])
            needed = sum(p for d, p in enumerate(master.degree) if d != axis)
            if slave.degree[0] < needed:
                raise ValueError(
                    f"tetrahedron face needs degree >= {needed} to match this hex trace; elevate it first"
                )
        if np.dot(outward_normal(master, a), outward_normal(slave, b)) > -0.1:
            raise ValueError(
                "mixed cells overlap rather than meet on opposite sides of a face"
            )
        T = trace_transfer(master, a, slave, b)
        hm, hs = homogeneous(master), homogeneous(slave)[b.local]
        target = T @ hm
        ratio = hs[0, -1] / target[0, -1]
        T *= ratio
        target *= ratio
        if not np.allclose(target, hs, rtol=1e-9, atol=1e-9):
            raise ValueError(
                "nonconforming mixed rational trace: geometry/weights differ; reconstruct compatible traces first"
            )
        for k, local in enumerate(b.local):
            row = {
                int(offsets[a.cell] + j): -v
                for j, v in enumerate(T[k])
                if abs(v) > 1e-11
            }
            idx = int(offsets[b.cell] + local)
            row[idx] = row.get(idx, 0) + 1
            rows.append(row)
        interfaces.append(Interface(a.cell, a.side, b.cell, b.side))
    # Shared vertices also join cells which touch only at a point.
    from scipy.spatial import cKDTree

    vertex_ids = np.concatenate(
        [offsets[e] + corners(p)[1] for e, p in enumerate(patches)]
    )
    H = np.concatenate([homogeneous(p) for p in patches])
    xyz = H[vertex_ids, :dim] / H[vertex_ids, -1:]
    for a, b in cKDTree(xyz).query_pairs(1e-10):
        i, j = map(int, vertex_ids[[a, b]])
        rows.append({i: 1 / H[i, -1], j: -1 / H[j, -1]})
    free, P = _eliminate(len(H), rows)
    if not np.allclose(P @ H[free], H, rtol=2e-9, atol=2e-9):
        raise ValueError("mixed constraints are inconsistent with the geometry")
    degree = max(max(p.degree) for p in patches)
    maps, operators = [], []
    for e, p in enumerate(patches):
        block = P[offsets[e] : offsets[e + 1]]
        support = np.unique(block.indices)
        maps.append(support)
        base = _embedding(dim, degree, p.degree, isinstance(p, SimplexPatch))
        operators.append(block[:, support].T @ base)
    width = max(map(len, maps))
    dofs = np.full((len(patches), width), len(free), dtype=int)
    extraction = np.zeros((len(patches), width, (degree + 1) ** dim))
    for e, (support, operator) in enumerate(zip(maps, operators)):
        dofs[e, : len(support)] = support
        extraction[e, : len(support)] = operator
    collected = {}
    for f in boundary:
        label = patches[f.cell].label_map.get(f.side, f"patch{f.cell}/{f.side}")
        entry = collected.setdefault(label, [[], [], []])
        entry[0].append(f.cell)
        entry[1].append(f.code)
        entry[2].extend(P[offsets[f.cell] + f.local].indices)
    boundaries = {
        k: BoundarySet(np.asarray(v[0]), np.asarray(v[1]), np.unique(v[2]))
        for k, v in collected.items()
    }
    # In a constrained space a patch has a restriction matrix, not a direct
    # local-to-global numbering; elem_dofs/extraction carry the full relation.
    topology = Topology(
        tuple(np.asarray(m) for m in maps), len(free), tuple(interfaces)
    )
    return _new_space(
        extraction=jnp.asarray(extraction),
        cpts=jnp.asarray(H[free, :dim] / H[free, -1:]),
        wgts=jnp.asarray(H[free, -1]),
        elem_dofs=StaticArray(dofs),
        elem_patch=StaticArray(np.arange(len(patches))),
        elem_vertex=np.tile(np.r_[np.zeros(dim), np.ones(dim)], (len(patches), 1)),
        boundaries=boundaries,
        topology=topology,
        cell_type="mixed",
        element_types=StaticArray([int(isinstance(p, SimplexPatch)) for p in patches]),
        dim=dim,
        vec=int(vec),
        degree=(degree,) * dim,
        n_scalar_basis=len(free),
        n_elems=len(patches),
        patches=StaticPatches(patches),
    )


def pointset(space, n, target=None, sampling=False):
    from jaxiga.space.pointset import PointSet, _tensor_rule
    from jaxiga.space.simplex import gauss_rule, face_points, lattice

    dim = space.dim
    kinds = np.asarray(space.element_types)
    if target is None:
        elems = np.arange(space.n_elems)
        sides = None
        tag = "grid" if sampling else "interior"
    else:
        bset = space.boundary(target) if isinstance(target, str) else target
        elems, sides = np.asarray(bset.elems), np.asarray(bset.sides)
        tag = f"boundary:{target if isinstance(target, str) else 'set'}"
    dimension = dim if sides is None else dim - 1
    if sampling:
        tensor = tensor_grid([np.linspace(-1, 1, n)] * dim)
        simplex = 2 * lattice(dim, n - 1) - 1
        rules = [(tensor, None), (simplex, None)]
    else:
        tensor, tw = _tensor_rule([n] * dimension)
        simplex, sw = gauss_rule(dimension, n)
        rules = [(tensor, tw), (2 * simplex - 1, sw * 2**dimension)]
    refs, weights = [], []
    count = max(len(r[0]) for r in rules)
    for i, e in enumerate(elems):
        pts, w = rules[kinds[e]]
        if sides is not None:
            side = int(sides[i])
            if kinds[e]:
                pts = 2 * face_points(dim, side, (pts + 1) / 2) - 1
            else:
                d, end = divmod(side, 2)
                full = np.full((len(pts), dim), 2 * end - 1, dtype=float)
                full[:, np.arange(dim) != d] = pts
                pts = full
        refs.append(np.concatenate([pts, np.repeat(pts[:1], count - len(pts), axis=0)]))
        if w is not None:
            weights.append(np.pad(w, (0, count - len(w))))
    return PointSet(
        elems,
        np.asarray(refs),
        None if sampling else np.asarray(weights),
        sides=sides,
        tag=tag,
    )
