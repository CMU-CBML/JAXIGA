"""Rational Bezier triangles and tetrahedra, with conforming C0 connectivity."""

import dataclasses
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from jaxiga.space.simplex import bernstein_simplex, indices


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SimplexPatch:
    """One rational Bezier cell on ``x >= 0, sum(x) <= 1``.

    Control ordering is given by :attr:`multi_indices`. Faces ``f0,...,fd``
    are opposite the corresponding barycentric vertex. Degree elevation
    preserves geometry. Knot insertion and tensor-product refinement do not
    apply to simplex cells; use ``jx.refine_cells`` for midpoint subdivision.
    """

    ctrl_pts: jnp.ndarray
    weights: jnp.ndarray
    degree: tuple = dataclasses.field(metadata=dict(static=True))
    labels: tuple = dataclasses.field(metadata=dict(static=True), default=())

    @classmethod
    def create(cls, degree, ctrl_pts, weights=None, labels=None):
        xyz = np.asarray(ctrl_pts, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] not in (2, 3):
            raise ValueError("simplex control points must have shape (n, 2) or (n, 3)")
        dim = xyz.shape[1]
        if not isinstance(degree, (int, np.integer)) or degree < 1:
            raise ValueError("simplex degree must be a positive integer")
        if len(xyz) != comb(degree + dim, dim):
            raise ValueError("wrong number of simplex control points for this degree")
        w = np.ones(len(xyz)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (len(xyz),) or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("simplex weights must be finite and positive")
        if not np.isfinite(xyz).all():
            raise ValueError("non-finite simplex control points")
        labels = dict(labels or {})
        if any(k not in {f"f{i}" for i in range(dim + 1)} for k in labels):
            raise ValueError("simplex faces are named f0,...,fd (opposite vertex i)")
        return cls(jnp.asarray(xyz), jnp.asarray(w), (int(degree),) * dim,
                   tuple(sorted(labels.items())))

    @classmethod
    def from_vertices(cls, vertices, labels=None):
        """Create an affine cell; vertices must have positive orientation."""
        patch = cls.create(1, vertices, labels=labels)
        vertices = np.asarray(vertices, dtype=float)
        if np.linalg.det(vertices[1:] - vertices[0]) <= 0:
            raise ValueError("simplex vertices must have positive orientation and nonzero volume")
        return patch

    @property
    def dim(self):
        return len(self.degree)

    @property
    def dim_phys(self):
        return self.ctrl_pts.shape[1]

    @property
    def n_cp(self):
        return self.ctrl_pts.shape[0]

    @property
    def multi_indices(self):
        return np.asarray(indices(self.dim, self.degree[0]))

    @property
    def vertex_indices(self):
        a = indices(self.dim, self.degree[0])
        return np.array([a.index(tuple(self.degree[0] if j == i else 0
                                      for j in range(self.dim + 1)))
                         for i in range(self.dim + 1)])

    @property
    def label_map(self):
        return dict(self.labels)

    def with_labels(self, **labels):
        if any(k not in {f"f{i}" for i in range(self.dim + 1)} for k in labels):
            raise ValueError("simplex faces are named f0,...,fd")
        return dataclasses.replace(self, labels=tuple(sorted({**self.label_map, **labels}.items())))

    def elevate(self, target_degree):
        if not isinstance(target_degree, (int, np.integer)) or target_degree < self.degree[0]:
            raise ValueError("target_degree must be an integer at least the current degree")
        p = self.degree[0]
        homogeneous = np.column_stack([np.asarray(self.ctrl_pts) * np.asarray(self.weights)[:, None],
                                       self.weights])
        while p < target_degree:
            lookup = {a: i for i, a in enumerate(indices(self.dim, p))}
            rows = []
            for alpha in indices(self.dim, p + 1):
                row = np.zeros(self.dim + 1)
                for k, value in enumerate(alpha):
                    if value:
                        beta = list(alpha)
                        beta[k] -= 1
                        row += value / (p + 1) * homogeneous[lookup[tuple(beta)]]
                rows.append(row)
            homogeneous = np.asarray(rows)
            p += 1
        return self.create(p, homogeneous[:, :-1] / homogeneous[:, -1:],
                           homogeneous[:, -1], self.label_map)

    def refine(self, n=1):
        raise NotImplementedError("use jx.refine_cells(patches, n) for simplex/mixed cell subdivision")

    def insert_knots(self, *args):
        raise NotImplementedError("simplex Bezier cells have no knot vectors; elevate or remesh")


def evaluate_simplex(patch, parameters):
    B = bernstein_simplex(np.atleast_2d(parameters), patch.degree)[0] * np.asarray(patch.weights)
    return B @ np.asarray(patch.ctrl_pts) / B.sum(axis=1, keepdims=True)


def simplex_topology(patches, tol=1e-10):
    """Identify traces by vertex support and barycentric exponents.

    Unlike merging coincident control points, this cannot merge unrelated
    interior DOFs. Every shared edge/face control point and weight is checked.
    """
    from jaxiga.geometry.multipatch import Interface, NonConformingError, Topology

    dim = patches[0].dim
    vertices = np.concatenate([np.asarray(p.ctrl_pts)[p.vertex_indices] for p in patches])
    parent = np.arange(len(vertices))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(vertices).query_pairs(tol):
        parent[root(b)] = root(a)
    ids = np.array([root(i) for i in range(len(vertices))]).reshape(-1, dim + 1)
    faces, seen_cells, controls, global_ids, mappings = {}, set(), [], {}, []
    for e, (patch, verts) in enumerate(zip(patches, ids)):
        key = tuple(sorted(verts))
        if len(set(verts)) != dim + 1 or key in seen_cells:
            raise NonConformingError("degenerate or duplicate simplex cell")
        seen_cells.add(key)
        for side in range(dim + 1):
            key = tuple(sorted(np.delete(verts, side)))
            faces.setdefault(key, []).append((e, side))
            if len(faces[key]) > 2:
                raise NonConformingError("non-manifold simplex face")
        local = []
        for i, alpha in enumerate(patch.multi_indices):
            key = tuple(sorted((int(v), int(a)) for v, a in zip(verts, alpha) if a))
            data = np.r_[np.asarray(patch.ctrl_pts[i]), float(patch.weights[i])]
            if key in global_ids:
                g = global_ids[key]
                if not np.allclose(controls[g], data, atol=tol, rtol=0):
                    raise NonConformingError("nonconforming simplex trace: control points or weights differ")
            else:
                g = len(controls)
                global_ids[key] = g
                controls.append(data)
            local.append(g)
        mappings.append(np.asarray(local))
    interfaces = []
    for owners in faces.values():
        if len(owners) == 2:
            (a, sa), (b, sb) = owners
            # With positively oriented cells, both opposite vertices must lie
            # on different sides of the shared corner plane.
            va = np.asarray(patches[a].ctrl_pts)[patches[a].vertex_indices]
            vb = np.asarray(patches[b].ctrl_pts)[patches[b].vertex_indices]
            face = np.delete(va, sa, axis=0)
            tangent = face[1] - face[0]
            normal = np.array([-tangent[1], tangent[0]]) if dim == 2 else np.cross(tangent, face[2] - face[0])
            if np.dot(va[sa] - face[0], normal) * np.dot(vb[sb] - face[0], normal) >= 0:
                raise NonConformingError("simplex cells overlap across a shared face")
            interfaces.append(Interface(a, f"f{sa}", b, f"f{sb}"))
    return Topology(tuple(mappings), len(controls), tuple(interfaces))


def build_simplex_space(patches, vec):
    from jaxiga.space.function_space import (
        BoundarySet, StaticArray, StaticPatches, _global_control_data, _new_space,
    )

    topology = simplex_topology(patches)
    dim, degree = patches[0].dim, patches[0].degree
    n, local = len(patches), patches[0].n_cp
    internal = {(i.patch_a, i.side_a) for i in topology.interfaces}
    internal |= {(i.patch_b, i.side_b) for i in topology.interfaces}
    collected = {}
    for e, patch in enumerate(patches):
        for side in range(dim + 1):
            name = f"f{side}"
            if (e, name) in internal:
                continue
            label = patch.label_map.get(name, f"patch{e}/{name}")
            entry = collected.setdefault(label, [[], [], []])
            entry[0].append(e)
            entry[1].append(side)
            entry[2].extend(topology.local_to_global[e][patch.multi_indices[:, side] == 0])
    boundaries = {k: BoundarySet(np.asarray(v[0]), np.asarray(v[1]), np.unique(v[2]))
                  for k, v in collected.items()}
    cpts, wgts = _global_control_data(patches, topology)
    return _new_space(
        extraction=jnp.asarray(np.broadcast_to(np.eye(local), (n, local, local))),
        cpts=jnp.asarray(cpts), wgts=jnp.asarray(wgts),
        elem_vertex=np.tile(np.r_[np.zeros(dim), np.ones(dim)], (n, 1)),
        elem_dofs=StaticArray(np.stack(topology.local_to_global)),
        elem_patch=StaticArray(np.arange(n)), boundaries=boundaries, topology=topology,
        dim=dim, vec=int(vec), degree=degree, n_scalar_basis=topology.n_global_dofs,
        n_elems=n, patches=StaticPatches(patches), cell_type="simplex",
    )
