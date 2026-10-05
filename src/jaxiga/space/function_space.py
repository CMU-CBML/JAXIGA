"""FunctionSpace and DofMap.

Setup stage builds all connectivity in NumPy; the resulting object is a pytree
whose *leaves* are the float arrays that enter arithmetic (extraction
operators, control points, weights) plus the integer index arrays.

On staticness
-------------
The architecture note wants index arrays to be static/aux pytree data. JAX
rejects bare arrays there -- metadata must be hashable -- so connectivity is
wrapped in :class:`StaticArray`, which supplies a content-based hash. That
keeps it genuinely static: concrete even inside ``jit``, and part of the
compilation cache key, which is what connectivity should be.

This matters because since JAX 0.11 a ``jnp.asarray`` of a NumPy array inside
a trace yields a *tracer*, so anything stored as a leaf becomes traced and can
no longer be used for host-side indexing.

Float arrays that enter arithmetic -- extraction operators, control points,
weights, element boxes -- are ordinary leaves, which is what makes shape
derivatives work.

The padded-element contract
---------------------------
Tensor-product elements all carry the same ``(p+1)^dim`` basis functions, but
hierarchical (THB) elements do not, so ``elem_dofs`` and ``extraction`` are
sized by ``n_local_max`` and short elements are padded:

* the padding entry of ``elem_dofs`` is the reserved index
  ``n_scalar_basis`` -- one past the last real scalar dof;
* the matching rows of ``extraction`` are **zero**, so every quantity the
  padded function contributes is identically zero.

Consumers gather through the one-longer arrays returned by
:func:`padded_control_data` and :func:`pad_coeffs`, whose extra entry is inert
precisely because those extraction rows vanish. A tensor-product space sets
``n_local_max == n_local`` and never emits the padding index, so the padded
arrays change nothing about its results.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.config import TOL
from jaxiga.geometry.multipatch import Topology, compute_topology, side_indices
from jaxiga.geometry.simplex import SimplexPatch
from jaxiga.geometry.nurbs import Patch, side_code, side_name
from jaxiga.space.bernstein import bezier_extraction, element_spans


class StaticArray:
    """A hashable, immutable NumPy array usable as pytree metadata.

    JAX requires metadata fields to be hashable and rejects bare arrays, but
    connectivity genuinely *is* static: it fixes gather/scatter patterns and
    must stay concrete even inside ``jit``. (Since JAX 0.11, ``jnp.asarray`` of
    a NumPy array inside a trace yields a tracer, so storing connectivity as a
    pytree leaf makes it traced and unusable for indexing.)

    Hashing is by content, so two structurally identical spaces share a
    compilation cache entry. The array is frozen to keep that hash honest.
    """

    __slots__ = ("array", "_hash")

    def __init__(self, array):
        arr = np.array(array, copy=True)
        arr.flags.writeable = False
        self.array = arr
        self._hash = hash((arr.shape, arr.dtype.str, arr.tobytes()))

    def __array__(self, dtype=None, copy=None):
        return self.array if dtype is None else self.array.astype(dtype)

    def __getitem__(self, key):
        return self.array[key]

    def __len__(self):
        return len(self.array)

    @property
    def shape(self):
        return self.array.shape

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, StaticArray):
            return self._hash == other._hash and np.array_equal(self.array, other.array)
        return NotImplemented

    def __repr__(self):
        return f"StaticArray(shape={self.array.shape}, dtype={self.array.dtype})"


class StaticPatches:
    """A hashable, immutable tuple of patches usable as pytree metadata.

    The space keeps the geometry it was built from so that operations needing
    the patches -- hierarchical refinement above all -- do not have to
    reconstruct them from connectivity. Hashing is by content, matching
    :class:`StaticArray`.
    """

    __slots__ = ("patches", "_hash")

    def __init__(self, patches):
        self.patches = tuple(patches)
        self._hash = hash(
            tuple(
                (
                    (type(p).__name__, getattr(p, "knots", ())),
                    p.degree,
                    p.labels,
                    np.asarray(p.ctrl_pts).tobytes(),
                    np.asarray(p.weights).tobytes(),
                )
                for p in self.patches
            )
        )

    def __iter__(self):
        return iter(self.patches)

    def __getitem__(self, i):
        return self.patches[i]

    def __len__(self):
        return len(self.patches)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, StaticPatches):
            return self._hash == other._hash
        return NotImplemented

    def __repr__(self):
        return f"StaticPatches({len(self.patches)} patches)"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BoundarySet:
    """Elements, sides and dofs making up one labeled piece of the boundary."""

    elems: np.ndarray  # (n_b,) element indices
    sides: np.ndarray  # (n_b,) side codes, 2*direction + end
    dofs: np.ndarray  # (n_d,) global SCALAR basis indices, sorted

    def __len__(self):
        return len(self.elems)


class _SpaceMeta(type):
    """Lets ``FunctionSpace(patches, vec=2)`` build from geometry.

    The user-facing call takes patches (the FEniCSx-style spelling the guide
    specifies), while the dataclass constructor takes the assembled fields.
    Dispatch on the first argument keeps both without splitting the name, so
    ``isinstance(V, FunctionSpace)`` still works.
    """

    def __call__(cls, *args, **kwargs):
        first = args[0] if args else kwargs.get("patches")
        is_patches = isinstance(first, (Patch, SimplexPatch)) or (
            isinstance(first, Sequence) and len(first) > 0 and isinstance(first[0], (Patch, SimplexPatch))
        )
        if is_patches:
            return build_function_space(*args, **kwargs)
        return super().__call__(*args, **kwargs)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FunctionSpace(metaclass=_SpaceMeta):
    """Global NURBS basis over one or more conforming patches."""

    # ---- dynamic leaves ----
    extraction: jnp.ndarray  # (n_elems, n_local_max, n_bernstein)
    cpts: jnp.ndarray  # (n_scalar_basis, dim_phys)
    wgts: jnp.ndarray  # (n_scalar_basis,)
    elem_vertex: np.ndarray  # (n_elems, 2*dim) as [mins..., maxs...]
    boundaries: dict  # label -> BoundarySet
    topology: Topology

    # ---- static metadata ----
    cell_type: str = dataclasses.field(metadata=dict(static=True), default="tensor")
    collapse_degenerate: bool = dataclasses.field(metadata=dict(static=True), default=False)
    element_types: StaticArray = dataclasses.field(metadata=dict(static=True), default=None)
    dim: int = dataclasses.field(metadata=dict(static=True), default=2)
    vec: int = dataclasses.field(metadata=dict(static=True), default=1)
    degree: tuple = dataclasses.field(metadata=dict(static=True), default=())
    n_scalar_basis: int = dataclasses.field(metadata=dict(static=True), default=0)
    n_elems: int = dataclasses.field(metadata=dict(static=True), default=0)
    # Per-patch knot vectors, kept for constructions that need multiplicities
    # (Greville abscissae) rather than just the element boxes. Nested tuples of
    # floats so the metadata stays hashable.
    patch_knots: tuple = dataclasses.field(metadata=dict(static=True), default=())
    # Connectivity: static metadata, so it stays concrete inside jit.
    elem_dofs: StaticArray = dataclasses.field(metadata=dict(static=True), default=None)
    elem_patch: StaticArray = dataclasses.field(metadata=dict(static=True), default=None)
    # Hierarchical (THB) refinement state, or None for a tensor-product space.
    # Static: it fixes connectivity, and repeated refinement reads it back.
    hierarchy: object = dataclasses.field(metadata=dict(static=True), default=None)
    # The geometry this space was built from, kept so refinement need not
    # reconstruct it, and (n_elems, 3) keys (patch, level, level-local index)
    # naming where each element sits in the hierarchy.
    patches: StaticPatches = dataclasses.field(metadata=dict(static=True), default=None)
    elem_key: StaticArray = dataclasses.field(metadata=dict(static=True), default=None)

    @property
    def n_bernstein(self) -> int:
        """Bernstein polynomials on the reference element, ``(p+1)^dim``.

        The column count of :attr:`extraction`, and the same for every element
        of every space of this degree.
        """
        return int(self.extraction.shape[-1])

    @property
    def n_local(self) -> int:
        """Scalar basis functions gathered per element, including padding.

        Equal to :attr:`n_bernstein` on a tensor-product space; on a
        hierarchical space it is ``n_local_max`` (see the module docstring).
        """
        if self.elem_dofs is None:
            return self.n_bernstein
        return int(self.elem_dofs.shape[1])

    @property
    def is_hierarchical(self) -> bool:
        return self.hierarchy is not None

    @property
    def n_dofs(self) -> int:
        return self.n_scalar_basis * self.vec

    @property
    def dim_phys(self) -> int:
        return self.cpts.shape[1]

    def elem_dofs_vec(self) -> np.ndarray:
        """(n_elems, n_local*vec) interleaved vector dofs."""
        return expand_dofs(self.elem_dofs, self.vec)

    def boundary(self, label: str) -> BoundarySet:
        if label not in self.boundaries:
            raise KeyError(
                f"unknown boundary label {label!r}; available: {sorted(self.boundaries)}"
            )
        return self.boundaries[label]

    def summary(self) -> str:
        kind = ""
        if self.is_hierarchical:
            kind = (
                f", levels={self.hierarchy.n_levels}, "
                f"n_local_max={self.n_local} (tensor-product: {self.n_bernstein})"
            )
        return (
            f"FunctionSpace(dim={self.dim}, vec={self.vec}, degree={self.degree}, "
            f"n_elems={self.n_elems}, n_scalar_basis={self.n_scalar_basis}, "
            f"n_dofs={self.n_dofs}, boundaries={sorted(self.boundaries)}{kind})"
        )


def _new_space(**fields) -> FunctionSpace:
    """Construct the dataclass directly, bypassing the patches dispatch."""
    return super(_SpaceMeta, _SpaceMeta).__call__(FunctionSpace, **fields)


def padded_control_data(space):
    """``(cpts, wgts)`` extended by one inert entry for the padding dof.

    The padding index is ``space.n_scalar_basis``, so it lands exactly on the
    appended row. Its values never reach a result -- the zero extraction rows
    under the padding annihilate them -- but they must be *finite*, and the
    weight must be non-zero so no rational quotient sees 0/0 during tracing.
    """
    cpts = jnp.concatenate([space.cpts, jnp.zeros((1, space.dim_phys), space.cpts.dtype)])
    wgts = jnp.concatenate([space.wgts, jnp.ones(1, space.wgts.dtype)])
    return cpts, wgts


def pad_coeffs(u, vec: int):
    """Extend a dof vector so gathers at the padding dof are inert."""
    return jnp.concatenate([u, jnp.zeros(vec, u.dtype)])


def expand_dofs(scalar_dofs, vec: int) -> np.ndarray:
    """Scalar dof indices -> interleaved vector dof indices.

    Component ``c`` of scalar basis ``i`` is global dof ``vec*i + c``, matching
    the legacy ``make_global_nodes_xy`` interleaving.

    ``(..., n) -> (..., n*vec)``.
    """
    scalar_dofs = np.asarray(scalar_dofs)
    if vec == 1:
        return scalar_dofs
    out = vec * scalar_dofs[..., :, None] + np.arange(vec)
    return out.reshape(*scalar_dofs.shape[:-1], -1)


def _tensor_extraction(local_ops):
    """Kronecker product of per-direction operators, first direction fastest.

    Matches ``np.kron(C_v[j], C_u[i])`` in ``IGAMesh2D``.
    """
    C = local_ops[-1]
    for d in range(len(local_ops) - 2, -1, -1):
        C = np.kron(C, local_ops[d])
    return C


def _patch_connectivity(patch: Patch):
    """Per-patch IEN, element boxes and Bezier operators.

    Dimension-generic port of ``IGAMesh2D.makeIEN`` plus the extraction
    assembly in ``IGAMesh2D.__init__``.
    """
    dim = patch.dim
    knots = patch.knot_arrays()
    degree = patch.degree
    n_per_dir = patch.n_cp_per_dir
    strides = np.array([1, *np.cumprod(n_per_dir[:-1])], dtype=int)

    per_dir = [element_spans(knots[d], degree[d]) for d in range(dim)]
    extraction_1d = [bezier_extraction(knots[d], degree[d])[0] for d in range(dim)]
    n_elem_per_dir = [len(spans[0]) for spans in per_dir]
    n_elem = int(np.prod(n_elem_per_dir))

    n_local = int(np.prod([p + 1 for p in degree]))
    ien = np.zeros((n_elem, n_local), dtype=int)
    boxes = np.zeros((n_elem, 2 * dim))
    ops = np.zeros((n_elem, n_local, n_local))

    # Element numbering: first direction fastest, as in the legacy index_matrix.
    for e in range(n_elem):
        idx, rem = [], e
        for d in range(dim):
            idx.append(rem % n_elem_per_dir[d])
            rem //= n_elem_per_dir[d]

        for d in range(dim):
            lo, hi, _ = per_dir[d]
            boxes[e, d] = lo[idx[d]]
            boxes[e, dim + d] = hi[idx[d]]

        ranges = [
            np.arange(per_dir[d][2][idx[d]], per_dir[d][2][idx[d]] + degree[d] + 1)
            for d in range(dim)
        ]
        mesh = np.meshgrid(*ranges, indexing="ij")
        ien[e] = sum(m.ravel(order="F") * strides[d] for d, m in enumerate(mesh))

        ops[e] = _tensor_extraction([extraction_1d[d][idx[d]] for d in range(dim)])

    return ien, boxes, ops, n_elem_per_dir


def build_function_space(patches, vec: int = 1, *, collapse_degenerate=False) -> FunctionSpace:
    """Build a function space over one or more conforming patches.

    Reached through ``FunctionSpace(patches, vec=...)``.

    Parameters
    ----------
    patches : Patch, SimplexPatch, or a sequence of patches
        Single-family meshes require a common degree and conforming traces.
        Mixed tensor/simplex meshes use Bezier cells with compatible rational
        traces and sufficient simplex degree to represent tensor face traces.
    vec : int
        Solution components per basis function (JAX-FEM naming). Global dof of
        component ``c`` of scalar basis ``i`` is ``vec*i + c``.
    collapse_degenerate : bool
        Tie equal-weight coefficients along tensor boundary faces that collapse
        in a tangential direction. This makes fields single-valued on collapsed
        edges/poles. It does not regularize singular Jacobians at those points.
    """
    if isinstance(patches, (Patch, SimplexPatch)):
        patches = [patches]
    patches = list(patches)
    if not patches:
        raise ValueError("at least one patch is required")
    if collapse_degenerate and any(isinstance(p, SimplexPatch) for p in patches):
        raise NotImplementedError("collapsed boundaries currently require tensor patches")

    if any(isinstance(p, SimplexPatch) != isinstance(patches[0], SimplexPatch) for p in patches):
        from jaxiga.space.mixed import build_mixed_space
        return build_mixed_space(patches, vec)
    degree = patches[0].degree
    dim = patches[0].dim
    for i, p in enumerate(patches):
        if p.degree != degree:
            raise ValueError(
                f"all patches must share a degree (v1 limitation): patch 0 has "
                f"{degree}, patch {i} has {p.degree}. Elevate them to a common degree."
            )
        if p.dim != dim:
            raise ValueError(f"patch {i} has dim {p.dim}, expected {dim}")

    if isinstance(patches[0], SimplexPatch):
        from jaxiga.geometry.simplex import build_simplex_space
        return build_simplex_space(patches, vec)

    topology = compute_topology(patches, collapse_degenerate=collapse_degenerate)

    ien_all, boxes_all, ops_all, patch_id = [], [], [], []
    n_elem_per_dir = []
    for i, patch in enumerate(patches):
        ien, boxes, ops, nepd = _patch_connectivity(patch)
        ien_all.append(topology.local_to_global[i][ien])
        boxes_all.append(boxes)
        ops_all.append(ops)
        patch_id.append(np.full(len(ien), i))
        n_elem_per_dir.append(nepd)

    elem_dofs = np.concatenate(ien_all, axis=0)
    elem_vertex = np.concatenate(boxes_all, axis=0)
    extraction = np.concatenate(ops_all, axis=0)
    elem_patch = np.concatenate(patch_id, axis=0)

    cpts, wgts = _global_control_data(patches, topology)
    boundaries = _classify_boundaries(patches, topology, n_elem_per_dir, elem_dofs)

    return _new_space(
        extraction=jnp.asarray(extraction),
        cpts=jnp.asarray(cpts),
        wgts=jnp.asarray(wgts),
        elem_dofs=StaticArray(elem_dofs),
        elem_patch=StaticArray(elem_patch),
        elem_vertex=elem_vertex,
        boundaries=boundaries,
        topology=topology,
        dim=dim,
        collapse_degenerate=bool(collapse_degenerate),
        vec=int(vec),
        degree=degree,
        n_scalar_basis=topology.n_global_dofs,
        n_elems=len(elem_dofs),
        patch_knots=tuple(p.knots for p in patches),
        patches=StaticPatches(patches),
    )


def _global_control_data(patches, topology):
    """Gather control points/weights into global scalar-dof order.

    Interface points are written once. Conforming patches make the duplicates
    identical by construction; that is asserted rather than assumed.
    """
    n = topology.n_global_dofs
    dim_phys = patches[0].dim_phys
    cpts = np.full((n, dim_phys), np.nan)
    wgts = np.full(n, np.nan)

    for i, patch in enumerate(patches):
        g = topology.local_to_global[i]
        local_cpts = np.asarray(patch.ctrl_pts)
        local_wgts = np.asarray(patch.weights)

        seen = ~np.isnan(wgts[g])
        if np.any(seen):
            dev_c = np.abs(cpts[g][seen] - local_cpts[seen]).max(initial=0.0)
            dev_w = np.abs(wgts[g][seen] - local_wgts[seen]).max(initial=0.0)
            if max(dev_c, dev_w) > TOL:
                raise ValueError(
                    f"patch {i} disagrees with an earlier patch about shared control "
                    f"data by {max(dev_c, dev_w):.3e} (tolerance {TOL:.1e})"
                )
        cpts[g] = local_cpts
        wgts[g] = local_wgts

    if np.any(np.isnan(wgts)):
        raise ValueError("internal error: some global dofs received no control data")
    return cpts, wgts


def _classify_boundaries(patches, topology, n_elem_per_dir, elem_dofs):
    """Group boundary elements/sides/dofs by label.

    Every side of every patch contributes to its label; unlabeled sides get the
    automatic label ``"patch{i}/{side}"``. Sides that turn out to be interior
    (glued to another patch) are skipped.
    """
    interface_sides = set()
    for iface in topology.interfaces:
        interface_sides.add((iface.patch_a, iface.side_a))
        interface_sides.add((iface.patch_b, iface.side_b))

    collected = {}
    elem_offset = 0
    for i, patch in enumerate(patches):
        dim = patch.dim
        nepd = n_elem_per_dir[i]
        n_elem = int(np.prod(nepd))
        labels = patch.label_map

        for d in range(dim):
            for end in (0, 1):
                side = side_name(d, end)
                if (i, side) in interface_sides:
                    continue

                label = labels.get(side, f"patch{i}/{side}")
                elems = _side_elements(nepd, d, end, dim) + elem_offset
                local_dofs = side_indices(patch, side)
                dofs = topology.local_to_global[i][local_dofs]

                entry = collected.setdefault(label, {"elems": [], "sides": [], "dofs": []})
                entry["elems"].append(elems)
                entry["sides"].append(np.full(len(elems), side_code(side)))
                entry["dofs"].append(dofs)

        elem_offset += n_elem

    return {
        label: BoundarySet(
            elems=np.concatenate(v["elems"]).astype(int),
            sides=np.concatenate(v["sides"]).astype(int),
            dofs=np.unique(np.concatenate(v["dofs"])).astype(int),
        )
        for label, v in collected.items()
    }


def _side_elements(n_elem_per_dir, direction, end, dim):
    """Element indices touching one side of a patch (element numbering: dir 0 fastest)."""
    strides = np.array([1, *np.cumprod(n_elem_per_dir[:-1])], dtype=int)
    fixed = 0 if end == 0 else n_elem_per_dir[direction] - 1

    ranges = [np.arange(n) for n in n_elem_per_dir]
    ranges[direction] = np.array([fixed])
    mesh = np.meshgrid(*ranges, indexing="ij")
    return sum(m.ravel(order="F") * strides[d] for d, m in enumerate(mesh)).astype(int)


# --------------------------------------------------------------------------
# DofMap
# --------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DofMap:
    """Partition of the global dofs into free and prescribed.

    The single home of constrained-dof logic, replacing the legacy
    ``get_bcdof_bcval``, the row surgery in ``applyBCElast2D``, and the
    ``trainable_indx`` bookkeeping in the energy-minimization solvers.

    ``values`` is a traced leaf, so Dirichlet data can depend on ``params`` and
    gradients flow back to it.
    """

    free: np.ndarray
    prescribed: np.ndarray
    values: jnp.ndarray
    n_dofs: int = dataclasses.field(metadata=dict(static=True), default=0)

    @property
    def n_free(self) -> int:
        return len(self.free)

    def lift(self, u_free: jnp.ndarray) -> jnp.ndarray:
        """Free dofs -> full dof vector, inserting the prescribed values."""
        return (
            jnp.zeros(self.n_dofs, dtype=u_free.dtype)
            .at[self.free]
            .set(u_free)
            .at[self.prescribed]
            .set(self.values)
        )

    def restrict(self, v_full: jnp.ndarray) -> jnp.ndarray:
        """Full dof vector -> free dofs."""
        return v_full[self.free]

    def zeros(self) -> jnp.ndarray:
        return jnp.zeros(self.n_free)


def make_dofmap(n_dofs: int, prescribed, values) -> DofMap:
    """Build a DofMap from prescribed dof indices and their values."""
    prescribed = np.asarray(prescribed, dtype=int)
    order = np.argsort(prescribed, kind="stable")
    prescribed = prescribed[order]
    values = (jnp.asarray(values)[order] if isinstance(values, jnp.ndarray)
              else np.asarray(values)[order]) if len(prescribed) else np.zeros(0)

    free = np.setdiff1d(np.arange(n_dofs), prescribed)
    return DofMap(free=free, prescribed=prescribed, values=values, n_dofs=int(n_dofs))
