"""Local refinement with truncated hierarchical B-splines (THB).

Setup stage, pure NumPy/SciPy. The output is an ordinary
:class:`~jaxiga.space.function_space.FunctionSpace`: everything downstream --
evaluation, assembly, all three methods, ``Solution``, VTK -- consumes only the
four-array contract ``(elem_dofs, extraction, cpts, wgts)``, and THB admits
exactly that representation via multi-level Bezier extraction. Local refinement
therefore lands as one new constructor plus the padding convention documented
in :mod:`jaxiga.space.function_space`; no compute-path code changes.

References
----------
Giannelli, Juttler, Speleers, *THB-splines: the truncated basis for
hierarchical splines*, CAGD 29 (2012); Vuong, Giannelli, Juttler, Simeon, *A
hierarchical approach to adaptive local refinement in isogeometric analysis*,
CMAME 200 (2011); D'Angella, Kollmannsberger, Rank, Reali, *Multi-level Bezier
extraction for hierarchical local refinement of IGA*, CMAME 328 (2018).

How it works
------------
Each patch carries a stack of dyadically refined levels. Refinement replaces a
marked leaf element by its ``2^dim`` children one level down; a level-m basis
function is *active* when its support lies inside the level-m domain and still
covers at least one level-m leaf.

Truncation is carried out per level as sparse matrices ``Q_l`` whose columns
are the level-l B-spline coefficients of each active function:

    Q_0   = e_j for each active level-0 function
    Q_l   = S_{l-1} Q_{l-1},  rows of active level-l functions zeroed
            (this *is* the truncation), then a fresh column e_j appended for
            each active level-l function.

For a leaf element at level l the supported functions are the columns of
``Q_l`` with a non-zero entry among that element's ``(p+1)^dim`` level-l
B-splines, and the extraction row of such a function is its coefficient vector
composed with the level-l Bezier extraction of the element. Truncating further
than l cannot change anything on that element, because deeper levels only touch
functions supported inside the refined region.

Why nothing tensor-product is ever built
----------------------------------------
Level l has ``2^l`` times as many elements per direction as level 0, so an
adapted mesh ten levels deep would have a *virtual* finest grid of millions of
elements even when only a handful are alive. Materializing per-level element
grids, control nets or global two-scale operators is therefore not an option.

Everything expensive here is kept strictly one-dimensional -- knot vectors,
Bezier operators, and two-scale matrices, all per direction -- while the
multi-dimensional bookkeeping (which elements exist, which functions are
active) is held as *sets of flat indices* covering only the live region. Even
the control point of a deep active function is computed on demand, by pushing
the level-0 homogeneous net through the composed 1D refinement operators, whose
rows carry at most ``p+1`` entries. The cost of a level therefore scales with
what is alive on it, not with ``2^(l*dim)``.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from jaxiga.config import TOL
from jaxiga.geometry.multipatch import NonConformingError, Topology, compute_topology
from jaxiga.geometry.nurbs import Patch, evaluate_patch, side_code, side_name
from jaxiga.space.bernstein import bezier_extraction, element_spans
from jaxiga.space.function_space import (
    BoundarySet,
    FunctionSpace,
    StaticArray,
    StaticPatches,
    _new_space,
    _tensor_extraction,
)

# A hierarchical element gathers more functions than a tensor-product one, and
# assembly cost grows with the square of that count. Admissibility keeps it
# close to (p+1)^dim; well past that means the marking produced a mesh this
# implementation should refuse rather than silently make expensive.
MAX_LOCAL_GROWTH = 3.0

# Coefficients below this are treated as truncated away. Truncation is exact
# arithmetic on well-scaled numbers, so anything at this level is round-off,
# and keeping it would inflate the support of coarse functions.
COEFF_TOL = 1e-13


class HierarchyError(Exception):
    """The requested refinement cannot be represented."""


@dataclasses.dataclass(frozen=True)
class Hierarchy:
    """Which elements have been refined, per patch and per level.

    ``refined[i][l]`` is a sorted tuple of level-``l`` element indices of patch
    ``i`` that were replaced by their children. Everything else about a
    hierarchical space is derived from this and the level-0 patches, which is
    what makes the state hashable and therefore usable as pytree metadata.
    """

    refined: tuple  # per patch: tuple over levels of tuple-of-int
    truncate: bool = True
    admissible: bool = True

    @property
    def n_patches(self) -> int:
        return len(self.refined)

    @property
    def n_levels(self) -> int:
        return 1 + max((len(r) for r in self.refined), default=0)

    def n_refined(self) -> int:
        return sum(len(lv) for r in self.refined for lv in r)


# --------------------------------------------------------------------------
# flat-index helpers
# --------------------------------------------------------------------------


def _ravel(idx, dims):
    return np.ravel_multi_index(idx, dims, order="F")


def _unravel(flat, dims):
    return np.unravel_index(np.asarray(flat, dtype=np.int64), dims, order="F")


def _tensor_indices(ranges, dims):
    """Flat indices of the tensor product of per-direction index ranges.

    The first direction varies fastest, matching the local numbering of
    ``_tensor_extraction`` and of the tensor-product IEN.

    Built by accumulating outer sums of the per-direction strides rather than
    through ``meshgrid``, which broadcasts and copies one full array per
    direction. This is called once per element, so its constant matters.
    """
    strides = np.concatenate([[1], np.cumprod(np.asarray(dims[:-1], dtype=np.int64))])
    flat = np.asarray(ranges[0], dtype=np.int64) * strides[0]
    for d in range(1, len(ranges)):
        term = np.asarray(ranges[d], dtype=np.int64) * strides[d]
        flat = (flat[:, None] + term[None, :]).ravel(order="F")
    return flat


# --------------------------------------------------------------------------
# per-patch, per-level 1D data
# --------------------------------------------------------------------------


def _tensor_offsets(shapes):
    """``(dim, prod(shapes))`` local tensor indices, first direction fastest."""
    grids = np.meshgrid(*[np.arange(n) for n in shapes], indexing="ij")
    return np.stack([g.ravel(order="F") for g in grids])


def _bisect(knot):
    """One dyadic refinement of a knot vector: bisect every non-empty span."""
    uniq = np.unique(knot)
    return np.sort(np.concatenate([knot, (uniq[:-1] + uniq[1:]) / 2]))


def _oslo_row(tau, deg, t, i):
    """One row of the two-scale operator, by the Oslo algorithm.

    Returns ``(offset, weights)`` such that the level-fine coefficient ``d_i``
    of any spline is ``weights @ c[offset : offset + deg + 1]`` in terms of its
    coarse coefficients ``c`` -- equivalently, row ``i`` of the matrix ``P``
    with ``N^coarse_j = sum_i P[i, j] N^fine_i``.

    Cost is ``O(p^2)`` for one row, independent of how long the knot vectors
    are. That is what makes deep hierarchies affordable: a level-``l`` knot
    vector is ``2^l`` times as long as level 0, and both the dense-identity
    route (``O(n^2)`` memory) and composing one sparse insertion per new knot
    (``O(n^2)`` work) become impossible well before the depth an adaptive run
    reaches. Only the rows belonging to live functions are ever asked for.

    Reference: Lyche and Morken, *Knot insertion*, the Oslo algorithm.
    """
    tau = np.asarray(tau, dtype=float)
    t = np.asarray(t, dtype=float)
    mu = int(np.searchsorted(tau, t[i], side="right") - 1)
    mu = min(max(mu, deg), len(tau) - deg - 2)

    B = np.eye(deg + 1)
    for q in range(deg, 0, -1):
        x = t[i + q]
        out = np.zeros((q, deg + 1))
        for r in range(q):
            hi, lo = tau[mu + 1 + r], tau[mu + 1 + r - q]
            denom = hi - lo
            if denom <= 0.0:
                out[r] = B[r + 1]
            else:
                out[r] = ((hi - x) * B[r] + (x - lo) * B[r + 1]) / denom
        B = out
    return mu - deg, B[0]


def _two_scale_rows(tau, deg, t, indices):
    """``(offsets, weights)`` for several rows at once.

    The rows repeat heavily. ``indices`` holds the index *in one direction* of
    every live function, and in a tensor-product patch each distinct value
    recurs once for every function sharing it in the other directions -- in 3D
    that is O(n^(2/3)) times each. Computing the distinct rows and gathering is
    exact and avoids recomputing the same rows while building the space.
    """
    indices = np.asarray(indices, dtype=np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    offsets = np.empty(len(unique), dtype=np.int64)
    weights = np.empty((len(unique), deg + 1))
    for k, i in enumerate(unique):
        offsets[k], weights[k] = _oslo_row(tau, deg, t, int(i))
    return offsets[inverse], weights[inverse]


class _Levels:
    """Per-direction data for every level of one patch.

    Strictly one-dimensional: nothing here grows like ``2^(l*dim)``.
    """

    def __init__(self, patch: Patch, n_levels: int):
        self.patch = patch
        self.dim = patch.dim
        self.degree = patch.degree
        self.knots = [patch.knot_arrays()]
        self.spans = []
        self.ext1d = []
        self.n_elem = []
        self.n_func = []
        self.extend(n_levels)

    @property
    def n_levels(self) -> int:
        return len(self.n_elem)

    def extend(self, n_levels: int):
        """Add levels until there are ``n_levels`` of them."""
        while len(self.knots) < n_levels:
            self.knots.append([_bisect(kv) for kv in self.knots[-1]])

        while len(self.n_elem) < n_levels:
            l = len(self.n_elem)
            kv = self.knots[l]
            spans = [element_spans(kv[d], self.degree[d]) for d in range(self.dim)]
            self.spans.append(spans)
            self.ext1d.append(
                [bezier_extraction(kv[d], self.degree[d])[0] for d in range(self.dim)]
            )
            self.n_elem.append(tuple(len(s[0]) for s in spans))
            self.n_func.append(tuple(len(kv[d]) - self.degree[d] - 1 for d in range(self.dim)))

    # -- 1D incidence -----------------------------------------------------

    def funcs_of_elem(self, l, d, e):
        """Level-``l`` function indices in direction ``d`` supported on element ``e``."""
        first = self.spans[l][d][2]
        return np.arange(first[e], first[e] + self.degree[d] + 1)

    def elems_of_func(self, l, d, j):
        """Level-``l`` element indices in direction ``d`` in the support of ``j``."""
        first = self.spans[l][d][2]
        lo = int(np.searchsorted(first, j - self.degree[d], side="left"))
        hi = int(np.searchsorted(first, j, side="right"))
        return np.arange(lo, hi)

    def element_box(self, l, e_flat):
        idx = np.array(_unravel([e_flat], self.n_elem[l]))[:, 0]
        box = np.zeros(2 * self.dim)
        for d in range(self.dim):
            lo, hi, _ = self.spans[l][d]
            box[d], box[self.dim + d] = lo[idx[d]], hi[idx[d]]
        return box, idx


# --------------------------------------------------------------------------
# refinement state
# --------------------------------------------------------------------------


def _children(flat, dims_coarse, dims_fine):
    """The ``2^dim`` level-(l+1) elements covering each level-l element."""
    idx = np.array(_unravel(flat, dims_coarse))
    dim = len(dims_coarse)
    out = []
    for corner in range(2**dim):
        offset = np.array([(corner >> d) & 1 for d in range(dim)])
        out.append(_ravel(tuple(2 * idx + offset[:, None]), dims_fine))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


class _State:
    """Mutable refinement state during marking and closure."""

    def __init__(self, patches, refined):
        self.patches = list(patches)
        self.refined = [[set(int(e) for e in lv) for lv in r] for r in refined]
        self.levels = [_Levels(p, self.n_levels) for p in self.patches]
        self._domains = None

    @property
    def n_levels(self) -> int:
        """One more than the deepest level that has been refined."""
        deepest = -1
        for r in self.refined:
            for l, marked in enumerate(r):
                if marked:
                    deepest = max(deepest, l)
        return deepest + 2

    def _sync(self):
        n = self.n_levels
        for lv in self.levels:
            lv.extend(n)
        for r in self.refined:
            while len(r) < n:
                r.append(set())
        self._domains = None

    def domains(self):
        """``dom[i][l]`` and ``leaf[i][l]``: live and leaf elements, as sets."""
        if self._domains is not None:
            return self._domains
        self._sync()
        n = self.n_levels
        dom, leaf = [], []
        for i, levels in enumerate(self.levels):
            d_i = [set(range(int(np.prod(levels.n_elem[0]))))]
            for l in range(n - 1):
                marked = self.refined[i][l] if l < len(self.refined[i]) else set()
                bad = marked - d_i[l]
                if bad:
                    raise HierarchyError(
                        f"patch {i}: {len(bad)} elements marked at level {l} lie "
                        f"outside the level-{l} domain"
                    )
                kids = _children(
                    np.array(sorted(marked), dtype=np.int64),
                    levels.n_elem[l],
                    levels.n_elem[l + 1],
                )
                d_i.append(set(int(k) for k in kids))
            dom.append(d_i)
            leaf.append(
                [d_i[l] - (self.refined[i][l] if l < len(self.refined[i]) else set())
                 for l in range(n)]
            )
        self._domains = (dom, leaf)
        return self._domains

    def ensure_live(self, i, level, elem) -> bool:
        """Make sure ``elem`` exists at ``level``, refining ancestors as needed."""
        dom, _ = self.domains()
        if level == 0 or int(elem) in dom[i][level]:
            return False
        idx = np.array(_unravel([int(elem)], self.levels[i].n_elem[level]))[:, 0]
        parent = int(_ravel(tuple(idx // 2), self.levels[i].n_elem[level - 1]))
        return self.mark(i, level - 1, parent)

    def mark(self, i, level, elem) -> bool:
        """Refine one element, refining its ancestors first if needed."""
        self._sync()
        if level >= self.n_levels:
            raise HierarchyError(f"level {level} does not exist yet")
        elem = int(elem)
        if level > 0:
            idx = np.array(_unravel([elem], self.levels[i].n_elem[level]))[:, 0]
            parent = int(_ravel(tuple(idx // 2), self.levels[i].n_elem[level - 1]))
            if parent not in self.refined[i][level - 1]:
                self.mark(i, level - 1, parent)
        while len(self.refined[i]) <= level:
            self.refined[i].append(set())
        if elem in self.refined[i][level]:
            return False
        self.refined[i][level].add(elem)
        self._domains = None
        self._sync()
        return True

    def to_hierarchy(self, truncate=True, admissible=True) -> Hierarchy:
        trimmed = []
        for r in self.refined:
            lv = [tuple(sorted(s)) for s in r]
            while lv and not lv[-1]:
                lv.pop()
            trimmed.append(tuple(lv))
        return Hierarchy(refined=tuple(trimmed), truncate=truncate, admissible=admissible)


# --------------------------------------------------------------------------
# closure
# --------------------------------------------------------------------------


def _side_elements(levels: _Levels, l, elems, direction, end):
    """Which of ``elems`` lie on one parametric side of the patch."""
    if not elems:
        return np.zeros(0, dtype=np.int64)
    flat = np.array(sorted(elems), dtype=np.int64)
    idx = np.array(_unravel(flat, levels.n_elem[l]))
    limit = 0 if end == 0 else levels.n_elem[l][direction] - 1
    return flat[idx[direction] == limit]


def _face_centres(levels: _Levels, l, flat, direction, end):
    """Physical centres of the faces those elements present on a patch side."""
    idx = np.array(_unravel(flat, levels.n_elem[l]))
    params = np.zeros((len(flat), levels.dim))
    for d in range(levels.dim):
        lo, hi, _ = levels.spans[l][d]
        params[:, d] = 0.5 * (lo[idx[d]] + hi[idx[d]])
    params[:, direction] = 0.0 if end == 0 else 1.0
    return evaluate_patch(levels.patch, params)


def _interface_partners(state: _State, interfaces, level):
    """``(patch, elem)`` on an interface side -> the neighbour's element.

    Matching is by geometric coincidence of element-face centres, the same
    principle the multipatch topology uses for control points, so it needs no
    orientation bookkeeping and works in any dimension.
    """
    if not interfaces:
        return {}
    dom, _ = state.domains()

    sides = set()
    for iface in interfaces:
        sides.add((iface.patch_a, iface.side_a))
        sides.add((iface.patch_b, iface.side_b))

    keys, pts = [], []
    for i, side in sorted(sides):
        direction, end = divmod(side_code(side), 2)
        flat = _side_elements(state.levels[i], level, dom[i][level], direction, end)
        if not len(flat):
            continue
        keys.extend((i, int(e)) for e in flat)
        pts.append(_face_centres(state.levels[i], level, flat, direction, end))

    if not pts:
        return {}
    pts = np.concatenate(pts, axis=0)
    partners = {}
    for a, b in cKDTree(pts).query_pairs(r=1e-9):
        if keys[a][0] == keys[b][0]:
            continue
        partners.setdefault(keys[a], []).append(keys[b])
        partners.setdefault(keys[b], []).append(keys[a])
    return partners


def _support_extension(levels: _Levels, l, elem):
    """Level-``(l-1)`` elements in the support extension of a level-``l`` element.

    That is: the elements met by the support of any level-(l-1) function whose
    support meets ``elem``. Per direction this is the index range around
    ``elem``'s parent spanned by ``2p+1`` elements, so it costs ``(2p+1)^dim``
    and never touches a tensor-product array.
    """
    idx = np.array(_unravel([int(elem)], levels.n_elem[l]))[:, 0] // 2
    ranges = []
    for d in range(levels.dim):
        first = levels.spans[l - 1][d][2]
        anchor = first[idx[d]]
        deg = levels.degree[d]
        lo = int(np.searchsorted(first, anchor - deg, side="left"))
        hi = int(np.searchsorted(first, anchor + deg, side="right"))
        ranges.append(np.arange(lo, hi))
    return _tensor_indices(ranges, levels.n_elem[l - 1])


def _support_extension_union(levels: _Levels, l, elems):
    """Level-``(l-1)`` elements in the support extension of *any* of ``elems``.

    The batched form of :func:`_support_extension`, and the reason it exists:
    the admissibility sweep asks this of every live element at a level, and only
    ever uses the union, so doing it one element at a time costs
    ``n * (2p+1)^dim`` Python-level operations -- three million of them on a
    14000-element 3D mesh. Iterating over the ``(2p+1)^dim`` *offsets* instead,
    with every element handled at once inside each, is the same set for a
    hundred-odd vectorised passes.
    """
    elems = np.asarray(elems, dtype=np.int64)
    if elems.size == 0:
        return np.zeros(0, dtype=np.int64)

    dim = levels.dim
    parent = np.array(_unravel(elems, levels.n_elem[l])) // 2  # (dim, n)

    lows, widths = [], []
    for d in range(dim):
        first = levels.spans[l - 1][d][2]
        anchor = first[parent[d]]
        deg = levels.degree[d]
        lows.append(np.searchsorted(first, anchor - deg, side="left"))
        widths.append(np.searchsorted(first, anchor + deg, side="right") - lows[-1])

    dims = levels.n_elem[l - 1]
    strides = np.concatenate([[1], np.cumprod(np.asarray(dims[:-1], dtype=np.int64))])
    spans = [int(w.max()) for w in widths]

    pieces = []
    for offset in np.ndindex(*spans):
        inside = np.ones(len(elems), dtype=bool)
        flat = np.zeros(len(elems), dtype=np.int64)
        for d in range(dim):
            inside &= offset[d] < widths[d]
            flat += (lows[d] + offset[d]) * strides[d]
        if inside.any():
            pieces.append(flat[inside])
    return np.unique(np.concatenate(pieces)) if pieces else np.zeros(0, dtype=np.int64)


def _close(state: _State, interfaces, admissible: bool):
    """Bring the marking to a representable mesh.

    Two closures run to a joint fixpoint.

    *Interfaces.* Refinement is mirrored across patch interfaces so the
    interface knot lines keep matching, without which the conforming zip would
    fail.

    *Admissibility.* Every element ``Q`` of level ``l`` must have its whole
    level-``(l-1)`` support extension inside the level-``(l-1)`` domain. This is
    the class-2 admissibility of Buffa and Giannelli: at most two successive
    levels of truncated functions act on any element, which is what bounds
    ``n_local_max``. Note that this is a genuinely stronger condition than
    "levels of touching elements differ by at most one" -- element adjacency
    alone leaves a coarse function reaching several levels down through its
    ``p+1``-element support, and in practice lets ``n_local_max`` grow to four
    or five times the tensor-product count.
    """
    for _ in range(256):
        changed = False

        if interfaces:
            for level in range(state.n_levels):
                partners = _interface_partners(state, interfaces, level)
                for (i, e), others in partners.items():
                    if level >= len(state.refined[i]) or e not in state.refined[i][level]:
                        continue
                    for j, e2 in others:
                        if (
                            level >= len(state.refined[j])
                            or e2 not in state.refined[j][level]
                        ):
                            changed |= state.mark(j, level, e2)

        if admissible:
            # One full sweep per outer iteration rather than restarting on the
            # first change: marks made here are picked up by the next sweep, and
            # restarting eagerly makes the closure quadratic in the marked count.
            for i, levels in enumerate(state.levels):
                for l in range(state.n_levels - 1, 0, -1):
                    dom, _ = state.domains()
                    if l >= len(dom[i]):
                        continue
                    required = _support_extension_union(
                        levels, l, np.fromiter(dom[i][l], dtype=np.int64, count=len(dom[i][l]))
                    )
                    # Only the ones not already live need anything doing, and
                    # there are few of them -- that is what makes the remaining
                    # per-element loop short.
                    coarse = dom[i][l - 1]
                    for c in required[[int(x) not in coarse for x in required]]:
                        changed |= state.ensure_live(i, l - 1, int(c))

        if not changed:
            return
    raise HierarchyError("refinement closure did not converge in 256 sweeps")


# --------------------------------------------------------------------------
# THB construction
# --------------------------------------------------------------------------


def _candidate_functions(levels: _Levels, l, elems):
    """Functions supported on at least one of ``elems`` (a set of flat indices).

    This is the *relevant* set at level l: a function outside it has support
    disjoint from the live region, so it contributes nothing on any leaf at
    this level or below, and nothing to the next level's two-scale product.
    """
    if not elems:
        return np.zeros(0, dtype=np.int64)
    flat = np.array(sorted(elems), dtype=np.int64)
    idx = np.array(_unravel(flat, levels.n_elem[l]))
    out = set()
    for k in range(flat.shape[0]):
        ranges = [levels.funcs_of_elem(l, d, idx[d, k]) for d in range(levels.dim)]
        out.update(int(v) for v in _tensor_indices(ranges, levels.n_func[l]))
    return np.array(sorted(out), dtype=np.int64)


def _active_functions(levels: _Levels, l, candidates, dom, leaf):
    """Active level functions: support inside the domain, still covering a leaf."""
    active = []
    for j in candidates:
        idx = np.array(_unravel([int(j)], levels.n_func[l]))[:, 0]
        ranges = [levels.elems_of_func(l, d, idx[d]) for d in range(levels.dim)]
        supp = _tensor_indices(ranges, levels.n_elem[l])
        supp = [int(e) for e in supp]
        if all(e in dom for e in supp) and any(e in leaf for e in supp):
            active.append(int(j))
    return np.array(active, dtype=np.int64)


def _restricted_two_scale(levels: _Levels, l, rel_coarse, rel_fine):
    """``S`` on the relevant index sets: rows ``rel_fine``, columns ``rel_coarse``.

    Assembled directly from per-direction Oslo rows. The full tensor-product
    two-scale matrix is never formed -- at ten levels deep it would have
    millions of rows for a handful of live functions.
    """
    dim = levels.dim
    n_coarse_dims = levels.n_func[l]
    fine_idx = np.array(_unravel(rel_fine, levels.n_func[l + 1]))

    per_dir = [
        _two_scale_rows(
            levels.knots[l][d], levels.degree[d], levels.knots[l + 1][d], fine_idx[d]
        )
        for d in range(dim)
    ]

    # Every fine row contributes the same (p+1)^dim stencil, shifted by its own
    # offsets, so the whole operator is one broadcast rather than a Python loop
    # over rows and then over the entries of each.
    n_fine = len(rel_fine)
    shapes = [levels.degree[d] + 1 for d in range(dim)]
    local = _tensor_offsets(shapes)  # (dim, n_local), first direction fastest
    strides = np.concatenate(
        [[1], np.cumprod(np.asarray(n_coarse_dims[:-1], dtype=np.int64))]
    )

    flat = np.zeros((n_fine, local.shape[1]), dtype=np.int64)
    coef = np.ones((n_fine, local.shape[1]))
    for d in range(dim):
        offsets_d, weights_d = per_dir[d]
        flat += (offsets_d[:, None] + local[d][None, :]) * strides[d]
        coef *= weights_d[:, local[d]]

    # Keep only the columns that are live, found by binary search rather than a
    # dense lookup: the coarse index space is 2^(l*dim) entries and must never
    # be materialised.
    order = np.argsort(rel_coarse)
    sorted_coarse = np.asarray(rel_coarse)[order]
    slot = np.searchsorted(sorted_coarse, flat).clip(0, max(len(order) - 1, 0))
    live = (len(order) > 0) & (sorted_coarse[slot] == flat) & (np.abs(coef) > COEFF_TOL)

    rows = np.repeat(np.arange(n_fine), local.shape[1]).reshape(flat.shape)[live]
    cols = order[slot][live]
    return sp.csr_matrix(
        (coef[live], (rows, cols)), shape=(n_fine, len(rel_coarse))
    )


def _thb_matrices(levels: _Levels, relevant, actives, two_scale, truncate):
    """Per-level coefficient matrices of the (truncated) hierarchical basis."""
    columns = []
    for l, a in enumerate(actives):
        columns.extend((l, int(j)) for j in a)
    n_total = len(columns)
    col_of = {key: c for c, key in enumerate(columns)}

    Q = []
    current = sp.csr_matrix((len(relevant[0]), n_total))
    for l in range(len(relevant)):
        if l > 0:
            current = (two_scale[l - 1] @ current).tocsr()
            if truncate and len(actives[l]):
                # Truncation: drop what the coarse functions contribute along
                # the active fine functions. This is what restores the partition
                # of unity and shrinks coarse supports.
                row_of = {int(j): r for r, j in enumerate(relevant[l])}
                keep = np.ones(len(relevant[l]))
                for j in actives[l]:
                    keep[row_of[int(j)]] = 0.0
                current = (sp.diags(keep) @ current).tocsr()
            current.data[np.abs(current.data) < COEFF_TOL] = 0.0
            current.eliminate_zeros()

        if len(actives[l]):
            row_of = {int(j): r for r, j in enumerate(relevant[l])}
            rows = np.array([row_of[int(j)] for j in actives[l]], dtype=int)
            cols = np.array([col_of[(l, int(j))] for j in actives[l]], dtype=int)
            current = (
                current
                + sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=current.shape)
            ).tocsr()

        # CSR, because the only consumer gathers whole rows of it: one per
        # element, for the functions that element supports.
        block = current.tocsr()
        block.sum_duplicates()
        Q.append(block)
    return Q, columns


def _extraction_ids(ops):
    """Label each 1D extraction operator by which distinct matrix it is."""
    ids, seen = np.empty(len(ops), dtype=np.int64), {}
    for k, op in enumerate(ops):
        key = np.ascontiguousarray(op).tobytes()
        ids[k] = seen.setdefault(key, len(seen))
    return ids


def _gather_rows(indptr, indices, data, rows):
    """Dense block of the named CSR rows, over the columns they actually touch.

    Returns ``(cols, V)`` with ``V[i, k]`` the entry at row ``rows[i]`` and
    column ``cols[k]``. Done on the CSR arrays rather than by slicing the
    sparse matrix twice: this runs once per element, and two scipy sparse
    constructions per element cost more than the arithmetic they carry.
    """
    starts, ends = indptr[rows], indptr[rows + 1]
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((len(rows), 0))

    begins = np.cumsum(counts) - counts
    entries = np.arange(total) + np.repeat(starts - begins, counts)
    owner = np.repeat(np.arange(len(rows)), counts)

    values = data[entries]
    live = values != 0.0
    cols = np.unique(indices[entries][live])
    V = np.zeros((len(rows), len(cols)))
    V[owner[live], np.searchsorted(cols, indices[entries][live])] = values[live]
    return cols, V


def _patch_connectivity(levels: _Levels, dom, leaf, truncate):
    """Leaf elements, their supported functions and extraction rows, per patch."""
    n_levels = len(dom)
    dim = levels.dim

    relevant = [_candidate_functions(levels, l, dom[l]) for l in range(n_levels)]
    actives = [
        _active_functions(levels, l, relevant[l], dom[l], leaf[l]) for l in range(n_levels)
    ]
    two_scale = [
        _restricted_two_scale(levels, l, relevant[l], relevant[l + 1])
        for l in range(n_levels - 1)
    ]
    Q, columns = _thb_matrices(levels, relevant, actives, two_scale, truncate)

    # The control net of each level, restricted to the live functions, obtained
    # by pushing the level-0 homogeneous net through the same operators. THB's
    # coefficient preservation then makes the hierarchical geometry map
    # reproduce the level-0 patch exactly.
    hom = [np.asarray(levels.patch.homogeneous())[relevant[0]]]
    for l in range(n_levels - 1):
        hom.append(np.asarray(two_scale[l] @ hom[l]))
    row_index = [{int(j): r for r, j in enumerate(relevant[l])} for l in range(n_levels)]
    control = np.array([hom[l][row_index[l][int(j)]] for l, j in columns])

    keys, boxes, rows_out, ops_out = [], [], [], []
    for l in range(n_levels):
        if not leaf[l]:
            continue
        row_of = row_index[l]
        indptr, indices, data = Q[l].indptr, Q[l].indices, Q[l].data
        # The Bezier extraction of an element depends only on which of the few
        # distinct 1D operators each direction contributes -- an open uniform
        # knot vector has 2p+1 of them however long it is -- so the Kronecker
        # products repeat, and there are two of them per element in 3D.
        bezier_ids = [_extraction_ids(levels.ext1d[l][d]) for d in range(dim)]
        bezier_cache = {}

        for e in sorted(leaf[l]):
            box, idx = levels.element_box(l, e)
            ranges = [levels.funcs_of_elem(l, d, idx[d]) for d in range(dim)]
            local = _tensor_indices(ranges, levels.n_func[l])
            local_rows = np.fromiter(
                (row_of[int(j)] for j in local), dtype=np.int64, count=len(local)
            )
            cols, V = _gather_rows(indptr, indices, data, local_rows)

            key = tuple(int(bezier_ids[d][idx[d]]) for d in range(dim))
            C_bez = bezier_cache.get(key)
            if C_bez is None:
                C_bez = _tensor_extraction(
                    [levels.ext1d[l][d][idx[d]] for d in range(dim)]
                )
                bezier_cache[key] = C_bez

            ops_out.append(V.T @ C_bez)
            rows_out.append(cols)
            boxes.append(box)
            keys.append((l, int(e)))

    return keys, boxes, rows_out, ops_out, columns, control


def _side_function_mask(levels: _Levels, columns, direction, end):
    """Which active functions have their tensor index on one patch side."""
    mask = np.zeros(len(columns), dtype=bool)
    for c, (l, j) in enumerate(columns):
        idx = np.array(_unravel([j], levels.n_func[l]))[:, 0]
        limit = 0 if end == 0 else levels.n_func[l][direction] - 1
        mask[c] = idx[direction] == limit
    return mask


def _glue(all_columns, all_cpts, all_wgts):
    """Global scalar-dof numbering across patches, by control-point coincidence.

    Only functions of the same level may merge: two patches whose interface
    carries different levels do not share a basis, and the closure is what
    prevents that from arising.
    """
    counts = [len(c) for c in all_columns]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(int)
    cpts = np.concatenate(all_cpts, axis=0)
    wgts = np.concatenate(all_wgts)
    patch_of = np.concatenate([np.full(n, i) for i, n in enumerate(counts)])
    level_of = np.concatenate([np.array([l for l, _ in c], dtype=int) for c in all_columns])

    parent = np.arange(len(cpts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if len(counts) > 1 and len(cpts):
        for a, b in cKDTree(cpts).query_pairs(r=TOL):
            if patch_of[a] == patch_of[b] or level_of[a] != level_of[b]:
                continue
            if abs(wgts[a] - wgts[b]) > TOL:
                raise NonConformingError(
                    f"patches {patch_of[a]} and {patch_of[b]} share a control point "
                    f"at {cpts[a]} but assign it different NURBS weights"
                )
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([find(i) for i in range(len(cpts))])
    _, global_index = np.unique(roots, return_inverse=True)
    n_global = int(global_index.max()) + 1 if len(global_index) else 0
    local_to_global = tuple(
        global_index[offsets[i] : offsets[i + 1]].astype(int) for i in range(len(counts))
    )
    return local_to_global, n_global


def _check_interface_conformity(all_levels, all_columns, local_to_global, interfaces):
    """Both sides of every interface must carry the same global dofs."""
    for iface in interfaces:
        a_dir, a_end = divmod(side_code(iface.side_a), 2)
        b_dir, b_end = divmod(side_code(iface.side_b), 2)
        ga = set(
            local_to_global[iface.patch_a][
                _side_function_mask(
                    all_levels[iface.patch_a], all_columns[iface.patch_a], a_dir, a_end
                )
            ].tolist()
        )
        gb = set(
            local_to_global[iface.patch_b][
                _side_function_mask(
                    all_levels[iface.patch_b], all_columns[iface.patch_b], b_dir, b_end
                )
            ].tolist()
        )
        if ga != gb:
            raise NonConformingError(
                f"{iface}: the hierarchical bases do not match across the interface "
                f"({len(ga)} vs {len(gb)} functions, {len(ga & gb)} shared). The "
                f"closure should have mirrored the marking; please report this "
                f"together with the geometry."
            )


def build_hierarchical_space(patches, hierarchy: Hierarchy, vec: int = 1) -> FunctionSpace:
    """Assemble the FunctionSpace described by a :class:`Hierarchy`."""
    patches = list(patches)
    state = _State(patches, hierarchy.refined)
    interfaces = compute_topology(patches).interfaces
    dom, leaf = state.domains()

    per_patch = []
    all_columns, all_cpts, all_wgts = [], [], []
    for i in range(len(patches)):
        levels = state.levels[i]
        keys, boxes, rows, ops, columns, hom = _patch_connectivity(
            levels, dom[i], leaf[i], hierarchy.truncate
        )
        per_patch.append((keys, boxes, rows, ops))
        all_columns.append(columns)
        all_wgts.append(hom[:, -1])
        all_cpts.append(hom[:, :-1] / hom[:, -1:])

    local_to_global, n_global = _glue(all_columns, all_cpts, all_wgts)
    all_levels = state.levels
    _check_interface_conformity(all_levels, all_columns, local_to_global, interfaces)

    degree = patches[0].degree
    dim = patches[0].dim
    n_bernstein = int(np.prod([p + 1 for p in degree]))
    n_local_max = max(
        (len(r) for _, _, rows, _ in per_patch for r in rows), default=n_bernstein
    )
    n_local_max = max(n_local_max, n_bernstein)
    if n_local_max > MAX_LOCAL_GROWTH * n_bernstein:
        raise HierarchyError(
            f"an element gathers {n_local_max} functions where a tensor-product "
            f"element of this degree gathers {n_bernstein}; assembly cost grows "
            f"with the square of that. Refine with admissible=True, or mark fewer "
            f"elements per cycle."
        )

    n_elems = sum(len(keys) for keys, _, _, _ in per_patch)
    elem_dofs = np.full((n_elems, n_local_max), n_global, dtype=int)
    extraction = np.zeros((n_elems, n_local_max, n_bernstein))
    elem_vertex = np.zeros((n_elems, 2 * dim))
    elem_patch = np.zeros(n_elems, dtype=int)
    elem_key = np.zeros((n_elems, 3), dtype=int)

    k = 0
    for i, (keys, boxes, rows, ops) in enumerate(per_patch):
        for (l, e), box, r, op in zip(keys, boxes, rows, ops):
            n = len(r)
            elem_dofs[k, :n] = local_to_global[i][r]
            extraction[k, :n] = op
            elem_vertex[k] = box
            elem_patch[k] = i
            elem_key[k] = (i, l, e)
            k += 1

    cpts = np.zeros((n_global, patches[0].dim_phys))
    wgts = np.zeros(n_global)
    for i in range(len(patches)):
        cpts[local_to_global[i]] = all_cpts[i]
        wgts[local_to_global[i]] = all_wgts[i]

    boundaries = _classify_boundaries(
        state, all_levels, all_columns, local_to_global, interfaces, elem_key
    )

    return _new_space(
        extraction=jnp.asarray(extraction),
        cpts=jnp.asarray(cpts),
        wgts=jnp.asarray(wgts),
        elem_dofs=StaticArray(elem_dofs),
        elem_patch=StaticArray(elem_patch),
        elem_key=StaticArray(elem_key),
        elem_vertex=elem_vertex,
        boundaries=boundaries,
        topology=Topology(
            local_to_global=tuple(local_to_global),
            n_global_dofs=n_global,
            interfaces=tuple(interfaces),
        ),
        dim=dim,
        vec=int(vec),
        degree=degree,
        n_scalar_basis=n_global,
        n_elems=n_elems,
        patch_knots=tuple(p.knots for p in patches),
        patches=StaticPatches(patches),
        hierarchy=hierarchy,
    )


def _classify_boundaries(
    state, all_levels, all_columns, local_to_global, interfaces, elem_key
):
    """Group leaf elements, sides and dofs by boundary label.

    A label collects the leaf elements (of any level) whose side lies on the
    labeled patch side, plus the active functions (of any level) whose tensor
    index sits on that side. For an open knot vector those are exactly the
    functions with a non-zero trace, and truncation preserves that: a coarse
    function is truncated only against active fine functions, which on the
    boundary face are themselves boundary functions.
    """
    interface_sides = set()
    for iface in interfaces:
        interface_sides.add((iface.patch_a, iface.side_a))
        interface_sides.add((iface.patch_b, iface.side_b))

    collected = {}
    for i, patch in enumerate(state.patches):
        dim = patch.dim
        labels = patch.label_map
        levels = all_levels[i]
        mine = np.flatnonzero(elem_key[:, 0] == i)
        for d in range(dim):
            for end in (0, 1):
                side = side_name(d, end)
                if (i, side) in interface_sides:
                    continue
                label = labels.get(side, f"patch{i}/{side}")

                elems = []
                for k in mine:
                    l, e = int(elem_key[k, 1]), int(elem_key[k, 2])
                    idx = np.array(_unravel([e], levels.n_elem[l]))[:, 0]
                    limit = 0 if end == 0 else levels.n_elem[l][d] - 1
                    if idx[d] == limit:
                        elems.append(int(k))
                elems = np.asarray(elems, dtype=int)

                mask = _side_function_mask(levels, all_columns[i], d, end)
                entry = collected.setdefault(label, {"elems": [], "sides": [], "dofs": []})
                entry["elems"].append(elems)
                entry["sides"].append(np.full(len(elems), side_code(side)))
                entry["dofs"].append(local_to_global[i][mask])

    return {
        label: BoundarySet(
            elems=np.concatenate(v["elems"]).astype(int),
            sides=np.concatenate(v["sides"]).astype(int),
            dofs=np.unique(np.concatenate(v["dofs"])).astype(int),
        )
        for label, v in collected.items()
    }


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------


def refine_elements(
    space: FunctionSpace,
    marked,
    *,
    truncate: bool = True,
    admissible: bool = True,
) -> FunctionSpace:
    """Refine the marked elements of ``space``, returning a new space.

    Parameters
    ----------
    space : FunctionSpace
        Tensor-product or already hierarchical. The returned space carries a
        :class:`Hierarchy`, so repeated calls deepen the tree.
    marked : array of int
        Element indices **of this space**.
    truncate : bool
        THB (default) rather than plain HB. Truncation restores the partition of
        unity and shrinks coarse supports, which is what keeps the number of
        functions per element small. ``False`` is kept for testing.
    admissible : bool
        Run the class-2 admissibility closure (default).

    Notes
    -----
    Marking and refinement change array shapes, so an adaptive loop is an outer
    Python loop -- ``jax.grad`` does not flow across cycles. Each adapted space
    is otherwise a full citizen of the differentiability contract.
    """
    if space.collapse_degenerate:
        raise NotImplementedError("hierarchical refinement of collapsed boundaries is unsupported; refine the tensor patches first")
    if space.cell_type in {"simplex", "mixed"}:
        raise NotImplementedError("hierarchical spline refinement is unavailable for simplices; remesh CAD")
    if space.patches is None:
        raise HierarchyError(
            "this space does not carry its patches, so it cannot be refined; "
            "build it with FunctionSpace(patches, ...)"
        )
    patches = list(space.patches)

    if space.hierarchy is None:
        base = Hierarchy(
            refined=tuple(() for _ in patches), truncate=truncate, admissible=admissible
        )
        keys = _initial_elem_keys(space, patches)
    else:
        base = space.hierarchy
        if space.elem_key is None:
            raise HierarchyError("hierarchical space is missing its element keys")
        keys = np.asarray(space.elem_key)

    marked = np.unique(np.asarray(marked, dtype=int).reshape(-1))
    if marked.size and (marked.min() < 0 or marked.max() >= space.n_elems):
        raise IndexError(
            f"marked element index out of range for a space with {space.n_elems} elements"
        )

    state = _State(patches, base.refined)
    for e in marked:
        i, l, local = (int(v) for v in keys[e])
        state.mark(i, l, local)

    _close(state, compute_topology(patches).interfaces, admissible)

    return build_hierarchical_space(
        patches,
        state.to_hierarchy(truncate=truncate, admissible=admissible),
        vec=space.vec,
    )


def _initial_elem_keys(space, patches):
    """``(patch, level, index)`` for a tensor-product space: everything at level 0."""
    keys = np.zeros((space.n_elems, 3), dtype=int)
    keys[:, 0] = np.asarray(space.elem_patch)
    offset = 0
    for patch in patches:
        n = int(
            np.prod(
                [
                    len(element_spans(kv, patch.degree[d])[0])
                    for d, kv in enumerate(patch.knot_arrays())
                ]
            )
        )
        keys[offset : offset + n, 2] = np.arange(n)
        offset += n
    if offset != space.n_elems:
        raise HierarchyError(
            f"element bookkeeping mismatch: patches account for {offset} elements "
            f"but the space has {space.n_elems}"
        )
    return keys


def dorfler_mark(eta, frac: float = 0.3) -> np.ndarray:
    """Dorfler (bulk) marking: the fewest elements carrying ``frac`` of ``eta^2``.

    Parameters
    ----------
    eta : (n_elems,) array
        Per-element error indicators.
    frac : float
        Fraction of the total squared indicator the marked set must reach.
    """
    eta = np.asarray(eta, dtype=float).reshape(-1)
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must lie in (0, 1], got {frac}")
    order = np.argsort(eta**2)[::-1]
    cumulative = np.cumsum(eta[order] ** 2)
    total = cumulative[-1] if cumulative.size else 0.0
    if total <= 0.0:
        return np.zeros(0, dtype=int)
    n = int(np.searchsorted(cumulative, frac * total) + 1)
    return np.sort(order[:n])
