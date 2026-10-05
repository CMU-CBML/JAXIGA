"""Point sets: quadrature, collocation and sampling.

One abstraction serves all three. A :class:`PointSet` names a group of elements
and, for each, points in the reference cube ``[-1, 1]^dim`` or the unit
simplex ``x >= 0, sum(x) <= 1``.

Reference points are static data (the differentiability contract explicitly
excludes point locations), so they are built in NumPy at setup and the
Bernstein bases over them are precomputed once.

``ref`` is shared across elements when the rule is the same everywhere (the
interior Gauss case, by far the largest), and per-element only where it must
be -- boundary sets whose elements sit on different sides. That distinction is
what keeps the interior quadrature from materializing one copy of the basis
per element.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class PointSet:
    """Evaluation points attached to elements of a function space.

    Attributes
    ----------
    elems : (n_e,) int array
        Elements this set covers.
    ref : (n_q, dim) or (n_e, n_q, dim) array
        Points in the reference cube or unit simplex. Two dimensions means the
        same rule on every element.
    weights : (n_q,) or (n_e, n_q) array or None
        Reference-cube quadrature weights, excluding the element Jacobian,
        which :func:`~jaxiga.space.evaluation.evaluate` applies. ``None`` for
        collocation and sampling sets.
    sides : (n_e,) int array or None
        Tensor side code ``2*direction + end`` or simplex opposite vertex
        index per element, for boundary sets.
    tag : str
        ``"interior"``, ``"boundary:<label>"``, ``"interface"`` or ``"grid"``.
    """

    elems: np.ndarray
    ref: jnp.ndarray
    weights: jnp.ndarray | None = None
    sides: np.ndarray | None = None
    point_dofs: np.ndarray | None = None
    tag: str = "interior"

    @property
    def n_elems(self) -> int:
        return len(self.elems)

    @property
    def n_q(self) -> int:
        return self.ref.shape[-2]

    @property
    def dim(self) -> int:
        return self.ref.shape[-1]

    @property
    def shared_ref(self) -> bool:
        """True when every element uses the same reference points."""
        return self.ref.ndim == 2

    @property
    def n_pts(self) -> int:
        return self.n_elems * self.n_q


def slice_elements(ps: PointSet, lo: int, hi: int) -> PointSet:
    """The part of a point set covering elements ``[lo, hi)`` of its own list.

    Used by the chunked assembly path, which evaluates a few elements at a time
    instead of materializing the basis over the whole mesh.
    """
    return PointSet(
        elems=ps.elems[lo:hi],
        ref=ps.ref if ps.shared_ref else ps.ref[lo:hi],
        weights=(
            ps.weights
            if ps.weights is None or ps.weights.ndim == 1
            else ps.weights[lo:hi]
        ),
        sides=None if ps.sides is None else ps.sides[lo:hi],
        point_dofs=None if ps.point_dofs is None else ps.point_dofs[lo:hi],
        tag=ps.tag,
    )


def gauss_1d(n: int):
    """Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def _tensor_rule(n_per_dir):
    """Tensor-product Gauss rule; first direction varies fastest."""
    per_dir = [gauss_1d(n) for n in n_per_dir]
    grids = np.meshgrid(*[p[0] for p in per_dir], indexing="ij")
    wgrids = np.meshgrid(*[p[1] for p in per_dir], indexing="ij")

    pts = np.stack([g.ravel(order="F") for g in grids], axis=1)
    wts = np.prod([w.ravel(order="F") for w in wgrids], axis=0)
    return pts, wts


def gauss(space, n: int | None = None) -> PointSet:
    """Gauss quadrature on every element (Duffy Gauss-Jacobi for simplices).

    ``n`` points per direction, defaulting to ``p + 1`` (exact for the degree
    ``2p`` integrands of a linear problem). Simplex rules default to ``p+2``
    points per Duffy direction to account for the transformation Jacobian.
    """
    if space.cell_type == "mixed":
        from jaxiga.space.mixed import pointset
        return pointset(space, max(space.degree)+2 if n is None else int(n))
    n_per_dir = [(p + 1) if n is None else int(n) for p in space.degree]
    if space.cell_type == "simplex":
        from jaxiga.space.simplex import gauss_rule
        pts, wts = gauss_rule(space.dim, (space.degree[0] + 2) if n is None else int(n))
    else:
        pts, wts = _tensor_rule(n_per_dir)
    return PointSet(
        elems=np.arange(space.n_elems),
        ref=np.asarray(pts),
        weights=np.asarray(wts),
        tag="interior",
    )


def _as_boundary_set(space, target):
    """Accept either a label or an explicit boundary set.

    Neumann conditions carry their own selection -- which may cover part of a
    labeled side -- so the point-set builders must take a set, not just a name.
    """
    return space.boundary(target) if isinstance(target, str) else target


def boundary_gauss(space, target, n: int | None = None) -> PointSet:
    """Gauss rule of dimension ``dim-1`` on a boundary label or explicit set.

    Reference points are per-element, since one selection may gather elements
    sitting on different parametric sides.
    """
    if space.cell_type == "mixed":
        from jaxiga.space.mixed import pointset
        return pointset(space, max(space.degree)+2 if n is None else int(n), target)
    bset = _as_boundary_set(space, target)
    label = target if isinstance(target, str) else "set"
    dim = space.dim
    n_per_dir = [(p + 1) if n is None else int(n) for p in space.degree]

    if space.cell_type == "simplex":
        from jaxiga.space.simplex import face_points, gauss_rule
        pts, wts = gauss_rule(dim - 1, space.degree[0] + 2 if n is None else int(n))
        return PointSet(
            elems=np.asarray(bset.elems),
            ref=np.asarray([face_points(dim, int(s), pts) for s in bset.sides]),
            weights=np.broadcast_to(wts, (len(bset.elems), len(wts))),
            sides=np.asarray(bset.sides), tag=f"boundary:{label}",
        )

    # one face rule per side code, then gathered per element
    face_rules = {}
    for code in np.unique(bset.sides):
        direction, end = divmod(int(code), 2)
        tangential = [d for d in range(dim) if d != direction]
        if tangential:
            face_pts, face_wts = _tensor_rule([n_per_dir[d] for d in tangential])
        else:  # 1D: the boundary of an interval is a point
            face_pts, face_wts = np.zeros((1, 0)), np.ones(1)

        pts = np.zeros((len(face_pts), dim))
        for k, d in enumerate(tangential):
            pts[:, d] = face_pts[:, k]
        pts[:, direction] = -1.0 if end == 0 else 1.0
        face_rules[int(code)] = (pts, face_wts)

    n_q = len(next(iter(face_rules.values()))[0])
    ref = np.zeros((len(bset.elems), n_q, dim))
    wts = np.zeros((len(bset.elems), n_q))
    for i, code in enumerate(bset.sides):
        ref[i], wts[i] = face_rules[int(code)]

    return PointSet(
        elems=np.asarray(bset.elems),
        ref=np.asarray(ref),
        weights=np.asarray(wts),
        sides=np.asarray(bset.sides),
        tag=f"boundary:{label}",
    )


def grid(space, n: int = 20) -> PointSet:
    """Uniform sample points per element, for postprocessing and plotting."""
    if space.cell_type == "mixed":
        from jaxiga.space.mixed import pointset
        return pointset(space, n, sampling=True)
    if space.cell_type == "simplex":
        from jaxiga.space.simplex import lattice
        if n < 2:
            raise ValueError("simplex grids require n >= 2")
        return PointSet(np.arange(space.n_elems), lattice(space.dim, n - 1), tag="grid")
    per_dir = [np.linspace(-1.0, 1.0, n) for _ in range(space.dim)]
    grids = np.meshgrid(*per_dir, indexing="ij")
    pts = np.stack([g.ravel(order="F") for g in grids], axis=1)
    return PointSet(
        elems=np.arange(space.n_elems),
        ref=np.asarray(pts),
        weights=None,
        tag="grid",
    )


def greville_1d(knot, deg):
    """Greville abscissae of a univariate B-spline basis.

    ``g_i = mean(U[i+1], ..., U[i+deg])``, one per basis function. These are the
    standard collocation points: each is the point where basis function ``i``
    is (essentially) centred.
    """
    knot = np.asarray(knot, dtype=float)
    n = len(knot) - deg - 1
    if deg == 0:
        return (knot[:n] + knot[1 : n + 1]) / 2
    return np.array([np.mean(knot[i + 1 : i + deg + 1]) for i in range(n)])


def _patch_element_offsets(space):
    """First global element index of each patch."""
    patch_of = np.asarray(space.elem_patch)
    n_patches = int(patch_of.max()) + 1 if len(patch_of) else 0
    return [int(np.argmax(patch_of == p)) for p in range(n_patches)]


def _reject_hierarchical(space, what):
    if space.cell_type in {"simplex", "mixed"}:
        raise NotImplementedError(f"{what} is not available on simplex spaces; use Galerkin quadrature")
    if getattr(space, "is_hierarchical", False):
        raise NotImplementedError(
            f"{what} is defined for a tensor-product basis; a hierarchical (THB) "
            f"space has no such construction (see section 23.7 of the "
            f"architecture guide)"
        )


def greville(space, tag: str = "interior", exclude_dofs=None) -> PointSet:
    """Greville abscissae, one per scalar basis function.

    Points shared across conforming interfaces are emitted once, matching the
    dof numbering. Each point is assigned to an element containing it; points
    on an element boundary go to the lower-numbered element.

    The set is ragged -- elements hold different numbers of Greville points --
    so it is represented with one point per entry (``n_q = 1``) and repeated
    element indices, rather than grouped per element like a quadrature rule.

    Parameters
    ----------
    exclude_dofs : array of int, optional
        Global scalar dofs whose points should be dropped, used by collocation
        to remove points belonging to prescribed dofs.
    """
    _reject_hierarchical(space, "Greville collocation")
    from jaxiga.space.bernstein import element_spans, locate

    if not space.patch_knots:
        raise ValueError("this FunctionSpace carries no patch knot vectors")

    dim = space.dim
    elem_vertex = np.asarray(space.elem_vertex)
    offsets = _patch_element_offsets(space)
    exclude = set() if exclude_dofs is None else {int(d) for d in np.asarray(exclude_dofs)}

    seen = set()
    pts_ref, pts_elem, pts_dof = [], [], []

    for p_idx, knots in enumerate(space.patch_knots):
        spans = [element_spans(np.asarray(kv, dtype=float), space.degree[d])
                 for d, kv in enumerate(knots)]
        per_dir = [greville_1d(knots[d], space.degree[d]) for d in range(dim)]

        grids = np.meshgrid(*per_dir, indexing="ij")
        # first direction fastest, matching the control-point ordering, so
        # local index k is the k-th scalar basis function of this patch
        local_pts = np.stack([g.ravel(order="F") for g in grids], axis=1)
        l2g = np.asarray(space.topology.local_to_global[p_idx])

        for local_idx, xi in enumerate(local_pts):
            gdof = int(l2g[local_idx])
            if gdof in seen or gdof in exclude:
                continue
            seen.add(gdof)

            elem_idx, ref = [], []
            for d in range(dim):
                lo_d, hi_d, _ = spans[d]
                e = locate(lo_d, hi_d, xi[d])
                elem_idx.append(e)
                ref.append(2 * (xi[d] - lo_d[e]) / (hi_d[e] - lo_d[e]) - 1)

            n_elem_per_dir = [len(s[0]) for s in spans]
            strides = np.array([1, *np.cumprod(n_elem_per_dir[:-1])], dtype=int)
            local_elem = int(sum(elem_idx[d] * strides[d] for d in range(dim)))

            pts_ref.append(ref)
            pts_elem.append(offsets[p_idx] + local_elem)
            pts_dof.append(gdof)

    ref = np.asarray(pts_ref).reshape(len(pts_ref), 1, dim)
    return PointSet(
        elems=np.asarray(pts_elem, dtype=int),
        ref=np.asarray(ref),
        weights=None,
        tag=tag,
        point_dofs=np.asarray(pts_dof, dtype=int),
    )


def boundary_greville(space, target, exclude_dofs=None) -> PointSet:
    """Greville points whose basis function lies on a boundary label or set."""
    _reject_hierarchical(space, "Greville collocation")
    bset = _as_boundary_set(space, target)
    label = target if isinstance(target, str) else "set"
    full = greville(space, tag=f"boundary:{label}", exclude_dofs=exclude_dofs)

    on_boundary = np.isin(full.point_dofs, np.asarray(bset.dofs))
    keep = np.flatnonzero(on_boundary)
    if keep.size == 0:
        raise ValueError(f"no Greville points lie on boundary {label!r}")

    elems = full.elems[keep]
    sides = _sides_for_elements(space, bset, elems)
    return PointSet(
        elems=elems,
        ref=np.asarray(np.asarray(full.ref)[keep]),
        weights=None,
        sides=sides,
        tag=f"boundary:{label}",
        point_dofs=full.point_dofs[keep],
    )


def _sides_for_elements(space, bset, elems):
    """Side code of each element within a boundary set."""
    lookup = {}
    for e, s in zip(np.asarray(bset.elems), np.asarray(bset.sides)):
        lookup.setdefault(int(e), int(s))
    missing = [int(e) for e in elems if int(e) not in lookup]
    if missing:
        raise ValueError(f"elements {missing[:5]} are not in this boundary set")
    return np.array([lookup[int(e)] for e in elems], dtype=int)


def interface_greville(space, interface):
    """Matched Greville point pairs on both sides of an interface.

    Returns two point sets whose entries correspond one-to-one, so a flux jump
    can be formed by subtracting the evaluations.
    """
    _reject_hierarchical(space, "Greville collocation")
    from jaxiga.geometry.multipatch import side_indices

    dim = space.dim
    if not space.patch_knots:
        raise ValueError("this FunctionSpace carries no patch knot vectors")

    sets = []
    for patch_idx, side in ((interface.patch_a, interface.side_a),
                            (interface.patch_b, interface.side_b)):
        sets.append(_side_greville(space, patch_idx, side))

    (dofs_a, elems_a, ref_a, side_a), (dofs_b, elems_b, ref_b, side_b) = sets

    # order both sides by shared global dof so entries correspond
    order_a = np.argsort(dofs_a)
    order_b = np.argsort(dofs_b)
    if not np.array_equal(dofs_a[order_a], dofs_b[order_b]):
        raise ValueError(
            f"interface {interface} does not share a matching set of Greville points"
        )

    make = lambda elems, ref, sides, order: PointSet(  # noqa: E731
        elems=elems[order],
        ref=jnp.asarray(ref[order]),
        weights=None,
        sides=sides[order],
        tag="interface",
        point_dofs=dofs_a[order_a],
    )
    return (make(elems_a, ref_a, side_a, order_a), make(elems_b, ref_b, side_b, order_b))


def _side_greville(space, patch_idx, side):
    """Greville points of one patch restricted to one of its sides."""
    from jaxiga.geometry.multipatch import side_indices
    from jaxiga.space.bernstein import element_spans, locate
    from jaxiga.geometry.nurbs import side_code

    dim = space.dim
    knots = space.patch_knots[patch_idx]
    spans = [element_spans(np.asarray(kv, dtype=float), space.degree[d])
             for d, kv in enumerate(knots)]
    per_dir = [greville_1d(knots[d], space.degree[d]) for d in range(dim)]
    n_per_dir = [len(g) for g in per_dir]

    direction = "uvw".index(side[0])
    end = int(side[1:])
    fixed = 0 if end == 0 else n_per_dir[direction] - 1

    ranges = [np.arange(n) for n in n_per_dir]
    ranges[direction] = np.array([fixed])
    mesh = np.meshgrid(*ranges, indexing="ij")
    strides = np.array([1, *np.cumprod(n_per_dir[:-1])], dtype=int)
    local = sum(m.ravel(order="F") * strides[d] for d, m in enumerate(mesh)).astype(int)

    idx_per_dir = [m.ravel(order="F") for m in mesh]
    offsets = _patch_element_offsets(space)
    n_elem_per_dir = [len(s[0]) for s in spans]
    e_strides = np.array([1, *np.cumprod(n_elem_per_dir[:-1])], dtype=int)

    elems, refs = [], []
    for k in range(len(local)):
        elem_idx, ref = [], []
        for d in range(dim):
            xi = per_dir[d][idx_per_dir[d][k]]
            lo_d, hi_d, _ = spans[d]
            e = locate(lo_d, hi_d, xi)
            elem_idx.append(e)
            ref.append(2 * (xi - lo_d[e]) / (hi_d[e] - lo_d[e]) - 1)
        elems.append(offsets[patch_idx] + int(sum(elem_idx[d] * e_strides[d] for d in range(dim))))
        refs.append(ref)

    l2g = np.asarray(space.topology.local_to_global[patch_idx])
    dofs = l2g[local]
    ref = np.asarray(refs).reshape(len(refs), 1, dim)
    sides = np.full(len(refs), side_code(side), dtype=int)
    return dofs, np.asarray(elems, dtype=int), ref, sides
