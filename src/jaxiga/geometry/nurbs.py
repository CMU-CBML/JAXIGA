"""NURBS patch representation and refinement operators.

Setup stage. The refinement routines (:func:`_bspdegelev`, :func:`_bspkntins`)
are NumPy ports of the legacy ``utils/splines.py`` helpers, themselves ports of
the Octave ``nurbs`` toolbox.

Conventions (chosen to match the legacy code so oracle tests compare directly):

* Control points are a tensor-product grid flattened with the **first
  parametric direction fastest**: ``i = i_u + i_v*n_u + i_w*n_u*n_v``. This is
  the ordering the IEN construction in ``utils_iga/IGA.py`` assumes
  (``t1 + t2*len_u``).
* Sides are named ``"u0"``, ``"u1"``, ``"v0"``, ``"v1"`` (and ``"w0"``,
  ``"w1"`` in 3D) for parametric value 0/1 of each direction. The legacy
  ``"left"/"right"/"down"/"up"`` correspond to ``u0/u1/v0/v1``.
* The parametric domain is ``[0, 1]^dim``; knot vectors are normalized on
  construction.
* Control points are stored **unweighted**, with weights separate. Degree
  elevation and knot insertion act on homogeneous (weighted) coordinates, so
  those are formed and undone internally.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import binom

from jaxiga.config import TOL

DIR_NAMES = ("u", "v", "w")


def side_name(direction: int, end: int) -> str:
    """``(0, 0) -> "u0"``, ``(1, 1) -> "v1"``, ..."""
    return f"{DIR_NAMES[direction]}{end}"


def side_code(name: str) -> int:
    """Inverse of :func:`side_name`, as ``2*direction + end``."""
    direction = DIR_NAMES.index(name[0])
    return 2 * direction + int(name[1:])


# Legacy side names, accepted everywhere a side is named.
LEGACY_SIDES = {"left": "u0", "right": "u1", "down": "v0", "up": "v1"}


def canonical_side(name: str) -> str:
    return LEGACY_SIDES.get(name, name)


# --------------------------------------------------------------------------
# Univariate B-spline refinement (NumPy, setup stage)
# --------------------------------------------------------------------------


def _bspdegelev(d, c, k, t):
    """Degree-elevate a univariate B-spline ``t`` times.

    Ported from ``utils/splines.py`` (Octave ``nurbs`` toolbox algorithm).

    Parameters
    ----------
    d : int
        Current degree.
    c : (mc, nc) array
        Control points; ``mc`` components, ``nc`` control points.
    k : 1D array
        Knot vector.
    t : int
        Number of times to raise the degree.

    Returns
    -------
    ic : (mc, nc_new) array
    ik : 1D array
    """
    c = np.asarray(c, dtype=float)
    k = np.asarray(k, dtype=float)
    mc, nc = c.shape
    ic = np.zeros((mc, nc * (t + 1)))
    n = nc - 1
    bezalfs = np.zeros((d + 1, d + t + 1))
    ebpts = np.zeros((mc, d + t + 1))
    Nextbpts = np.zeros((mc, d + 1))
    alfs = np.zeros(d)

    m = n + d + 1
    ph = d + t
    ph2 = int(np.floor(ph / 2))

    # Bezier degree-elevation coefficients
    bezalfs[0, 0] = 1.0
    bezalfs[d, ph] = 1.0
    for i in range(1, ph2 + 1):
        inv = 1 / binom(ph, i)
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = inv * binom(d, j) * binom(t, i - j)
    for i in range(ph2 + 1, ph):
        mpi = min(d, i)
        for j in range(max(0, i - t), mpi + 1):
            bezalfs[j, i] = bezalfs[d - j, ph - i]

    mh = ph
    kind = ph + 1
    r = -1
    a = d
    b = d + 1
    cind = 1
    ua = k[0]

    ic[0:mc, 0] = c[0:mc, 0]
    ik = (ua * np.ones(ph + 1)).tolist()

    bpts = c.copy()

    while b < m:
        i = b
        while b < m and k[b] == k[b + 1]:
            b += 1
        mul = b - i + 1
        mh = mh + mul + t
        ub = k[b]
        oldr = r
        r = d - mul

        lbz = int(np.floor((oldr + 2) / 2)) if oldr > 0 else 1
        rbz = int(np.floor((r + 1) / 2)) if r > 0 else ph

        if r > 0:
            # Insert knot u[b] r times to isolate a Bezier segment
            numer = ub - ua
            for q in range(d, mul, -1):
                alfs[q - mul - 1] = numer / (k[a + q] - ua)
            for j in range(1, r + 1):
                save = r - j
                s = mul + j
                for q in range(d, s - 1, -1):
                    for ii in range(0, mc):
                        tmp1 = alfs[q - s] * bpts[ii, q]
                        tmp2 = (1 - alfs[q - s + 1]) * bpts[ii, q - 1]
                        bpts[ii, q] = tmp1 + tmp2
                Nextbpts[:, save] = bpts[:, d]

        # Degree-elevate the Bezier segment
        for i in range(lbz, ph + 1):
            ebpts[:, i] = np.zeros(mc)
            mpi = min(d, i)
            for j in range(max(0, i - t), mpi + 1):
                for ii in range(0, mc):
                    ebpts[ii, i] = ebpts[ii, i] + bezalfs[j, i] * bpts[ii, j]

        if oldr > 1:
            # Remove the knot k[a] oldr times
            first = kind - 2
            last = kind
            den = ub - ua
            bet = int(np.floor((ub - ik[kind - 1] / den)))
            for tr in range(1, oldr):
                i = first
                j = last
                kj = j - kind + 1
                while j - i > tr:
                    if i < cind:
                        alf = (ub - ik[i]) / (ua - ik[i])
                        ic[:, i] = alf * ic[:, i] + (1 - alf) * ic[:, i - 1]
                    if j >= lbz:
                        if j - tr <= kind - ph + oldr:
                            gam = (ub - ik[j - tr]) / den
                            ebpts[:, kj] = gam * ebpts[:, kj] + (1 - gam) * ebpts[:, kj + 1]
                        else:
                            ebpts[:, kj] = bet * ebpts[:, kj] + (1 - bet) * ebpts[:, kj + 1]
                    i += 1
                    j -= 1
                    kj -= 1
                first -= 1
                last += 1

        if a != d:
            for i in range(0, ph - oldr):
                ik.append(ua)
                kind += 1

        for j in range(lbz, rbz + 1):
            for ii in range(0, mc):
                ic[ii, cind] = ebpts[ii, j]
            cind += 1

        if b < m:
            bpts[:, 0:r] = Nextbpts[:, 0:r]
            bpts[:, r : d + 1] = c[:, b - d + r : b + 1]
            a = b
            b += 1
            ua = ub
        else:
            for i in range(0, ph + 1):
                ik.append(ub)

    return ic[:, 0:cind], np.asarray(ik, dtype=float)


def _findspan(n, u, knot):
    """Knot span index of each parametric value in ``u``."""
    knot = np.asarray(knot, dtype=float)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    if np.min(u) < knot[0] or np.max(u) > knot[-1]:
        raise ValueError("parametric value outside the knot span")
    s = np.zeros(len(u), dtype=int)
    for j in range(len(s)):
        if u[j] == knot[n + 1]:
            s[j] = n
            continue
        s[j] = np.argwhere(knot <= u[j])[-1].item()
    return s


def _bspkntins(d, c, k, u):
    """Insert knots ``u`` into a univariate B-spline.

    Ported from ``utils/splines.py``.
    """
    c = np.asarray(c, dtype=float)
    mc, nc = c.shape
    u = np.sort(np.asarray(u, dtype=float))
    k = np.asarray(k, dtype=float)
    nu = len(u)
    nk = len(k)

    ic = np.zeros((mc, nc + nu))
    ik = np.zeros(nk + nu)

    n = nc - 1
    r = nu - 1
    m = n + d + 1
    a = _findspan(n, [u[0]], k)[0]
    b = _findspan(n, [u[r]], k)[0] + 1

    ic[:, 0 : a - d + 1] = c[:, 0 : a - d + 1]
    ic[:, b + nu - 1 : nc + nu] = c[:, b - 1 : nc]
    ik[0 : a + 1] = k[0 : a + 1]
    ik[b + d + nu : m + nu + 1] = k[b + d : m + 1]

    ii = b + d - 1
    ss = ii + nu

    for jj in range(r, -1, -1):
        ind = np.arange(a + 1, ii + 1)
        ind = ind[np.where(k[ind] >= u[jj])]
        ic[:, ind + ss - ii - d - 1] = c[:, ind - d - 1]
        ik[ind + ss - ii] = k[ind]
        ii -= len(ind)
        ss -= len(ind)

        ic[:, ss - d - 1] = ic[:, ss - d]
        for lo in range(1, d + 1):
            ind = ss - d + lo
            alfa = ik[ss + lo] - u[jj]
            if abs(alfa) < TOL:
                ic[:, ind - 1] = ic[:, ind]
            else:
                alfa = alfa / (ik[ss + lo] - k[ii - d + lo])
                ic[:, ind - 1] = alfa * ic[:, ind - 1] + (1 - alfa) * ic[:, ind]
        ik[ss] = u[jj]
        ss -= 1

    return ic, ik


# --------------------------------------------------------------------------
# Patch
# --------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Patch:
    """A single NURBS patch.

    ``knots``, ``degree`` and ``labels`` are pytree *metadata*: they must be
    hashable, so knots are stored as nested tuples of floats rather than arrays
    (JAX forbids arrays in metadata fields). ``ctrl_pts`` and ``weights`` are
    pytree *leaves*, which is what makes shape derivatives work.
    """

    ctrl_pts: jnp.ndarray  # (n_cp, dim_phys), unweighted
    weights: jnp.ndarray  # (n_cp,)
    knots: tuple = dataclasses.field(metadata=dict(static=True))
    degree: tuple = dataclasses.field(metadata=dict(static=True))
    labels: tuple = dataclasses.field(default=(), metadata=dict(static=True))

    # -- construction -------------------------------------------------------

    @classmethod
    def create(cls, knots, degree, ctrl_pts, weights=None, labels=None) -> "Patch":
        """Build a patch, normalizing knots to [0, 1] and validating sizes."""
        degree = tuple(int(p) for p in degree)
        knots = tuple(_normalize_knots(kv) for kv in knots)
        if len(knots) != len(degree):
            raise ValueError(f"{len(knots)} knot vectors but {len(degree)} degrees")

        ctrl_pts = jnp.asarray(ctrl_pts, dtype=float)
        if ctrl_pts.ndim != 2:
            raise ValueError(f"ctrl_pts must be (n_cp, dim_phys), got shape {ctrl_pts.shape}")
        n_cp = ctrl_pts.shape[0]

        expected = [len(kv) - p - 1 for kv, p in zip(knots, degree)]
        if int(np.prod(expected)) != n_cp:
            raise ValueError(
                f"knots/degree imply {expected} = {int(np.prod(expected))} control points "
                f"but {n_cp} were given"
            )

        weights = jnp.ones(n_cp) if weights is None else jnp.asarray(weights, dtype=float)
        if weights.shape != (n_cp,):
            raise ValueError(f"weights must be ({n_cp},), got {weights.shape}")

        return cls(ctrl_pts, weights, knots, degree, _labels_to_tuple(labels))

    # -- derived properties -------------------------------------------------

    @property
    def dim(self) -> int:
        """Number of parametric directions."""
        return len(self.degree)

    @property
    def dim_phys(self) -> int:
        return self.ctrl_pts.shape[1]

    @property
    def n_cp_per_dir(self) -> tuple:
        return tuple(len(kv) - p - 1 for kv, p in zip(self.knots, self.degree))

    @property
    def n_cp(self) -> int:
        return int(np.prod(self.n_cp_per_dir))

    @property
    def label_map(self) -> dict:
        """Side name -> user label, as a dict."""
        return dict(self.labels)

    def knot_arrays(self) -> list:
        return [np.asarray(kv, dtype=float) for kv in self.knots]

    def homogeneous(self) -> jnp.ndarray:
        """(n_cp, dim_phys+1) weighted coordinates + weights."""
        return jnp.concatenate([self.ctrl_pts * self.weights[:, None], self.weights[:, None]], 1)

    # -- labels -------------------------------------------------------------

    def with_labels(self, **side_labels: str) -> "Patch":
        """Attach user labels to sides, e.g. ``patch.with_labels(v0="hole")``."""
        merged = self.label_map
        for side, label in side_labels.items():
            side = canonical_side(side)
            if side not in self.side_names():
                raise ValueError(f"unknown side {side!r}; expected one of {self.side_names()}")
            merged[side] = label
        return dataclasses.replace(self, labels=_labels_to_tuple(merged))

    def side_names(self) -> tuple:
        return tuple(side_name(d, e) for d in range(self.dim) for e in (0, 1))

    # -- refinement ---------------------------------------------------------

    def reverse(self, axis: int = 0) -> "Patch":
        """Reverse a parametric direction, preserving the physical geometry."""
        if not isinstance(axis, (int, np.integer)) or not 0 <= axis < self.dim:
            raise ValueError("axis must identify a parametric direction")
        ids = np.flip(np.arange(self.n_cp).reshape(self.n_cp_per_dir, order="F"), axis).ravel(order="F")
        knots = list(self.knots)
        knots[axis] = tuple(1 - np.asarray(knots[axis])[::-1])
        labels = self.label_map
        a, b = side_name(axis, 0), side_name(axis, 1)
        labels = {(b if k == a else a if k == b else k): v for k, v in labels.items()}
        return Patch.create(knots, self.degree, self.ctrl_pts[ids], self.weights[ids], labels)

    def elevate(self, target_degree) -> "Patch":
        """Raise the degree **to** ``target_degree`` in each direction.

        Note this is a target, not an increment: ``elevate(3)`` on a bilinear
        patch produces a bicubic one. A scalar applies to every direction.
        """
        if np.isscalar(target_degree):
            target = (int(target_degree),) * self.dim
        else:
            target = tuple(int(p) for p in target_degree)
        if len(target) != self.dim:
            raise ValueError(f"expected {self.dim} target degrees, got {len(target)}")

        times = [t - p for t, p in zip(target, self.degree)]
        if any(t < 0 for t in times):
            raise ValueError(
                f"cannot lower degree: patch has degree {self.degree}, target {target}"
            )
        if all(t == 0 for t in times):
            return self

        coefs = np.asarray(self.homogeneous())
        knots = self.knot_arrays()
        n_per_dir = list(self.n_cp_per_dir)
        degree = list(self.degree)

        for d in range(self.dim):
            if times[d] == 0:
                continue
            coefs, new_kv = _apply_along_dir(
                coefs, n_per_dir, d, lambda c: _bspdegelev(degree[d], c, knots[d], times[d])
            )
            knots[d] = new_kv
            degree[d] = target[d]
            n_per_dir[d] = len(new_kv) - degree[d] - 1

        return _from_homogeneous(coefs, knots, degree, self.labels)

    def insert_knots(self, *knots_per_dir) -> "Patch":
        """Insert knots, one sequence per parametric direction."""
        if len(knots_per_dir) != self.dim:
            raise ValueError(f"expected {self.dim} knot sequences, got {len(knots_per_dir)}")

        coefs = np.asarray(self.homogeneous())
        knots = self.knot_arrays()
        n_per_dir = list(self.n_cp_per_dir)
        degree = list(self.degree)

        for d in range(self.dim):
            new = np.asarray(knots_per_dir[d], dtype=float)
            if new.size == 0:
                continue
            if new.min() <= 0.0 or new.max() >= 1.0:
                raise ValueError(f"interior knots must lie in (0, 1); got {new}")
            coefs, new_kv = _apply_along_dir(
                coefs, n_per_dir, d, lambda c: _bspkntins(degree[d], c, knots[d], new)
            )
            knots[d] = new_kv
            n_per_dir[d] = len(new_kv) - degree[d] - 1

        return _from_homogeneous(coefs, knots, degree, self.labels)

    def refine(self, n: int = 1) -> "Patch":
        """``n`` rounds of uniform bisection of every non-empty knot span."""
        patch = self
        for _ in range(int(n)):
            new_knots = []
            for kv in patch.knot_arrays():
                uniq = np.unique(kv)
                new_knots.append((uniq[:-1] + uniq[1:]) / 2)
            patch = patch.insert_knots(*new_knots)
        return patch

    def refinement_operator(self, fine: "Patch") -> np.ndarray:
        """Linear map ``P`` with ``fine.homogeneous() == P @ self.homogeneous()``.

        ``fine`` must be a refinement of ``self`` (same geometry, richer space).
        Enables coarse-design / fine-analysis workflows: optimize a coarse
        control net, analyze on ``P @ cpts``. The map is linear, so gradients
        flow through it.
        """
        for d in range(self.dim):
            if fine.degree[d] < self.degree[d]:
                raise ValueError(
                    f"fine patch has lower degree in direction {d}: "
                    f"{fine.degree[d]} < {self.degree[d]}"
                )

        # Refinement is linear in the control coefficients, so pushing the
        # identity through the same operations yields the operator itself.
        eye = Patch.create(self.knots, self.degree, np.eye(self.n_cp), np.ones(self.n_cp))
        elevated = eye.elevate(fine.degree)

        inserts = []
        for d in range(self.dim):
            have = np.asarray(elevated.knots[d], dtype=float)
            want = np.asarray(fine.knots[d], dtype=float)
            inserts.append(_multiset_difference(want, have))
        refined = elevated.insert_knots(*inserts)

        if refined.n_cp != fine.n_cp:
            raise ValueError(
                f"fine patch is not a refinement of this one: reconstructed "
                f"{refined.n_cp} control points, fine patch has {fine.n_cp}"
            )
        # ctrl_pts were the identity with unit weights, so they are P itself.
        return np.asarray(refined.ctrl_pts)

    def summary(self) -> str:
        return (
            f"Patch(dim={self.dim}, degree={self.degree}, "
            f"n_cp={self.n_cp} {self.n_cp_per_dir}, labels={self.label_map})"
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _normalize_knots(kv) -> tuple:
    kv = np.asarray(kv, dtype=float)
    lo, hi = kv[0], kv[-1]
    if hi - lo <= 0:
        raise ValueError(f"degenerate knot vector with span [{lo}, {hi}]")
    return tuple(((kv - lo) / (hi - lo)).tolist())


def _labels_to_tuple(labels) -> tuple:
    if labels is None:
        return ()
    if isinstance(labels, dict):
        items = labels.items()
    else:
        items = labels
    return tuple(sorted((canonical_side(k), v) for k, v in items))


def _multiset_difference(want: np.ndarray, have: np.ndarray) -> np.ndarray:
    """Knots present in ``want`` beyond those in ``have``, respecting multiplicity."""
    remaining = list(have)
    extra = []
    for value in want:
        match = next((i for i, r in enumerate(remaining) if abs(r - value) < TOL), None)
        if match is None:
            extra.append(value)
        else:
            remaining.pop(match)
    return np.asarray(extra, dtype=float)


def _apply_along_dir(coefs, n_per_dir, d, fn):
    """Apply a univariate spline operator along parametric direction ``d``.

    ``coefs`` is ``(n_cp, n_comp)`` flattened with direction 0 fastest, so a
    C-order reshape gives axes ``[i_{D-1}, ..., i_0, comp]`` and direction ``d``
    lives at axis ``D-1-d``. That axis is moved last so the operator sees a
    ``(everything_else, n_d)`` matrix, then moved back.
    """
    coefs = np.asarray(coefs, dtype=float)
    dim = len(n_per_dir)
    n_comp = coefs.shape[1]

    grid = coefs.reshape(*reversed(n_per_dir), n_comp)
    grid = np.moveaxis(grid, dim - 1 - d, -1)  # (..., comp, n_d)
    rest = grid.shape[:-1]
    flat = grid.reshape(-1, grid.shape[-1])

    out, new_kv = fn(flat)

    out = out.reshape(*rest, out.shape[-1])
    out = np.moveaxis(out, -1, dim - 1 - d)
    return out.reshape(-1, n_comp), np.asarray(new_kv, dtype=float)


def _from_homogeneous(coefs, knots, degree, labels) -> Patch:
    """Rebuild a patch from homogeneous coefficients."""
    coefs = np.asarray(coefs, dtype=float)
    weights = coefs[:, -1]
    ctrl_pts = coefs[:, :-1] / weights[:, None]
    return Patch.create(knots, degree, ctrl_pts, weights, labels)


def evaluate_patch(patch: Patch, params: np.ndarray) -> np.ndarray:
    """Evaluate the geometry map at parametric points (setup-stage, NumPy).

    Used for validation and for point-location; the traced evaluation path is
    :func:`jaxiga.space.evaluation.evaluate`.

    Parameters
    ----------
    patch : Patch
    params : (n_pts, dim) array of parametric coordinates in [0, 1]^dim

    Returns
    -------
    (n_pts, dim_phys) array of physical coordinates.
    """
    from jaxiga.geometry.simplex import SimplexPatch, evaluate_simplex
    if isinstance(patch, SimplexPatch):
        return evaluate_simplex(patch, params)

    from jaxiga.space.bernstein import bernstein_tensor, bezier_extraction, element_spans, locate

    params = np.atleast_2d(np.asarray(params, dtype=float))
    dim = patch.dim
    knots = patch.knot_arrays()
    n_per_dir = patch.n_cp_per_dir

    extraction = [bezier_extraction(knots[d], patch.degree[d])[0] for d in range(dim)]
    spans = [element_spans(knots[d], patch.degree[d]) for d in range(dim)]
    offsets = np.array([1, *np.cumprod(n_per_dir[:-1])])

    cw = np.asarray(patch.homogeneous())
    out = np.zeros((len(params), patch.dim_phys))

    for q, xi in enumerate(params):
        # locate the element and map into the reference cube
        first, ref, local = [], [], []
        for d in range(dim):
            lo_d, hi_d, first_d = spans[d]
            e = locate(lo_d, hi_d, xi[d])
            first.append(first_d[e])
            ref.append(2 * (xi[d] - lo_d[e]) / (hi_d[e] - lo_d[e]) - 1)
            local.append(extraction[d][e])

        B, _, _ = bernstein_tensor(np.array([ref]), patch.degree, order=0)
        # tensor-product extraction, first direction fastest (kron of later
        # directions first), matching np.kron(C_v, C_u) in IGAMesh2D
        C = local[-1]
        for d in range(dim - 2, -1, -1):
            C = np.kron(C, local[d])
        N = C @ B[0]

        # gather the supported control points
        rng = [np.arange(first[d], first[d] + patch.degree[d] + 1) for d in range(dim)]
        mesh = np.meshgrid(*rng, indexing="ij")
        flat = sum(m.ravel(order="F") * offsets[d] for d, m in enumerate(mesh))

        num = N @ cw[flat]
        out[q] = num[:-1] / num[-1]

    return out
