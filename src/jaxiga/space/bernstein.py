"""Bernstein bases and Bezier extraction.

Setup stage: pure NumPy, runs once per space, produces constant arrays.

This unifies the two legacy copies (``utils/bernstein.py`` and
``utils_iga/bernstein.py``), which had drifted apart only in which derivative
orders they implemented.

Reference interval is ``[-1, 1]`` throughout, matching the legacy code and the
Gauss-Legendre rules that feed it.
"""

import numpy as np


def bezier_extraction(knot, deg):
    """Bezier extraction operators for a univariate B-spline.

    Algorithm 1 of Borden et al., "Isogeometric finite element data structures
    based on Bezier extraction". Ported verbatim from the legacy implementation
    so that oracle comparisons are exact.

    Parameters
    ----------
    knot : 1D array of floats
        Open knot vector.
    deg : int
        Polynomial degree.

    Returns
    -------
    C : list of (deg+1, deg+1) arrays
        One extraction operator per non-empty knot span.
    nb : int
        Number of non-empty knot spans (elements).
    """
    knot = np.asarray(knot, dtype=float)
    m = len(knot) - deg - 1
    a = deg + 1
    b = a + 1
    C = [np.eye(deg + 1)]
    nb = 1

    while b <= m:
        C.append(np.eye(deg + 1))
        i = b
        while (b <= m) and (knot[b] == knot[b - 1]):
            b = b + 1
        multiplicity = b - i + 1
        alphas = np.zeros(deg - multiplicity)
        if multiplicity < deg:
            numerator = knot[b - 1] - knot[a - 1]
            for j in range(deg, multiplicity, -1):
                alphas[j - multiplicity - 1] = numerator / (knot[a + j - 1] - knot[a - 1])
            r = deg - multiplicity
            for j in range(1, r + 1):
                save = r - j + 1
                s = multiplicity + j
                for k in range(deg + 1, s, -1):
                    alpha = alphas[k - s - 1]
                    C[nb - 1][:, k - 1] = (
                        alpha * C[nb - 1][:, k - 1] + (1 - alpha) * C[nb - 1][:, k - 2]
                    )
                if b <= m:
                    C[nb][save - 1 : save + j, save - 1] = C[nb - 1][deg - j : deg + 1, deg]
            nb = nb + 1
            if b <= m:
                a = b
                b = b + 1
        elif multiplicity == deg:
            if b <= m:
                nb = nb + 1
                a = b
                b = b + 1

    return C, nb


def bernstein_basis(uhat, deg):
    """Bernstein polynomials of degree ``deg`` at points ``uhat`` in [-1, 1].

    Returns an array of shape ``(len(uhat), deg + 1)``.
    """
    uhat = np.atleast_1d(np.asarray(uhat, dtype=float))
    B = np.zeros((len(uhat), deg + 1))
    B[:, 0] = 1.0
    u1 = 1 - uhat
    u2 = 1 + uhat

    for j in range(1, deg + 1):
        saved = 0.0
        for k in range(0, j):
            temp = B[:, k].copy()
            B[:, k] = saved + u1 * temp
            saved = u2 * temp
        B[:, j] = saved
    return B / np.power(2.0, deg)


def bernstein_basis_deriv(uhat, deg):
    """First derivatives of the Bernstein basis w.r.t. the reference coordinate.

    Returns an array of shape ``(len(uhat), deg + 1)``.
    """
    uhat = np.atleast_1d(np.asarray(uhat, dtype=float))
    if deg == 0:
        return np.zeros((len(uhat), 1))

    u1 = 1 - uhat
    u2 = 1 + uhat
    dB = np.zeros((len(uhat), deg))
    dB[:, 0] = 1.0
    for j in range(1, deg):
        saved = 0.0
        for k in range(0, j):
            temp = dB[:, k].copy()
            dB[:, k] = saved + u1 * temp
            saved = u2 * temp
        dB[:, j] = saved
    dB = dB / np.power(2.0, deg)

    # Pad with a zero column on each side, then difference: the degree-p
    # derivative is p * (B_{k-1}^{p-1} - B_k^{p-1}).
    zero = np.zeros((len(uhat), 1))
    dB = np.concatenate((zero, dB, zero), axis=1)
    return (dB[:, 0:-1] - dB[:, 1:]) * deg


def bernstein_basis_2nd_deriv(uhat, deg):
    """Second derivatives of the Bernstein basis w.r.t. the reference coordinate.

    Returns an array of shape ``(len(uhat), deg + 1)``. Obtained by applying the
    difference formula of :func:`bernstein_basis_deriv` twice.
    """
    uhat = np.atleast_1d(np.asarray(uhat, dtype=float))
    if deg < 2:
        return np.zeros((len(uhat), deg + 1))

    u1 = 1 - uhat
    u2 = 1 + uhat
    ddB = np.zeros((len(uhat), deg - 1))
    ddB[:, 0] = 1.0
    for j in range(1, deg - 1):
        saved = 0.0
        for k in range(0, j):
            temp = ddB[:, k].copy()
            ddB[:, k] = saved + u1 * temp
            saved = u2 * temp
        ddB[:, j] = saved
    ddB = ddB / np.power(2.0, deg)

    zero = np.zeros((len(uhat), 1))
    ddB = np.concatenate((zero, ddB, zero), axis=1)
    ddB = (ddB[:, 0:-1] - ddB[:, 1:]) * (deg - 1)
    ddB = np.concatenate((zero, ddB, zero), axis=1)
    return (ddB[:, 0:-1] - ddB[:, 1:]) * deg


def _tensor(factors):
    """Per-point tensor product of per-direction ``(n_pts, deg_d + 1)`` factors.

    Local basis functions are numbered with the **first direction fastest**
    (``i_0 + i_1*(p_0+1) + ...``), matching the IEN convention
    ``t1 + t2*len_u`` in ``utils_iga/IGA.py``. Building the outer product in
    reverse direction order puts the basis axes as ``(b_{dim-1}, ..., b_0)``,
    so a plain C-order reshape yields exactly that numbering.
    """
    out = factors[-1]
    for d in range(len(factors) - 2, -1, -1):
        out = np.einsum("q...,qb->q...b", out, factors[d])
    return out.reshape(out.shape[0], -1)


def bernstein_tensor(pts, deg, order=1):
    """Tensor-product Bernstein basis at arbitrary reference points.

    Dimension-generic replacement for the legacy ``bernstein_basis_2d`` /
    ``bernstein_basis_3d``, which were restricted to Cartesian grids.

    Reference points are static data, so this is evaluated once at setup in
    NumPy; only the extraction operator, control points and weights are traced
    downstream.

    Parameters
    ----------
    pts : (n_pts, dim) array
        Points in the reference cube ``[-1, 1]^dim``.
    deg : sequence of int
        Degree per direction.
    order : int
        Highest derivative order to return (0, 1 or 2).

    Returns
    -------
    B : (n_pts, n_local) array
    dB : (n_pts, dim, n_local) array or None
        First derivatives w.r.t. each reference coordinate; None if order < 1.
    d2B : (n_pts, n_d2, n_local) array or None
        Second derivatives in Voigt order (2D: uu, uv, vv); None if order < 2.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    dim = pts.shape[1]
    if dim != len(deg):
        raise ValueError(f"points have dim {dim} but {len(deg)} degrees were given")

    vals = [bernstein_basis(pts[:, d], deg[d]) for d in range(dim)]
    ders = [bernstein_basis_deriv(pts[:, d], deg[d]) for d in range(dim)] if order >= 1 else None
    ders2 = (
        [bernstein_basis_2nd_deriv(pts[:, d], deg[d]) for d in range(dim)] if order >= 2 else None
    )

    B = _tensor(vals)

    dB = None
    if order >= 1:
        dB = np.stack(
            [_tensor([ders[k] if k == d else vals[k] for k in range(dim)]) for d in range(dim)],
            axis=1,
        )

    d2B = None
    if order >= 2:
        cols = []
        for i, j in voigt_pairs(dim):
            if i == j:
                factors = [ders2[k] if k == i else vals[k] for k in range(dim)]
            else:
                factors = [ders[k] if k in (i, j) else vals[k] for k in range(dim)]
            cols.append(_tensor(factors))
        d2B = np.stack(cols, axis=1)

    return B, dB, d2B


def bernstein_basis_jnp(uhat, deg):
    """Bernstein basis at a single **traced** point, in JAX.

    The NumPy routines above cover the normal path, where evaluation points are
    static. This one exists for the collocation fallback, which differentiates
    the field with respect to the parametric coordinate and therefore needs the
    basis itself to be traceable.

    Mirrors :func:`bernstein_basis` exactly, including its in-place update
    order, via a Python list of scalars (``deg`` is small and static).
    """
    import jax.numpy as jnp

    B = [jnp.ones(())] + [jnp.zeros(()) for _ in range(deg)]
    u1 = 1 - uhat
    u2 = 1 + uhat
    for j in range(1, deg + 1):
        saved = jnp.zeros(())
        for k in range(j):
            temp = B[k]
            B[k] = saved + u1 * temp
            saved = u2 * temp
        B[j] = saved
    return jnp.stack(B) / 2.0**deg


def element_spans(knot, deg):
    """Non-empty knot spans and the first basis function supported on each.

    The first-basis index is ``i - deg`` where ``i`` indexes the **full** knot
    vector, matching ``IGAMesh2D.makeIEN`` (``t1 in range(i-deg, i+1)``). Using
    the position among *unique* knots instead is only equivalent when there are
    no repeated interior knots, and silently gives wrong connectivity for
    C^0 parametrizations such as the plate-with-hole patch.

    Returns
    -------
    lo, hi : (n_elem,) arrays
        Span bounds in parameter space.
    first : (n_elem,) int array
        Index of the first basis function supported on each span.
    """
    knot = np.asarray(knot, dtype=float)
    lo, hi, first = [], [], []
    for i in range(len(knot) - 1):
        if knot[i + 1] > knot[i]:
            lo.append(knot[i])
            hi.append(knot[i + 1])
            first.append(i - deg)
    return np.asarray(lo), np.asarray(hi), np.asarray(first, dtype=int)


def locate(lo, hi, value):
    """Index of the span containing ``value``; the last span is closed at ``hi``."""
    e = int(np.searchsorted(hi, value, side="left"))
    return min(max(e, 0), len(lo) - 1)


def voigt_pairs(dim):
    """Index pairs of the second-derivative Voigt ordering used by this package.

    2D -> [(0,0), (0,1), (1,1)];  3D -> [(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)].
    """
    return [(i, j) for i in range(dim) for j in range(i, dim)]
