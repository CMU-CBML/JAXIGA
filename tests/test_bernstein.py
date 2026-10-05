"""Bernstein basis and Bezier extraction, checked against SciPy and exact identities."""

import numpy as np
import pytest

from jaxiga.space.bernstein import (
    bernstein_basis,
    bernstein_basis_2nd_deriv,
    bernstein_basis_deriv,
    bernstein_tensor,
    bezier_extraction,
    voigt_pairs,
)

DEGREES = [1, 2, 3, 4, 5]


def open_knots(deg, n_elem, repeat_interior=0):
    """Open knot vector on [0, 1] with n_elem uniform spans."""
    interior = np.repeat(np.linspace(0, 1, n_elem + 1)[1:-1], 1 + repeat_interior)
    return np.concatenate([np.zeros(deg + 1), interior, np.ones(deg + 1)])








@pytest.mark.parametrize("deg", DEGREES)
def test_partition_of_unity(deg):
    u = np.linspace(-1, 1, 13)
    np.testing.assert_allclose(bernstein_basis(u, deg).sum(axis=1), 1.0, rtol=1e-14)
    # Derivatives of a partition of unity sum to zero.
    np.testing.assert_allclose(bernstein_basis_deriv(u, deg).sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(bernstein_basis_2nd_deriv(u, deg).sum(axis=1), 0.0, atol=1e-11)


@pytest.mark.parametrize("deg", DEGREES)
def test_bernstein_derivatives_vs_finite_differences(deg):
    """Check first and second derivatives against finite differences."""
    u = np.linspace(-0.8, 0.8, 11)

    h = 1e-6
    d_fd = (bernstein_basis(u + h, deg) - bernstein_basis(u - h, deg)) / (2 * h)
    np.testing.assert_allclose(bernstein_basis_deriv(u, deg), d_fd, rtol=1e-6, atol=1e-8)

    # Second differences amplify roundoff as eps/h**2, so h must be much larger
    # here than for the first derivative: h=1e-4 balances that against the
    # O(h**2) truncation error.
    h = 1e-4
    d2_fd = (
        bernstein_basis(u + h, deg) - 2 * bernstein_basis(u, deg) + bernstein_basis(u - h, deg)
    ) / h**2
    np.testing.assert_allclose(bernstein_basis_2nd_deriv(u, deg), d2_fd, rtol=1e-6, atol=1e-7)






@pytest.mark.parametrize("dim,deg", [(1, (3,)), (2, (2, 3)), (3, (2, 2, 2))])
def test_tensor_second_derivatives_vs_finite_differences(dim, deg):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.7, 0.7, size=(6, dim))
    _, _, d2B = bernstein_tensor(pts, deg, order=2)
    h = 1e-4

    for k, (i, j) in enumerate(voigt_pairs(dim)):
        ei = np.zeros(dim)
        ei[i] = h
        ej = np.zeros(dim)
        ej[j] = h
        fd = (
            bernstein_tensor(pts + ei + ej, deg, order=0)[0]
            - bernstein_tensor(pts + ei - ej, deg, order=0)[0]
            - bernstein_tensor(pts - ei + ej, deg, order=0)[0]
            + bernstein_tensor(pts - ei - ej, deg, order=0)[0]
        ) / (4 * h**2)
        np.testing.assert_allclose(d2B[:, k, :], fd, rtol=1e-6, atol=1e-7)


def test_tensor_partition_of_unity_and_ordering():
    """First direction varies fastest in the local basis numbering."""
    deg = (2, 3)
    pts = np.array([[0.3, -0.4]])
    B, dB, d2B = bernstein_tensor(pts, deg, order=2)

    assert B.shape == (1, 12)
    np.testing.assert_allclose(B.sum(), 1.0, rtol=1e-14)
    np.testing.assert_allclose(dB.sum(axis=2), 0.0, atol=1e-12)
    np.testing.assert_allclose(d2B.sum(axis=2), 0.0, atol=1e-11)

    bu = bernstein_basis(pts[:, 0], deg[0])[0]
    bv = bernstein_basis(pts[:, 1], deg[1])[0]
    expected = np.array([bu[i] * bv[j] for j in range(deg[1] + 1) for i in range(deg[0] + 1)])
    np.testing.assert_allclose(B[0], expected, rtol=1e-14)


@pytest.mark.parametrize("deg", DEGREES)
@pytest.mark.parametrize("n_elem", [1, 2, 5])
@pytest.mark.parametrize("repeat_interior", [0, 1])
def test_bezier_extraction_matches_scipy(deg, n_elem, repeat_interior):
    from scipy.interpolate import BSpline

    if repeat_interior >= deg:
        pytest.skip("interior multiplicity must not exceed degree")
    knots = open_knots(deg, n_elem, repeat_interior)
    matrices, count = bezier_extraction(knots, deg)
    spans = np.flatnonzero(np.diff(knots) > 0)
    assert count == len(spans) == n_elem
    reference = np.linspace(-0.9, 0.9, 7)
    for span, matrix in zip(spans, matrices):
        x = knots[span] + (reference + 1) * (knots[span + 1] - knots[span]) / 2
        expected = BSpline.design_matrix(x, knots, deg).toarray()
        actual = bernstein_basis(reference, deg) @ matrix.T
        np.testing.assert_allclose(actual, expected[:, span - deg:span + 1],
                                   rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("deg", DEGREES)
def test_bernstein_basis_matches_scipy(deg):
    from scipy.interpolate import BSpline

    knots = np.r_[np.full(deg + 1, -1.), np.full(deg + 1, 1.)]
    spline = BSpline(knots, np.eye(deg + 1), deg)
    x = np.linspace(-1, 1, 17)
    np.testing.assert_allclose(bernstein_basis(x, deg), spline(x), atol=1e-14)
    np.testing.assert_allclose(bernstein_basis_deriv(x, deg), spline(x, nu=1),
                               atol=1e-13)
    np.testing.assert_allclose(bernstein_basis_2nd_deriv(x, deg), spline(x, nu=2),
                               atol=1e-12)
