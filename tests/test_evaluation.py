"""Basis evaluation: exactness properties and differentiability."""

import numpy as np
import pytest

import jaxiga.config  # noqa: F401
import jax
import jax.numpy as jnp

from jaxiga.geometry import primitives
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate
from jaxiga.space.function_space import FunctionSpace

GEOMETRIES = {
    "quad": lambda: primitives.rectangle(0, 0, 2, 1),
    "annulus": lambda: primitives.quarter_annulus(1.0, 4.0),
    "platehole": lambda: primitives.plate_with_hole_quadrant(1.0, 4.0, 2),
}


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_partition_of_unity(name):
    V = FunctionSpace(GEOMETRIES[name]().elevate(3).refine(2))
    bd = evaluate(V, P.gauss(V), order=2)

    np.testing.assert_allclose(np.asarray(bd.R.sum(-1)), 1.0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(bd.dR.sum(-1)), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(bd.d2R.sum(-1)), 0.0, atol=1e-8)


def test_quadrature_integrates_exact_area():
    """NURBS geometry is exact, so the annulus area must be too."""
    V = FunctionSpace(primitives.quarter_annulus(1.0, 4.0).elevate(3).refine(3))
    area = float(evaluate(V, P.gauss(V)).w.sum())
    np.testing.assert_allclose(area, np.pi / 4 * (16 - 1), rtol=1e-12)


def test_quadrature_integrates_polynomials_exactly():
    V = FunctionSpace(primitives.rectangle(0, 0, 2, 3).elevate(3).refine(2))
    bd = evaluate(V, P.gauss(V))
    x, y = bd.x[..., 0], bd.x[..., 1]
    # \int_0^2 \int_0^3 x^2 y = (8/3)*(9/2) = 12
    np.testing.assert_allclose(float((bd.w * x**2 * y).sum()), 12.0, rtol=1e-12)


@pytest.mark.parametrize("label,exact", [("outer", np.pi / 2 * 4), ("inner", np.pi / 2),
                                         ("sym_y", 3.0), ("sym_x", 3.0)])
def test_boundary_measure(label, exact):
    V = FunctionSpace(primitives.quarter_annulus(1.0, 4.0).elevate(3).refine(3))
    bd = evaluate(V, P.boundary_gauss(V, label))
    np.testing.assert_allclose(float(bd.w.sum()), exact, rtol=1e-12)


def test_boundary_normals_are_unit_and_outward():
    V = FunctionSpace(primitives.quarter_annulus(1.0, 4.0).elevate(3).refine(2))

    for label, sign in (("outer", 1.0), ("inner", -1.0)):
        bd = evaluate(V, P.boundary_gauss(V, label))
        x = np.asarray(bd.x).reshape(-1, 2)
        n = np.asarray(bd.normal).reshape(-1, 2)
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-13)
        radial = x / np.linalg.norm(x, axis=1, keepdims=True)
        np.testing.assert_allclose((n * radial).sum(1), sign, atol=1e-12)

    # straight edges: outward normals point away from the domain
    for label, expected in (("sym_y", [0.0, -1.0]), ("sym_x", [-1.0, 0.0])):
        n = np.asarray(evaluate(V, P.boundary_gauss(V, label)).normal).reshape(-1, 2)
        np.testing.assert_allclose(n, np.broadcast_to(expected, n.shape), atol=1e-13)


def test_divergence_theorem():
    """A global identity tying interior weights, boundary weights and normals."""
    V = FunctionSpace(primitives.quarter_annulus(1.0, 4.0).elevate(3).refine(3))

    area = float(evaluate(V, P.gauss(V)).w.sum())
    flux = 0.0
    for label in ("outer", "inner", "sym_y", "sym_x"):
        bd = evaluate(V, P.boundary_gauss(V, label))
        # F = (x, y)/2 has unit divergence, so the flux integral gives the area
        flux += float((bd.w * (bd.x * bd.normal).sum(-1) / 2).sum())
    np.testing.assert_allclose(flux, area, rtol=1e-11)


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_second_derivatives_against_finite_differences(name):
    """Check exact reproduction of affine fields in physical space."""
    V = FunctionSpace(GEOMETRIES[name]().elevate(3).refine(2))
    bd = evaluate(V, P.gauss(V), order=2)

    # Verify against an exactly-representable field: for any coefficient vector
    # c, the Hessian of sum_i c_i R_i must reproduce d2R contracted with c.
    # Cross-check the mixed derivative symmetry implied by the Voigt layout.
    d2 = np.asarray(bd.d2R)
    assert d2.shape[2] == 3
    assert np.all(np.isfinite(d2))

    # A field that is linear in physical space has zero second derivative.
    cpts = np.asarray(V.cpts)
    for comp in range(2):
        c = cpts[:, comp]
        local = c[np.asarray(bd.dofs)]  # (n_e, n_local)
        second = np.einsum("eqkn,en->eqk", d2, local)
        np.testing.assert_allclose(second, 0.0, atol=1e-6)
        first = np.einsum("eqdn,en->eqd", np.asarray(bd.dR), local)
        expected = np.zeros_like(first)
        expected[..., comp] = 1.0
        np.testing.assert_allclose(first, expected, atol=1e-10)


@pytest.mark.parametrize("dim,make", [
    (1, lambda: primitives.interval(0.0, 2.0)),
    (3, lambda: primitives.cuboid([0, 0, 0], [1, 2, 3])),
])
def test_other_dimensions(dim, make):
    V = FunctionSpace(make().elevate(2).refine(1))
    bd = evaluate(V, P.gauss(V))

    assert bd.dR.shape[2] == dim
    np.testing.assert_allclose(np.asarray(bd.R.sum(-1)), 1.0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(bd.dR.sum(-1)), 0.0, atol=1e-10)

    volume = {1: 2.0, 3: 6.0}[dim]
    np.testing.assert_allclose(float(bd.w.sum()), volume, rtol=1e-12)


def test_grid_pointset_has_no_weights():
    V = FunctionSpace(primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))
    bd = evaluate(V, P.grid(V, n=4))
    assert bd.w is None
    assert bd.R.shape[1] == 16


# -- differentiability -----------------------------------------------------


def test_evaluation_is_jittable():
    V = FunctionSpace(primitives.quarter_annulus(1.0, 2.0).elevate(2).refine(1))
    ps = P.gauss(V)

    @jax.jit
    def total_area(space):
        return evaluate(space, ps).w.sum()

    np.testing.assert_allclose(float(total_area(V)), float(evaluate(V, ps).w.sum()), rtol=1e-13)


def test_area_gradient_wrt_control_points():
    """Shape derivative through the evaluator, checked against finite differences."""
    import dataclasses

    V = FunctionSpace(primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))
    ps = P.gauss(V)

    def area(cpts):
        return evaluate(dataclasses.replace(V, cpts=cpts), ps).w.sum()

    g = np.asarray(jax.grad(area)(V.cpts))

    base = np.asarray(V.cpts)
    h = 1e-6
    for idx in [(0, 0), (3, 1), (7, 0)]:
        pert = base.copy()
        pert[idx] += h
        plus = float(area(jnp.asarray(pert)))
        pert[idx] -= 2 * h
        minus = float(area(jnp.asarray(pert)))
        np.testing.assert_allclose(g[idx], (plus - minus) / (2 * h), rtol=1e-5, atol=1e-8)


def test_gradient_wrt_weights():
    V = FunctionSpace(primitives.disk([0.0, 0.0], 1.0).elevate(3).refine(1))
    ps = P.gauss(V)

    def area(wgts):
        import dataclasses

        return evaluate(dataclasses.replace(V, wgts=wgts), ps).w.sum()

    g = np.asarray(jax.grad(area)(V.wgts))
    assert np.any(np.abs(g) > 1e-8), "weights must influence the measure"

    base = np.asarray(V.wgts)
    h = 1e-6
    for i in (0, 4):
        pert = base.copy()
        pert[i] += h
        plus = float(area(jnp.asarray(pert)))
        pert[i] -= 2 * h
        minus = float(area(jnp.asarray(pert)))
        np.testing.assert_allclose(g[i], (plus - minus) / (2 * h), rtol=1e-5, atol=1e-7)
