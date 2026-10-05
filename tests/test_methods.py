"""Energy minimization and collocation, and their consistency with Galerkin.

The three methods share one physics definition, so agreement between them is a
genuine cross-validation rather than a tautology: Galerkin solves the discrete
residual, energy minimization minimizes the potential whose gradient that
residual is, and collocation enforces the strong form the same energy implies.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxiga as jx
from jaxiga.solvers.newton import history_list
from jaxiga.solvers.optimize import minimize_adam, minimize_lbfgs

ALL = lambda x: jnp.full(x.shape[0], True)
EXACT = lambda x: jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
SOURCE = lambda x, p: 8 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])


def poisson(deg=3, refine=2, patches=None):
    if patches is None:
        patches = jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine)
    V = jx.FunctionSpace(patches)
    return jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=SOURCE)


def annulus_elasticity(deg=3, refine=1, pressure=10.0):
    V = jx.FunctionSpace(jx.primitives.quarter_annulus(1.0, 4.0).elevate(deg).refine(refine), vec=2)
    return jx.LinearElasticity(
        V,
        plane="stress",
        dirichlet=[jx.DirichletBC(0.0, where="sym_y", component=1),
                   jx.DirichletBC(0.0, where="sym_x", component=0)],
        neumann=[jx.Neumann(lambda x, n, p: -pressure * n, where="inner")],
    )


# -- optimizers ------------------------------------------------------------


def test_lbfgs_minimizes_a_quadratic():
    A = jnp.array([[3.0, 1.0], [1.0, 2.0]])
    b = jnp.array([1.0, -2.0])
    loss = lambda x: 0.5 * x @ A @ x - b @ x
    res = minimize_lbfgs(loss, jnp.zeros(2), tol=1e-12)
    assert res.converged
    np.testing.assert_allclose(np.asarray(res.x), np.asarray(jnp.linalg.solve(A, b)), atol=1e-9)


def test_adam_makes_progress():
    loss = lambda x: jnp.sum((x - 3.0) ** 2)
    res = minimize_adam(loss, jnp.zeros(3), learning_rate=0.1, max_steps=2000)
    np.testing.assert_allclose(np.asarray(res.x), 3.0, atol=1e-4)


def test_unknown_optimizer_raises():
    from jaxiga.solvers.optimize import minimize

    with pytest.raises(ValueError, match="unknown optimizer"):
        minimize(lambda x: jnp.sum(x**2), jnp.zeros(2), optimizer="nope")


# -- energy method ---------------------------------------------------------


def test_energy_matches_galerkin_scalar():
    problem = poisson(deg=3, refine=2)
    g = jx.solve(problem, params={"a0": 1.0})
    e = jx.solve(problem, params={"a0": 1.0}, method="energy", max_steps=4000)

    assert e.stats["converged"], e.stats
    diff = float(jnp.max(jnp.abs(e.u - g.u)))
    scale = float(jnp.max(jnp.abs(g.u)))
    assert diff <= 1e-8 * scale, f"{diff:.3e} > 1e-8 * {scale:.3e}"


def test_energy_matches_galerkin_vector():
    """Section 10.3 of the architecture guide, on the quarter annulus."""
    problem = annulus_elasticity(deg=3, refine=1)
    params = {"E": 1e5, "nu": 0.3}
    g = jx.solve(problem, params=params)
    e = jx.solve(problem, params=params, method="energy", max_steps=4000)

    diff = float(jnp.max(jnp.abs(e.u - g.u)))
    scale = float(jnp.max(jnp.abs(g.u)))
    assert diff <= 1e-8 * scale, f"{diff:.3e} > 1e-8 * {scale:.3e}"


def test_galerkin_residual_vanishes_at_the_energy_minimizer():
    """The Galerkin residual *is* the gradient of the energy loss."""
    from jaxiga.methods._common import build_context, residual

    problem = annulus_elasticity(deg=2, refine=1)
    params = {"E": 1e5, "nu": 0.3}
    e = jx.solve(problem, params=params, method="energy", max_steps=4000)

    ctx = build_context(problem, params)
    r = residual(ctx.dofmap.restrict(e.u), params, ctx)
    assert float(jnp.linalg.norm(r)) < 1e-8


def test_energy_solves_a_nonlinear_problem():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1), vec=2)
    problem = jx.Hyperelasticity(
        V,
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                   jx.DirichletBC([0.0, 0.2], where="top")],
    )
    params = {"mu": 384.6, "lam": 576.9}

    newton = jx.solve(problem, params=params)
    em = jx.solve(problem, params=params, method="energy", max_steps=5000)

    assert bool(newton.stats["converged"]) and int(newton.stats["iterations"]) <= 10
    # quadratic convergence: the residual should fall off a cliff
    history = history_list(newton.stats)
    assert history[-1] < 1e-9
    # ... and the last step should have squared the previous residual, roughly
    assert history[-1] < history[-2] ** 1.5

    diff = float(jnp.max(jnp.abs(em.u - newton.u)))
    assert diff <= 1e-8 * float(jnp.max(jnp.abs(newton.u)))


def test_energy_rejects_a_problem_without_energy():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))

    class NoEnergy(jx.Problem):
        pass

    problem = NoEnergy(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)])
    with pytest.raises(TypeError, match="no energy density"):
        jx.solve(problem, params={}, method="energy")


# -- collocation -----------------------------------------------------------


def test_greville_points_are_correct():
    from jaxiga.space.pointset import greville, greville_1d

    np.testing.assert_allclose(greville_1d([0, 0, 0, 0.5, 1, 1, 1], 2), [0, 0.25, 0.75, 1.0])
    np.testing.assert_allclose(greville_1d([0, 0, 1, 1], 1), [0.0, 1.0])

    # one point per basis function, and for affine geometry they coincide
    # exactly with the control points
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 2, 1).elevate(3).refine(2))
    ps = greville(V)
    assert len(ps.elems) == V.n_scalar_basis

    from jaxiga.space.evaluation import evaluate

    x = np.asarray(evaluate(V, ps).x).reshape(-1, 2)
    np.testing.assert_allclose(x, np.asarray(V.cpts)[ps.point_dofs], atol=1e-12)


def test_collocation_solves_poisson():
    problem = poisson(deg=3, refine=3)
    c = jx.solve(problem, params={"a0": 1.0}, method="collocation")
    assert c.stats["residual"] < 1e-9
    assert jx.errornorm(c, EXACT, "L2") < 0.06


@pytest.mark.parametrize("deg,expected", [(3, 2), (4, 4)])
def test_collocation_convergence_rate(deg, expected):
    """Greville collocation converges at p-1 for odd p and p for even p."""
    errs = [
        jx.errornorm(
            jx.solve(poisson(deg, r), params={"a0": 1.0}, method="collocation"), EXACT, "L2"
        )
        for r in (2, 3, 4)
    ]
    assert errs[0] > errs[1] > errs[2], errs
    rate = np.log2(errs[-2] / errs[-1])
    assert rate > expected - 0.7, f"rate {rate:.2f} well below the expected {expected}"


def test_both_strong_residual_variants_agree():
    """The d2R expansion and nested jacfwd must give the same answer.

    Run on a curved NURBS patch so the full chain rule, including the
    second-order push-forward, is exercised.
    """
    V = jx.FunctionSpace(jx.primitives.quarter_annulus(1.0, 2.0).elevate(3).refine(2))
    problem = jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=SOURCE)

    a = jx.solve(problem, params={"a0": 1.0}, method="collocation", variant="d2R")
    b = jx.solve(problem, params={"a0": 1.0}, method="collocation", variant="autodiff")

    diff = float(jnp.max(jnp.abs(a.u - b.u)))
    assert diff <= 1e-10 * float(jnp.max(jnp.abs(a.u))), diff


def test_collocation_is_square_on_a_single_patch():
    problem = poisson(deg=3, refine=2)
    c = jx.solve(problem, params={"a0": 1.0}, method="collocation", system="square")
    assert c.stats["n_equations"] == c.stats["n_free"]


def test_collocation_multipatch_adds_interface_equations():
    two_exact = lambda x: jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
    src = lambda x, p: 2 * jnp.pi**2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])

    errs = []
    for r in (1, 2, 3):
        V = jx.FunctionSpace([
            jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(r),
            jx.primitives.rectangle(1, 0, 2, 1).elevate(3).refine(r),
        ])
        problem = jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=src)
        c = jx.solve(problem, params={"a0": 1.0}, method="collocation")

        # flux continuity makes the system overdetermined
        assert c.stats["n_equations"] > c.stats["n_free"]
        errs.append(jx.errornorm(c, two_exact, "L2"))

    assert errs[0] > errs[1] > errs[2], errs


def test_square_system_rejected_when_overdetermined():
    V = jx.FunctionSpace([
        jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(1),
        jx.primitives.rectangle(1, 0, 2, 1).elevate(3).refine(1),
    ])
    problem = jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=SOURCE)
    with pytest.raises(ValueError, match="one equation per free dof"):
        jx.solve(problem, params={"a0": 1.0}, method="collocation", system="square")


def test_collocation_handles_neumann():
    """u = x is exact; a unit flux on the right edge must be reproduced."""
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(2))
    problem = jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where="left")],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([1.0]), where="right")],
        source=lambda x, p: 0.0,
    )
    c = jx.solve(problem, params={"a0": 1.0}, method="collocation")
    assert jx.errornorm(c, lambda x: x[0], "L2") < 1e-8


# -- jit / differentiability -----------------------------------------------


def test_solve_is_jittable_end_to_end():
    """Requires connectivity to stay concrete inside a trace."""
    problem = poisson(deg=2, refine=1)
    loss = lambda a0: jnp.sum(jx.solve(problem, params={"a0": a0}).u ** 2)

    plain = float(loss(1.0))
    np.testing.assert_allclose(float(jax.jit(loss)(1.0)), plain, rtol=1e-12)
    np.testing.assert_allclose(
        float(jax.jit(jax.grad(loss))(1.0)), float(jax.grad(loss)(1.0)), rtol=1e-10
    )


def test_static_array_is_hashable_and_content_keyed():
    from jaxiga.space.function_space import StaticArray

    a = StaticArray(np.arange(6).reshape(2, 3))
    b = StaticArray(np.arange(6).reshape(2, 3))
    c = StaticArray(np.arange(6, 12).reshape(2, 3))

    assert hash(a) == hash(b) and a == b
    assert a != c
    np.testing.assert_array_equal(np.asarray(a), np.arange(6).reshape(2, 3))
    with pytest.raises(ValueError):
        a.array[0, 0] = 99  # must be immutable, or the hash would lie


def test_energy_method_gradient_wrt_parameters():
    problem = annulus_elasticity(deg=2, refine=1)

    def loss(E):
        sol = jx.solve(problem, params={"E": E, "nu": 0.3}, method="energy", max_steps=2000)
        return jnp.sum(sol.u**2)

    # displacement scales as 1/E, so the loss scales as 1/E^2
    lo, hi = float(loss(1e5)), float(loss(2e5))
    np.testing.assert_allclose(lo / hi, 4.0, rtol=1e-4)
