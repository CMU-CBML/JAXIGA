"""Failure paths and transformation contracts that successful solves miss."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxiga as jx
from jaxiga.solvers.linear import LinearOptions
from jaxiga.solvers.newton import newton
from jaxiga.solvers.optimize import minimize, minimize_adam, minimize_lbfgs


@pytest.mark.parametrize("tol", [0, -1e-6, np.inf, np.nan])
def test_linear_options_reject_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        LinearOptions(tol=tol)


@pytest.mark.parametrize("maxiter", [0, -1, 1.5, True])
def test_linear_options_reject_invalid_iteration_limit(maxiter):
    with pytest.raises(ValueError, match="maxiter"):
        LinearOptions(maxiter=maxiter)


def test_newton_backtracks_out_of_nonfinite_domain():
    residual = lambda u, p: jnp.log(u) + 5
    u, stats = newton(
        residual, jnp.ones(1), None,
        tangent_fn=lambda u, p: (np.array([[0, 0]]), 1 / u),
        lin_opts=LinearOptions(method="dense"),
    )
    assert stats["converged"]
    np.testing.assert_allclose(u, [np.exp(-5)], rtol=1e-10)
    assert float(stats["residual"]) == float(jnp.linalg.norm(residual(u, None)))


def test_newton_failed_root_is_not_corrected_and_stats_survive_jit():
    def run():
        return newton(
            lambda u, p: u * u - 4, jnp.ones(1), None,
            tangent_fn=lambda u, p: (np.array([[0, 0]]), 2 * u),
            lin_opts=LinearOptions(method="dense"), max_steps=0,
        )
    with pytest.warns(RuntimeWarning, match="0-step limit"):
        u, stats = run()
    np.testing.assert_array_equal(u, [1.])
    assert not stats["converged"]
    assert stats["iteration_limit"]
    uj, sj = jax.jit(run)()
    np.testing.assert_array_equal(uj, u)
    assert not sj["converged"]


def test_newton_reports_nonfinite_initial_state():
    with pytest.warns(RuntimeWarning, match="non-finite state"):
        u, stats = newton(
            lambda u, p: jnp.log(u), -jnp.ones(1), None,
            tangent_fn=lambda u, p: (np.array([[0, 0]]), 1 / u),
            lin_opts=LinearOptions(method="dense"),
        )
    assert not stats["finite"]
    assert int(stats["iterations"]) == 0


def test_adam_lbfgs_concatenates_histories_of_different_lengths():
    loss = lambda x: jnp.sum((x - 1) ** 2)
    x = jnp.zeros(2)
    warm = minimize_adam(loss, x, max_steps=3)
    finish = minimize_lbfgs(loss, warm.x, max_steps=7)
    combined = minimize(loss, x, optimizer="adam+lbfgs", warmup_steps=3, max_steps=7)
    assert combined.converged
    np.testing.assert_allclose(combined.x, 1., atol=1e-10)
    np.testing.assert_allclose(combined.history, jnp.concatenate([warm.history, finish.history]),
                               equal_nan=True)
    assert combined.steps == warm.steps + finish.steps


def small_problem():
    V = jx.FunctionSpace(jx.primitives.interval(0., 1.).elevate(3).refine(2))
    return jx.Poisson(V, dirichlet=[jx.DirichletBC(0., where="left"),
                                   jx.DirichletBC(0., where="right")],
                      source=lambda x, p: 1 + 10 * x[0] ** 2)


def test_compiled_truncated_linear_solve_reports_failure():
    p = small_problem()
    def run():
        s = jx.solve(p, linear=LinearOptions(method="cg", preconditioner="none", maxiter=1))
        return s.u, {k: s.stats[k] for k in ("converged", "relative_residual")}
    u, stats = jax.jit(run)()
    assert jnp.all(jnp.isfinite(u))
    assert not stats["converged"]
    assert stats["relative_residual"] > 1e-4


def test_matrix_free_solve_does_not_construct_csr(monkeypatch):
    from jaxiga.methods import _common
    p = small_problem()
    def forbidden(*args):
        raise AssertionError("matrix-free solve constructed a CSR pattern")
    monkeypatch.setattr(_common, "_build_csr_pattern", forbidden)
    s = jx.solve(p, linear=LinearOptions(method="cg", matrix_free=True))
    assert s.stats["converged"]
    assert s.stats["nnz"] is None


def test_cpu_cg_native_kernel_preserves_jitted_material_and_shape_gradients(monkeypatch):
    if jax.default_backend() != "cpu":
        pytest.skip("native CSR CG is the CPU path")
    from dataclasses import replace
    import scipy.sparse.linalg
    calls = []
    native = scipy.sparse.linalg.cg

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return native(*args, **kwargs)

    monkeypatch.setattr(scipy.sparse.linalg, "cg", counted)
    space = jx.FunctionSpace(jx.primitives.interval(0., 1.).elevate(3).refine(2))
    base = jx.Poisson(space, source=lambda x, p: 1.0,
                      dirichlet=[jx.DirichletBC(0., where="left"),
                                 jx.DirichletBC(0., where="right")])

    def loss(a0, length):
        V = replace(space, cpts=length * space.cpts)
        solution = jx.solve(base.with_space(V), params={"a0": a0},
                            linear=LinearOptions(method="cg", tol=1e-12, maxiter=200))
        return jnp.sum(solution.u**2)

    value, (material, shape) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(1.3, 1.2)
    np.testing.assert_allclose(material, -2 * value / 1.3, rtol=1e-9)
    np.testing.assert_allclose(shape, 4 * value / 1.2, rtol=1e-9)
    assert len(calls) == 2  # one primal and one adjoint solve
    assert all(call["M"] is not None for call in calls)


@pytest.mark.parametrize("budget", [2, 20])
def test_cpu_cg_corrects_recursive_residual_drift_with_shared_budget(monkeypatch, budget):
    if jax.default_backend() != "cpu":
        pytest.skip("native CSR CG is the CPU path")
    import scipy.sparse
    import scipy.sparse.linalg
    from jaxiga.solvers.linear import CSRPattern, sparse_solve

    A = scipy.sparse.csr_matrix([[4., 1.], [1., 3.]])
    pattern = CSRPattern(A.indptr, A.indices, np.repeat(np.arange(2), 2),
                         np.array([], dtype=int), A.shape, symmetric=True)
    rhs = np.array([1., 2.])
    native = scipy.sparse.linalg.cg
    calls = []

    def drifted(A, b, **kwargs):
        calls.append((b.copy(), kwargs["maxiter"]))
        u, info = native(A, b, **kwargs)
        # Model a recursive residual reaching tolerance before the true one.
        if len(calls) == 1:
            u *= 1 - 1e-4
        return u, info

    monkeypatch.setattr(scipy.sparse.linalg, "cg", drifted)
    opts = LinearOptions(method="cg", tol=1e-10, maxiter=budget)
    result = np.asarray(sparse_solve(opts, pattern, jnp.asarray(A.data), jnp.asarray(rhs)))
    error = np.linalg.norm(A @ result - rhs) / np.linalg.norm(rhs)
    assert calls[0][1] == budget
    if budget == 20:
        assert len(calls) == 2 and calls[1][1] == 18
        np.testing.assert_allclose(calls[1][0], rhs * 1e-4, atol=1e-15)
        assert error <= opts.tol
    else:
        assert len(calls) == 1  # no correction iterations remain
        assert error > opts.tol
    count = len(calls)
    zero = np.asarray(sparse_solve(opts, pattern, jnp.asarray(A.data), jnp.zeros(2)))
    np.testing.assert_array_equal(zero, np.zeros(2))
    assert len(calls) == count


def test_automatic_cg_requires_spd_declaration_matching_energy(monkeypatch):
    from jaxiga.methods.galerkin import default_linear
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    p = small_problem()
    assert default_linear(p, 4000).method == "cg"
    class Indefinite(jx.Poisson):
        def energy(self, g, u, x, params):
            return 0.5 * jnp.sum(g*g) - 100 * jnp.sum(u*u)
    assert default_linear(Indefinite(p.space), 4000).method == "scipy"
    assert default_linear(p, 100).method == "scipy"


def test_parameter_gradient_has_second_order_taylor_remainder():
    p = small_problem()
    loss = jax.jit(lambda a: jnp.sum(jx.solve(p, params={"a0": a}).u ** 2))
    a = 1.3
    value, grad = jax.value_and_grad(loss)(a)
    errors = np.array([abs(float(loss(a+h)-value-h*grad)) for h in (0.02, 0.01, 0.005)])
    np.testing.assert_allclose(errors[:-1]/errors[1:], 4., rtol=.04)


def test_nurbs_weight_gradient_through_solved_pde():
    import dataclasses

    V = jx.FunctionSpace(jx.primitives.quarter_annulus(1., 2.).elevate(3).refine(1))
    direction = jnp.asarray(np.random.default_rng(7).normal(size=V.wgts.shape)) * .1
    def loss(t):
        W = dataclasses.replace(V, wgts=V.wgts + t*direction)
        p = jx.Poisson(W, dirichlet=[jx.DirichletBC(0., where=side) for side in V.boundaries],
                       source=lambda x, p: 1. + x[0])
        return jnp.sum(jx.solve(p).u**2)
    loss = jax.jit(loss)
    derivative = jax.jit(jax.grad(loss))(0.)
    h = 1e-4
    difference = (loss(h)-loss(-h))/(2*h)
    assert abs(float(derivative)) > 1e-6
    np.testing.assert_allclose(derivative, difference, rtol=2e-6, atol=1e-10)


def test_nested_projection_reproduces_curved_rational_field_off_quadrature():
    from jaxiga.post.solution import Solution
    from jaxiga.space import pointset as P
    from jaxiga.space.evaluation import evaluate
    from jaxiga.space.transfer import coarse_basis_at, parent_map, project

    patch = jx.primitives.quarter_annulus(1., 2.).elevate(3).refine(2)
    coarse = jx.refine_elements(jx.FunctionSpace(patch), [])
    fine = jx.refine_elements(coarse, [0, 2, 5])
    assert float(jnp.ptp(coarse.wgts)) > .1
    coefficients = jnp.asarray(np.random.default_rng(8).normal(size=coarse.n_dofs))
    field = Solution(coarse, coefficients)
    mapping = parent_map(coarse, fine)
    quadrature = P.gauss(fine)
    values = field.at(basis=coarse_basis_at(coarse, mapping, quadrature))[..., 0]
    transferred = Solution(fine, project(values, fine, evaluate(fine, quadrature)))
    # A different quadrature order probes reproduction away from the fit points.
    check = P.gauss(fine, n=6)
    expected = field.at(basis=coarse_basis_at(coarse, mapping, check))
    np.testing.assert_allclose(transferred.at(basis=evaluate(fine, check)), expected,
                               rtol=1e-10, atol=1e-11)


def test_collocation_failure_exposes_stationarity_and_skips_correction():
    class Nonlinear(jx.Problem):
        def energy(self, g, u, x, params):
            return 0.5*jnp.sum(g*g) + 0.25*jnp.sum(u**4)
    p = small_problem()
    p = Nonlinear(p.space, dirichlet=p.dirichlet, source=p.source)
    with pytest.warns(RuntimeWarning, match="stationarity"):
        s = jx.solve(p, method="collocation", max_steps=0)
    assert not s.stats["converged"]
    assert s.stats["grad_norm"] > 0
    np.testing.assert_array_equal(s.u, jnp.zeros_like(s.u))


def test_fracture_sweep_limit_is_visible_and_can_raise():
    from jaxiga.solvers.phase_field import StaggeredSolver, Material, Fracture, Adaptivity
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))
    s = StaggeredSolver(
        V, Material(100., .3), Fracture(2.7, .2),
        dirichlet=[jx.DirichletBC([0., 0.], where="bottom"),
                   jx.DirichletBC([0., 1.], where="top")], reaction=("top", 1),
        adaptivity=Adaptivity(enabled=False), max_sweeps=1,
    )
    with pytest.warns(RuntimeWarning, match="did not converge"):
        step = next(s.run([.001], keep_solutions=False))
    assert not step.converged
    assert np.isinf(step.residual)
    with pytest.raises(RuntimeError, match="did not converge"):
        next(s.run([.002], on_failure="raise"))
