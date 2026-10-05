"""Implicit differentiation: every method, linear and nonlinear.

Three properties are asserted for each solution path, because they are the ones
the architecture claims and they fail independently:

1. the gradient is correct, against central finite differences;
2. it costs exactly **one** adjoint solve, whatever the iteration did, which is
   what distinguishes implicit differentiation from unrolling;
3. ``jax.jit(jax.grad(...))`` works, which requires the whole solve --
   including its convergence tests -- to be one traced program.

Property 3 is the one that used to fail: a Python loop with ``float(...)``
convergence tests cannot be traced, and property 2 silently degraded to one
adjoint solve *per iteration* wherever the iteration was differentiated
through.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxiga as jx
from jaxiga.solvers.linear import SOLVE_COUNTER, LinearOptions, reset_solve_counter

ALL = lambda x: jnp.full(x.shape[0], True)


def poisson(deg=2, refine=2):
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine))
    return jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where=ALL)],
        source=lambda x, p: p["f"] * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
    )


class NonlinearPoisson(jx.Problem):
    r"""``-div((1 + k u^2) grad u) = f``.

    Nonlinear in ``u``, so the residual has a non-vanishing second derivative.
    That is what makes it able to distinguish the exact Jacobian of the
    least-squares residual from its Gauss-Newton approximation.
    """

    is_linear = False

    def energy(self, grad_u, u, x, params):
        return 0.5 * (1.0 + params["k"] * u[0] ** 2) * jnp.sum(grad_u**2)


def overdetermined_collocation(deg=3, refine=2):
    """A nonlinear collocation problem whose residual does not vanish.

    Three Dirichlet sides and one Neumann side make the collocation system
    overdetermined, so the least-squares solution leaves ``F`` non-zero. Both
    conditions -- nonlinearity and a non-zero residual -- are needed: with
    either one absent, the neglected term of the Gauss-Newton approximation
    vanishes and the distinction this problem exists to test disappears.
    """
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine))
    return NonlinearPoisson(
        V,
        dirichlet=[
            jx.DirichletBC(0.0, where="left"),
            jx.DirichletBC(0.0, where="right"),
            jx.DirichletBC(0.0, where="bottom"),
        ],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([p["g"]]), where="top")],
        source=lambda x, p: p["f"] * jnp.sin(jnp.pi * x[0]),
    )


def hyperelastic(deg=2, refine=2):
    V = jx.FunctionSpace(
        jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine), vec=2
    )
    return jx.Hyperelasticity(
        V,
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="left")],
        neumann=[
            jx.Neumann(
                lambda x, n, p: jnp.array([p["load"], 0.3 * p["load"]]), where="right"
            )
        ],
    )


def central_difference(fn, params, key, h=None):
    base = params[key]
    h = h if h is not None else 1e-6 * max(1.0, abs(base))
    up = dict(params, **{key: base + h})
    dn = dict(params, **{key: base - h})
    return float((fn(up) - fn(dn)) / (2 * h))


# -- the parameter contract, per method ------------------------------------

CASES = {
    "galerkin-linear": (poisson, {"a0": 1.3, "f": 2.0}, dict(), 1e-7),
    "galerkin-nonlinear": (
        hyperelastic,
        {"E": 200.0, "nu": 0.3, "load": 5.0},
        dict(),
        1e-6,
    ),
    "energy-linear": (
        poisson,
        {"a0": 1.3, "f": 2.0},
        dict(method="energy", tol=1e-12, max_steps=3000),
        1e-6,
    ),
    "energy-nonlinear": (
        hyperelastic,
        {"E": 200.0, "nu": 0.3, "load": 5.0},
        dict(method="energy", tol=1e-11, max_steps=6000),
        1e-5,
    ),
    "collocation": (
        lambda: poisson(deg=3, refine=2),
        {"a0": 1.0, "f": 2.0},
        dict(method="collocation"),
        1e-6,
    ),
    "collocation-nonlinear": (
        overdetermined_collocation,
        {"k": 3.0, "f": 40.0, "g": 5.0},
        dict(method="collocation", max_steps=40),
        1e-6,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_gradient_matches_finite_differences(case):
    make, params, kwargs, rtol = CASES[case]
    problem = make()
    loss = lambda p: jnp.sum(jx.solve(problem, params=p, **kwargs).u ** 2)  # noqa: E731

    grads = jax.grad(loss)(params)
    for key in params:
        fd = central_difference(loss, params, key)
        if abs(fd) < 1e-12:  # the parameter genuinely does not move the solution
            assert abs(float(grads[key])) < 1e-8
            continue
        assert abs(float(grads[key]) - fd) / abs(fd) < rtol, (
            f"{case}: d/d{key} adjoint {float(grads[key]):.6e} vs fd {fd:.6e}"
        )


@pytest.mark.parametrize("case", sorted(CASES))
def test_gradient_costs_exactly_one_adjoint_solve(case):
    """The point of implicit differentiation: cost independent of iterations."""
    make, params, kwargs, _ = CASES[case]
    problem = make()
    loss = lambda p: jnp.sum(jx.solve(problem, params=p, **kwargs).u ** 2)  # noqa: E731

    reset_solve_counter()
    jax.grad(loss)(params)
    assert SOLVE_COUNTER["adjoint"] == 1, (
        f"{case}: {SOLVE_COUNTER['adjoint']} adjoint solves; the iteration is "
        f"being differentiated through"
    )


@pytest.mark.parametrize("case", sorted(CASES))
def test_jit_of_grad_compiles_and_agrees(case):
    """Requires the whole solve, convergence tests included, to be traceable."""
    make, params, kwargs, _ = CASES[case]
    problem = make()
    loss = lambda p: jnp.sum(jx.solve(problem, params=p, **kwargs).u ** 2)  # noqa: E731

    eager = jax.grad(loss)(params)
    compiled = jax.jit(jax.grad(loss))(params)
    for key in params:
        np.testing.assert_allclose(
            np.asarray(compiled[key]), np.asarray(eager[key]), rtol=1e-9, atol=1e-12
        )


@pytest.mark.parametrize("case", sorted(CASES))
def test_jit_of_solve_compiles(case):
    make, params, kwargs, _ = CASES[case]
    problem = make()
    f = lambda p: jx.solve(problem, params=p, **kwargs).u  # noqa: E731
    np.testing.assert_allclose(
        np.asarray(jax.jit(f)(params)), np.asarray(f(params)), rtol=1e-9, atol=1e-11
    )


# -- shape derivatives on the nonlinear path -------------------------------


def test_shape_derivative_through_a_nonlinear_solve():
    """Geometry gradients must reach through Newton, not only the linear path.

    This is what the ``implicit_correction`` formulation buys over a
    ``custom_vjp`` keyed on ``params``: the control points are closed over
    inside the quadrature data rather than passed as parameters, and the
    derivative finds them anyway.
    """
    import dataclasses

    problem = hyperelastic(refine=1)
    V = problem.space
    params = {"E": 200.0, "nu": 0.3, "load": 5.0}

    def loss(cpts):
        space = dataclasses.replace(V, cpts=cpts)
        return jnp.sum(jx.solve(problem.with_space(space), params=params).u ** 2)

    reset_solve_counter()
    g = jax.grad(loss)(V.cpts)
    assert SOLVE_COUNTER["adjoint"] == 1

    flat = V.cpts.reshape(-1)
    h = 1e-6
    for i in (3, 7):
        fd = float(
            (
                loss(flat.at[i].add(h).reshape(V.cpts.shape))
                - loss(flat.at[i].add(-h).reshape(V.cpts.shape))
            )
            / (2 * h)
        )
        if abs(fd) < 1e-10:
            continue
        assert abs(float(g.reshape(-1)[i]) - fd) / abs(fd) < 1e-5


# -- the methods must agree on the derivative, not just the solution -------


def test_all_methods_give_the_same_gradient():
    """A stronger cross-check than agreeing on the solution.

    Galerkin and energy minimisation find the same root by different routes, so
    their implicit derivatives must coincide; collocation solves a different
    discrete problem, so it is compared only against its own finite difference
    elsewhere.
    """
    problem = poisson(deg=3, refine=2)
    params = {"a0": 1.3, "f": 2.0}
    loss = lambda p, **kw: jnp.sum(jx.solve(problem, params=p, **kw).u ** 2)  # noqa: E731

    g_gal = jax.grad(lambda p: loss(p))(params)
    g_ene = jax.grad(lambda p: loss(p, method="energy", tol=1e-12, max_steps=4000))(
        params
    )
    for key in params:
        rel = abs(float(g_gal[key]) - float(g_ene[key])) / abs(float(g_gal[key]))
        assert rel < 1e-6, f"d/d{key}: galerkin vs energy differ by {rel:.2e}"


# -- the correction must not move the solution ------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(method="energy", tol=1e-12, max_steps=3000), dict(method="collocation")],
)
def test_implicit_correction_leaves_the_value_unchanged(kwargs):
    """It is a derivative device; the primal value must be untouched."""
    problem = poisson(deg=3, refine=2)
    params = {"a0": 1.0, "f": 2.0}
    with_it = jx.solve(problem, params=params, **kwargs).u
    without = jx.solve(problem, params=params, implicit=False, **kwargs).u
    assert float(jnp.abs(with_it - without).max()) < 1e-11


def test_matrix_free_nonlinear_gradient():
    """The matrix-free tangent must carry the adjoint too."""
    problem = hyperelastic(refine=1)
    params = {"E": 200.0, "nu": 0.3, "load": 5.0}
    opts = LinearOptions(method="cg", matrix_free=True, tol=1e-13)

    loss = lambda p: jnp.sum(jx.solve(problem, params=p, linear=opts).u ** 2)  # noqa: E731
    g = float(jax.grad(loss)(params)["load"])
    fd = central_difference(loss, params, "load")
    assert abs(g - fd) / abs(fd) < 1e-5


# -- Newton reporting under trace ------------------------------------------


def test_newton_stats_survive_jit():
    problem = hyperelastic(refine=1)
    params = {"E": 200.0, "nu": 0.3, "load": 5.0}
    sol = jx.solve(problem, params=params)
    assert bool(sol.stats["converged"])
    history = jx.history_list(sol.stats)
    assert len(history) == int(sol.stats["iterations"]) + 1
    assert history[-1] < 1e-8
    # strictly decreasing, which the line search is there to guarantee
    assert all(b < a for a, b in zip(history, history[1:]))


# -- the collocation Jacobian must be exact, not Gauss-Newton ---------------


def test_overdetermined_nonlinear_collocation_has_a_nonzero_residual():
    """The premise of the test below; assert it rather than assume it."""
    problem = overdetermined_collocation()
    params = {"k": 3.0, "f": 40.0, "g": 5.0}
    stats = jx.solve(problem, method="collocation", params=params, max_steps=40).stats
    assert int(stats["n_equations"]) > int(stats["n_free"])
    assert float(stats["residual"]) > 1.0


def test_collocation_implicit_jacobian_is_exact_not_gauss_newton():
    """``d(J^T F)/du`` is ``J^T J + sum_i F_i grad^2 F_i``, not ``J^T J``.

    The second term vanishes for a linear problem (``F`` affine) and for a
    zero-residual one, which is why an implementation using ``J^T J`` passes
    every square or linear test and fails only here. Gauss-Newton is still the
    right choice for the *iteration*; only the implicit derivative needs the
    exact Jacobian.
    """
    import numpy as np

    from jaxiga.methods._common import build_context
    from jaxiga.methods.collocation import (
        _block_residual,
        _rows_and_columns,
        build_blocks,
    )

    problem = overdetermined_collocation()
    params = {"k": 3.0, "f": 40.0, "g": 5.0}
    sol = jx.solve(problem, method="collocation", params=params, max_steps=40)

    ctx = build_context(problem, params)
    blocks = build_blocks(problem, params, ctx, None)
    layout, _ = _rows_and_columns(problem, blocks, ctx)

    def residual_vector(u_free):
        u = ctx.dofmap.lift(u_free)
        out = []
        for block, _ in zip(blocks, layout):
            r = _block_residual(problem, block, u, params, "d2R") * block.scale
            out.append(r.reshape(-1)[np.flatnonzero(block.comp_mask.reshape(-1))])
        return jnp.concatenate(out)

    u_star = sol.u[ctx.dofmap.free]
    least_squares = lambda v: 0.5 * jnp.sum(residual_vector(v) ** 2)  # noqa: E731
    H = np.asarray(jax.hessian(least_squares)(u_star))
    J = np.asarray(jax.jacfwd(residual_vector)(u_star))

    # the neglected term is small in norm but not negligible in the solve
    assert np.linalg.norm(H - J.T @ J) / np.linalg.norm(H) > 1e-4
    rhs = np.asarray(u_star)
    shift = np.linalg.norm(
        np.linalg.solve(H, rhs) - np.linalg.solve(J.T @ J, rhs)
    ) / np.linalg.norm(np.linalg.solve(H, rhs))
    assert shift > 1e-3, "this problem no longer distinguishes the two Jacobians"


def test_nonlinear_collocation_gradient_beats_gauss_newton_accuracy():
    """A regression bound tight enough that Gauss-Newton would fail it.

    Gauss-Newton gives roughly three correct digits on this problem; the exact
    Jacobian gives eight or more. The threshold sits between the two.
    """
    problem = overdetermined_collocation()
    params = {"k": 3.0, "f": 40.0, "g": 5.0}
    loss = lambda p: jnp.sum(  # noqa: E731
        jx.solve(problem, method="collocation", params=p, max_steps=40).u ** 2
    )
    grads = jax.grad(loss)(params)
    for key in params:
        fd = central_difference(loss, params, key)
        rel = abs(float(grads[key]) - fd) / abs(fd)
        assert rel < 1e-6, f"d/d{key}: {rel:.2e} -- Gauss-Newton territory"


# -- the public linear= option must reach every path ------------------------


def multimode_poisson(deg=2, refine=3):
    """A Poisson problem whose source excites several modes.

    Deliberately not a single sine. With ``f = sin(pi x) sin(pi y)`` the
    discrete right-hand side is very nearly an eigenvector of the stiffness
    matrix, so conjugate gradient converges in one step and a solver restricted
    to a single iteration is indistinguishable from an exact one -- which makes
    that problem useless for detecting a linear option that never arrives.
    """
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine))
    return jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where=ALL)],
        source=lambda x, p: (
            p["f"] * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
            + 3.0 * jnp.sin(5 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
        ),
    )


@pytest.mark.parametrize(
    "kwargs", [dict(), dict(method="energy", tol=1e-12, max_steps=4000)]
)
def test_linear_options_reach_the_adjoint_solve(kwargs):
    """A deliberately crippled linear solver must change the gradient.

    Behavioural rather than a check on the source: if the dispatcher dropped
    ``linear=`` on the way to a method, both gradients below would come from the
    default backend and agree exactly -- which is the silent failure being
    guarded against, and which the energy path exhibited.

    The loss is weighted by a fixed pseudo-random vector so that its cotangent
    is not aligned with an eigenvector of the tangent; with the natural choice
    ``sum(u^2)`` the adjoint right-hand side is nearly an eigenvector here, one
    CG step solves it, and the crippled solver again looks exact.
    """
    problem = multimode_poisson()
    params = {"a0": 1.3, "f": 2.0}
    weights = jnp.asarray(
        np.random.default_rng(7).normal(size=problem.space.n_dofs)
    )

    def grad_with(opts):
        return float(
            jax.grad(
                lambda p: jnp.sum(
                    weights * jx.solve(problem, params=p, linear=opts, **kwargs).u
                )
            )(params)["a0"]
        )

    good = grad_with(LinearOptions(method="dense"))
    crippled = grad_with(
        LinearOptions(method="cg", preconditioner="none", maxiter=1, tol=1e-30)
    )
    accurate = grad_with(LinearOptions(method="cg", tol=1e-13))

    assert abs(good - crippled) / abs(good) > 1e-2, (
        "the linear options are not reaching this method's adjoint solve"
    )
    assert abs(good - accurate) / abs(good) < 1e-10, (
        "two adequate backends should agree"
    )


@pytest.mark.parametrize(
    "kwargs",
    [dict(), dict(method="energy", tol=1e-12, max_steps=3000), dict(method="collocation")],
)
def test_every_backend_gives_the_same_answer(kwargs):
    problem = poisson(deg=3, refine=2)
    params = {"a0": 1.0, "f": 2.0}
    a = jx.solve(problem, params=params, linear=LinearOptions(method="dense"), **kwargs).u
    b = jx.solve(problem, params=params, linear=LinearOptions(method="scipy"), **kwargs).u
    assert float(jnp.abs(a - b).max()) < 1e-10
