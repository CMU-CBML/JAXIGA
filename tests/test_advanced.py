"""Second-gradient energies, matrix-free and chunked assembly, batching, dynamics.

These are the v2.1 capabilities that sit outside the adaptivity work: the
fourth-order plate (which is where C^1 splines earn their keep), the two ways
of assembling that must agree with the eager one, batched solves, and time
integration.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxiga as jx
from jaxiga.methods._common import build_context
from jaxiga.methods.galerkin import (
    assemble_triplets,
    mass_triplets,
    tangent_diagonal,
    tangent_matvec,
)
from jaxiga.solvers import dynamics
from jaxiga.solvers.linear import (
    SOLVE_COUNTER,
    LinearOptions,
    reset_solve_counter,
)

ALL = lambda x: jnp.full(x.shape[0], True)
EXACT = lambda x: jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
SOURCE = lambda x, p: 2 * jnp.pi**2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def poisson(deg=2, refine=3):
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine))
    return jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=SOURCE)


def dense_from(indices, data, n):
    K = np.zeros((n, n))
    np.add.at(K, (indices[:, 0], indices[:, 1]), np.asarray(data))
    return K


# ==========================================================================
# Section 24 -- second-gradient energies and the Kirchhoff plate
# ==========================================================================

E_PLATE, NU_PLATE, T_PLATE, Q0 = 210e3, 0.3, 0.01, 1.0
D_PLATE = E_PLATE * T_PLATE**3 / (12 * (1 - NU_PLATE**2))
NAVIER = lambda x: Q0 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) / (
    4 * jnp.pi**4 * D_PLATE
)
PLATE_PARAMS = {"E": E_PLATE, "nu": NU_PLATE, "t": T_PLATE}


def simply_supported(deg, refine):
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine))
    return jx.KirchhoffPlate(
        V,
        dirichlet=[jx.DirichletBC(0.0, where=ALL)],
        source=lambda x, p: Q0 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
    )


@pytest.mark.parametrize("deg,expected", [(2, 2), (3, 4)])
def test_plate_matches_the_navier_solution_at_the_predicted_rate(deg, expected):
    """L2 rate ``min(p+1, 2p-2)`` for a simply supported plate under sine load."""
    errs, hs = [], []
    for n in (3, 4, 5):
        sol = jx.solve(simply_supported(deg, n), params=PLATE_PARAMS)
        errs.append(jx.errornorm(sol, NAVIER, "L2"))
        hs.append(1.0 / 2**n)
    rate = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert abs(rate - expected) / expected < 0.10, f"rate {rate}"
    assert errs[-1] < 1e-3


def test_clamped_plate_matches_timoshenko():
    """Uniform load, clamped edges: ``w_max = 0.0012653 q a^4 / D``."""
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(4))
    prob = jx.KirchhoffPlate(
        V,
        dirichlet=[jx.ClampedBC(where=lbl) for lbl in ("left", "right", "bottom", "top")],
        source=lambda x, p: Q0,
    )
    sol = jx.solve(prob, params=PLATE_PARAMS)
    w = float(sol.probe(np.array([[0.5, 0.5]]))[0, 0])
    assert abs(w / (0.0012653 * Q0 / D_PLATE) - 1.0) < 2e-3


def test_clamping_constrains_two_control_point_layers():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(2))
    simple = jx.KirchhoffPlate(V, dirichlet=[jx.DirichletBC(0.0, where="left")])
    clamped = jx.KirchhoffPlate(V, dirichlet=[jx.ClampedBC(where="left")])
    n_simple = len(simple.resolve_dirichlet()[0])
    n_clamped = len(clamped.resolve_dirichlet()[0])
    assert n_clamped == 2 * n_simple


def test_clamped_bc_rejects_a_non_zero_value():
    with pytest.raises(ValueError, match="homogeneous"):
        jx.ClampedBC(value=1.0, where="left")


def test_plate_rejects_degree_one():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).refine(2))
    with pytest.raises(ValueError, match="degree"):
        jx.KirchhoffPlate(V)


def test_needs_hessian_is_off_by_default():
    assert jx.Poisson(poisson().space).needs_hessian is False
    assert jx.KirchhoffPlate(simply_supported(2, 1).space).needs_hessian is True


# ==========================================================================
# Section 25 -- matrix-free and chunked assembly
# ==========================================================================


def test_matrix_free_matvec_equals_the_assembled_product():
    prob = poisson()
    ctx = build_context(prob, {"a0": 1.0})
    indices, data = assemble_triplets(ctx, {"a0": 1.0})
    n = len(ctx.dofmap.free)
    K = dense_from(indices, data, n)

    v = jnp.asarray(np.random.default_rng(0).normal(size=n))
    mv = tangent_matvec(ctx, {"a0": 1.0})(v)
    assert float(jnp.abs(mv - K @ v).max() / jnp.abs(K @ v).max()) < 1e-12


def test_assembly_returns_one_entry_per_csr_nonzero():
    prob = poisson(deg=3, refine=2)
    ctx = build_context(prob, {"a0": 1.0})
    repeated = build_context(prob, {"a0": 2.0})
    indices, data = assemble_triplets(ctx, {"a0": 1.0})
    n = len(ctx.dofmap.free)
    keys = indices[:, 0] * n + indices[:, 1]

    assert repeated.assembly is ctx.assembly
    assert len(data) == ctx.assembly.nnz
    assert np.all(keys[1:] > keys[:-1])
    assert ctx.assembly.indptr.shape == (n + 1,)
    assert ctx.assembly.indptr[-1] == len(data)
    # Interior supports overlap, so the unique pattern must be smaller than
    # the valid raw element-entry stream scattered into it.
    n_raw = np.count_nonzero(ctx.assembly.scatter < ctx.assembly.nnz)
    assert len(data) < n_raw


def test_builtin_element_kernels_match_generic_energy_hessians():
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1)
    scalar = jx.FunctionSpace(patch)
    vector = jx.FunctionSpace(patch, vec=2)

    class GenericPoisson(jx.Poisson):
        def energy(self, grad_u, u, x, params):
            return super().energy(grad_u, u, x, params)

    class GenericElasticity(jx.LinearElasticity):
        def energy(self, grad_u, u, x, params):
            return super().energy(grad_u, u, x, params)

    class GenericPlate(jx.KirchhoffPlate):
        def energy(self, grad_u, u, x, params, hess_u=None):
            return super().energy(grad_u, u, x, params, hess_u)

    cases = [
        (
            jx.Poisson(scalar),
            GenericPoisson(scalar),
            {"a0": lambda x, p: 1.0 + x[0]},
        ),
        (
            jx.LinearElasticity(vector),
            GenericElasticity(vector),
            {"E": 2.0, "nu": 0.27},
        ),
        (
            jx.KirchhoffPlate(scalar),
            GenericPlate(scalar),
            {"E": 2.0, "nu": 0.27, "t": 0.1},
        ),
    ]

    for optimized, generic, params in cases:
        fast_ctx = build_context(optimized, params)
        slow_ctx = build_context(generic, params)
        fast_indices, fast_data = assemble_triplets(fast_ctx, params)
        slow_indices, slow_data = assemble_triplets(slow_ctx, params)
        np.testing.assert_array_equal(fast_indices, slow_indices)
        np.testing.assert_allclose(fast_data, slow_data, rtol=1e-13, atol=1e-14)


def test_matrix_free_diagonal_equals_the_assembled_diagonal():
    prob = poisson()
    ctx = build_context(prob, {"a0": 1.0})
    indices, data = assemble_triplets(ctx, {"a0": 1.0})
    n = len(ctx.dofmap.free)
    K = dense_from(indices, data, n)
    d = tangent_diagonal(ctx, {"a0": 1.0})
    np.testing.assert_allclose(np.asarray(d), np.diag(K), rtol=1e-12)


def test_matrix_free_solution_equals_the_assembled_one():
    ref = jx.solve(poisson(), params={"a0": 1.0}, linear=LinearOptions(method="dense"))
    mf = jx.solve(
        poisson(),
        params={"a0": 1.0},
        linear=LinearOptions(method="cg", matrix_free=True, tol=1e-13),
    )
    assert float(jnp.abs(mf.u - ref.u).max()) < 1e-12


def test_symmetric_superlu_ordering_matches_colamd():
    """The symmetry-aware factorisation is an ordering change, not a new solve."""
    problem = poisson(deg=3, refine=3)
    params = {"a0": 1.0}
    symmetric = jx.solve(
        problem,
        params=params,
        linear=LinearOptions(method="scipy", superlu_ordering="auto"),
    )
    colamd = jx.solve(
        problem,
        params=params,
        linear=LinearOptions(method="scipy", superlu_ordering="COLAMD"),
    )
    np.testing.assert_allclose(symmetric.u, colamd.u, rtol=2e-13, atol=2e-14)


def test_superlu_ordering_is_validated():
    with pytest.raises(ValueError, match="SuperLU ordering"):
        LinearOptions(superlu_ordering="not-an-ordering")


def test_matrix_free_takes_the_same_iteration_count_as_assembled():
    """Same operator and same preconditioner, so the Krylov histories match."""
    prob = poisson()
    ctx = build_context(prob, {"a0": 1.0})
    indices, data = assemble_triplets(ctx, {"a0": 1.0})
    n = len(ctx.dofmap.free)
    K = dense_from(indices, data, n)
    diag = np.diag(K)
    b = np.asarray(jx.methods.galerkin.rhs(prob, {"a0": 1.0}, ctx))

    def cg_iterations(matvec):
        u, r = np.zeros(n), b.copy()
        z = r / diag
        p, rz, it = z.copy(), r @ z, 0
        while np.linalg.norm(r) > 1e-10 * np.linalg.norm(b) and it < 500:
            Ap = matvec(p)
            alpha = rz / (p @ Ap)
            u += alpha * p
            r -= alpha * Ap
            z = r / diag
            rz_new = r @ z
            p = z + (rz_new / rz) * p
            rz = rz_new
            it += 1
        return it

    free_mv = tangent_matvec(ctx, {"a0": 1.0})
    n_assembled = cg_iterations(lambda v: K @ v)
    n_free = cg_iterations(lambda v: np.asarray(free_mv(jnp.asarray(v))))
    assert n_assembled == n_free


def test_matrix_free_vjp_costs_one_adjoint_solve():
    opts = LinearOptions(method="cg", matrix_free=True, tol=1e-13)

    def loss(a0):
        return jnp.sum(jx.solve(poisson(), params={"a0": a0}, linear=opts).u ** 2)

    reset_solve_counter()
    g = float(jax.grad(loss)(1.0))
    assert SOLVE_COUNTER["adjoint"] == 1
    h = 1e-6
    fd = float((loss(1.0 + h) - loss(1.0 - h)) / (2 * h))
    assert abs(g - fd) / abs(fd) < 1e-7


def test_matrix_free_rejects_a_direct_method():
    with pytest.raises(ValueError, match="iterative"):
        LinearOptions(method="scipy", matrix_free=True)


@pytest.mark.parametrize("chunk", [1, 7, 1000])
def test_chunked_assembly_matches_eager_assembly(chunk):
    prob = poisson()
    eager = build_context(prob, {"a0": 1.0})
    chunked = build_context(prob, {"a0": 1.0}, chunk=chunk)
    assert chunked.chunked and not eager.chunked

    i1, d1 = assemble_triplets(eager, {"a0": 1.0})
    i2, d2 = assemble_triplets(chunked, {"a0": 1.0})
    n = len(eager.dofmap.free)
    np.testing.assert_allclose(
        dense_from(i1, d1, n), dense_from(i2, d2, n), atol=1e-14
    )


def test_chunked_solve_matches_and_still_differentiates():
    ref = jx.solve(poisson(), params={"a0": 1.0}, linear=LinearOptions(method="dense"))
    got = jx.solve(
        poisson(), params={"a0": 1.0}, linear=LinearOptions(method="dense"), chunk=5
    )
    assert float(jnp.abs(got.u - ref.u).max()) < 1e-13

    def loss(a0, **kw):
        return jnp.sum(jx.solve(poisson(), params={"a0": a0}, **kw).u ** 2)

    g_eager = float(jax.grad(loss)(1.0))
    g_chunk = float(jax.grad(lambda a: loss(a, chunk=5))(1.0))
    assert abs(g_eager - g_chunk) / abs(g_eager) < 1e-10


def test_chunked_and_matrix_free_compose():
    ref = jx.solve(poisson(), params={"a0": 1.0}, linear=LinearOptions(method="dense"))
    got = jx.solve(
        poisson(),
        params={"a0": 1.0},
        linear=LinearOptions(method="cg", matrix_free=True, tol=1e-13),
        chunk=5,
    )
    assert float(jnp.abs(got.u - ref.u).max()) < 1e-12


# ==========================================================================
# Section 26 -- batched solves under vmap
# ==========================================================================


@pytest.mark.parametrize("method", ["dense", "cg", "bicgstab", "scipy"])
def test_batched_solve_equals_the_python_loop(method):
    opts = LinearOptions(method=method, tol=1e-13)
    f = lambda a0: jx.solve(poisson(deg=2, refine=2), params={"a0": a0}, linear=opts).u
    thetas = jnp.linspace(0.5, 4.0, 8)
    loop = jnp.stack([f(a) for a in thetas])
    batch = jax.vmap(f)(thetas)
    assert float(jnp.abs(batch - loop).max()) < 1e-12


def test_batched_solve_differentiates():
    opts = LinearOptions(method="dense")

    def total(thetas):
        f = lambda a0: jnp.sum(
            jx.solve(poisson(deg=2, refine=2), params={"a0": a0}, linear=opts).u ** 2
        )
        return jnp.sum(jax.vmap(f)(thetas))

    thetas = jnp.array([1.0, 2.0])
    g = jax.grad(total)(thetas)
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.abs(g).min()) > 0


# ==========================================================================
# Section 27 -- transient dynamics
# ==========================================================================

E_BAR, RHO_BAR, L_BAR = 2.0, 3.0, 1.0


def bar(deg=3, refine=5):
    V = jx.FunctionSpace(jx.primitives.interval(0.0, L_BAR).elevate(deg).refine(refine))
    prob = jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where="left"), jx.DirichletBC(0.0, where="right")],
    )
    indices, m, k, ctx = jx.mass_matrix(prob, rho=RHO_BAR, params={"a0": E_BAR})
    return indices, m, k, ctx, V


def test_mass_matrix_shares_the_stiffness_sparsity_pattern():
    prob = poisson(deg=2, refine=2)
    ctx = build_context(prob, {"a0": 1.0})
    ik, _ = assemble_triplets(ctx, {"a0": 1.0})
    im, _ = mass_triplets(ctx, 1.0)
    np.testing.assert_array_equal(ik, im)


def test_mass_matrix_integrates_to_the_domain_mass():
    """``1^T M 1`` is the total mass, once no dof is constrained away."""
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 2, 3).elevate(2).refine(2))
    prob = jx.Poisson(V)
    ctx = build_context(prob, {"a0": 1.0})
    indices, data = mass_triplets(ctx, 5.0)
    n = len(ctx.dofmap.free)
    total = float(np.ones(n) @ dense_from(indices, data, n) @ np.ones(n))
    assert abs(total - 5.0 * 2.0 * 3.0) < 1e-10


def test_bar_natural_frequencies_match_the_analytic_ones():
    indices, m, k, _, _ = bar()
    omega = dynamics.natural_frequencies(indices, k, m, 5)
    exact = np.array(
        [(n + 1) * np.pi / L_BAR * np.sqrt(E_BAR / RHO_BAR) for n in range(5)]
    )
    np.testing.assert_allclose(omega, exact, rtol=1e-6)


def test_newmark_conserves_energy_over_a_thousand_steps():
    indices, m, k, ctx, V = bar()
    n = len(ctx.dofmap.free)
    x = np.asarray(V.cpts).reshape(-1)[ctx.dofmap.free]
    u0 = jnp.asarray(np.sin(np.pi * x / L_BAR))
    t, U, Vv = dynamics.integrate(
        indices, k, m, dt=2e-3, n_steps=1000, u0=u0, v0=jnp.zeros(n),
        linear=LinearOptions(method="dense"),
    )
    e = dynamics.energy(indices, k, m, U, Vv)
    assert float(jnp.abs(e - e[0]).max() / e[0]) < 1e-8


def test_generalized_alpha_damps_the_high_modes():
    indices, m, k, ctx, V = bar()
    n = len(ctx.dofmap.free)
    x = np.asarray(V.cpts).reshape(-1)[ctx.dofmap.free]
    u0 = jnp.asarray(np.sin(np.pi * x / L_BAR))
    kw = dict(dt=2e-3, n_steps=200, u0=u0, v0=jnp.zeros(n),
              linear=LinearOptions(method="dense"))
    _, U1, V1 = dynamics.integrate(indices, k, m, stepper=dynamics.newmark(), **kw)
    _, U2, V2 = dynamics.integrate(
        indices, k, m, stepper=dynamics.generalized_alpha(0.5), **kw
    )
    e1 = dynamics.energy(indices, k, m, U1, V1)
    e2 = dynamics.energy(indices, k, m, U2, V2)
    # dissipative by construction, and strictly more so than Newmark
    assert e2[-1] < e1[-1]
    assert float(jnp.abs(e1 - e1[0]).max() / e1[0]) < 1e-8


def test_free_vibration_follows_the_analytic_period():
    """A single-mode initial condition oscillates at that mode's frequency."""
    indices, m, k, ctx, V = bar()
    n = len(ctx.dofmap.free)
    x = np.asarray(V.cpts).reshape(-1)[ctx.dofmap.free]
    u0 = jnp.asarray(np.sin(np.pi * x / L_BAR))
    omega = np.pi / L_BAR * np.sqrt(E_BAR / RHO_BAR)
    period = 2 * np.pi / omega
    n_steps = 400
    dt = period / n_steps
    _, U, _ = dynamics.integrate(
        indices, k, m, dt=dt, n_steps=n_steps, u0=u0, v0=jnp.zeros(n),
        linear=LinearOptions(method="dense"),
    )
    # after exactly one period the state should return
    assert float(jnp.abs(U[-1] - u0).max() / jnp.abs(u0).max()) < 1e-3


def test_time_loop_is_differentiable_and_checkpointing_agrees():
    indices, m, k, ctx, V = bar(deg=2, refine=3)
    n = len(ctx.dofmap.free)
    x = np.asarray(V.cpts).reshape(-1)[ctx.dofmap.free]
    u0 = jnp.asarray(np.sin(np.pi * x / L_BAR))

    def final_energy(scale, checkpoint_every=None):
        _, U, Vv = dynamics.integrate(
            indices, k * scale, m, dt=1e-3, n_steps=40, u0=u0, v0=jnp.zeros(n),
            linear=LinearOptions(method="dense"), checkpoint_every=checkpoint_every,
        )
        return jnp.sum(U[-1] ** 2)

    g = float(jax.grad(final_energy)(1.0))
    g_ck = float(jax.grad(lambda s: final_energy(s, checkpoint_every=10))(1.0))
    h = 1e-6
    fd = float((final_energy(1.0 + h) - final_energy(1.0 - h)) / (2 * h))
    assert abs(g - fd) / abs(fd) < 1e-5
    assert abs(g - g_ck) / abs(g) < 1e-10


def test_checkpoint_every_must_divide_the_step_count():
    indices, m, k, ctx, V = bar(deg=2, refine=2)
    n = len(ctx.dofmap.free)
    with pytest.raises(ValueError, match="divide"):
        dynamics.integrate(
            indices, k, m, dt=1e-3, n_steps=10, u0=jnp.zeros(n), v0=jnp.zeros(n),
            checkpoint_every=3,
        )


def test_generalized_alpha_reduces_to_newmark_at_unit_spectral_radius():
    st = dynamics.generalized_alpha(1.0)
    assert abs(st.beta - 0.25) < 1e-12 and abs(st.gamma - 0.5) < 1e-12
    assert abs(st.alpha_m - st.alpha_f) < 1e-12


# ==========================================================================
# Boundary-condition selection is per problem, not per space
# ==========================================================================


def test_neumann_predicates_do_not_interfere():
    """Two problems selecting different parts of one labeled side.

    The selection used to be registered on the shared FunctionSpace under a
    name derived from the contributing labels and the element count, so two
    predicates that agreed on both silently overwrote one another -- and the
    first problem's traction moved to the other half of the edge.
    """
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(1).refine(2))
    on_bottom = lambda x: np.abs(x[:, 1]) < 1e-9  # noqa: E731
    left_half = lambda x: on_bottom(x) & (x[:, 0] < 0.5)  # noqa: E731
    right_half = lambda x: on_bottom(x) & (x[:, 0] > 0.5)  # noqa: E731

    def make(pred):
        return jx.Poisson(
            V,
            dirichlet=[jx.DirichletBC(0.0, where="left")],
            neumann=[jx.Neumann(lambda x, n, p: jnp.array([1.0]), where=pred)],
        )

    first = make(left_half)
    first_elems = np.array(first.resolve_neumann()[0][1].elems).copy()
    second = make(right_half)

    assert not np.array_equal(first_elems, np.array(second.resolve_neumann()[0][1].elems))
    np.testing.assert_array_equal(
        np.array(first.resolve_neumann()[0][1].elems), first_elems
    )


def test_neumann_predicate_does_not_mutate_the_space():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(1).refine(1))
    before = set(V.boundaries)
    jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where="left")],
        neumann=[
            jx.Neumann(
                lambda x, n, p: jnp.array([1.0]), where=lambda x: np.abs(x[:, 1]) < 1e-9
            )
        ],
    )
    assert set(V.boundaries) == before


def test_predicate_and_label_selections_agree():
    """A predicate covering a whole side must reproduce that side exactly."""
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2))
    traction = lambda x, n, p: jnp.array([1.0])  # noqa: E731

    by_label = jx.Poisson(
        V, dirichlet=[jx.DirichletBC(0.0, where="left")],
        neumann=[jx.Neumann(traction, where="right")],
    )
    by_predicate = jx.Poisson(
        V, dirichlet=[jx.DirichletBC(0.0, where="left")],
        neumann=[jx.Neumann(traction, where=lambda x: np.abs(x[:, 0] - 1.0) < 1e-9)],
    )
    a = jx.solve(by_label, params={"a0": 1.0})
    b = jx.solve(by_predicate, params={"a0": 1.0})
    assert float(jnp.abs(a.u - b.u).max()) < 1e-12


def test_neumann_predicate_selecting_nothing_raises():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(1).refine(1))
    with pytest.raises(ValueError, match="selected no boundary elements"):
        jx.Poisson(
            V,
            dirichlet=[jx.DirichletBC(0.0, where="left")],
            neumann=[
                jx.Neumann(lambda x, n, p: jnp.array([1.0]), where=lambda x: x[:, 0] > 99.0)
            ],
        )
