"""Kirchhoff-Love shells on embedded NURBS surfaces."""

import jax.numpy as jnp
import numpy as np
import pytest

import jaxiga as jx
from jaxiga.examples import scordelis_lo_roof as roof
from jaxiga.geometry.nurbs import Patch
from jaxiga.space.pointset import boundary_gauss, grid


def _roof_problem(degree=3, elements=4):
    knots = np.linspace(0, 1, elements + 1)[1:-1]
    space = jx.FunctionSpace(roof.roof_patch().elevate(degree).insert_knots(knots, knots), vec=3)
    return jx.KirchhoffLoveShell(space)


def _point(problem, e=3, q=5):
    params = problem.parameters(1.0, 0.3, 0.1)
    return {"E": 1.0, "nu": 0.3, "t": 0.1, "a": params["a"].values[e, q], "da": params["da"].values[e, q]}


def test_flat_shell_reproduces_the_kirchhoff_plate():
    E, nu, t, q = 210e3, 0.3, 0.01, 1.0
    flat = jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(3)
    clamps = lambda: [jx.ClampedBC(where=s) for s in ("left", "right", "bottom", "top")]
    plate = jx.solve(jx.KirchhoffPlate(jx.FunctionSpace(flat), source=lambda x, p: q, dirichlet=clamps()),
                     params={"E": E, "nu": nu, "t": t})
    surface = Patch.create(flat.knots, flat.degree, np.c_[np.asarray(flat.ctrl_pts), np.zeros(len(flat.ctrl_pts))],
                           np.asarray(flat.weights), labels=dict(flat.labels))
    shell = jx.KirchhoffLoveShell(jx.FunctionSpace(surface, vec=3), dirichlet=clamps(),
                                  source=lambda x, p: jnp.array([0.0, 0.0, q]))
    w = jx.solve(shell, params=shell.parameters(E, nu, t))
    a = np.asarray(plate.at(grid(plate.space, 4)))[..., 0]
    b = np.asarray(w.at(grid(shell.space, 4)))
    np.testing.assert_allclose(b[..., 2], a, rtol=0, atol=1e-12 * np.abs(a).max())
    assert np.abs(b[..., :2]).max() == 0.0


def test_rigid_rotation_is_strain_free_on_a_curved_surface():
    problem = _roof_problem()
    p = _point(problem)
    omega = jnp.array([0.3, -0.7, 0.5])
    grad_u = jnp.cross(omega, p["a"]).T          # u = omega x x  =>  u_,a = omega x a_a
    hess_u = jnp.cross(omega, p["da"]).T
    eps, kappa = problem.strains(grad_u, hess_u, p)
    assert float(jnp.abs(eps).max()) < 1e-12 and float(jnp.abs(kappa).max()) < 1e-12

    # A finite rotation leaves the exact (geometrically nonlinear) strains at zero.
    exact = jx.KirchhoffLoveShell(problem.space, geometrically_nonlinear=True)
    c, s = np.cos(0.8), np.sin(0.8)
    Q = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    eps, kappa = exact.strains(((Q - jnp.eye(3)) @ p["a"].T), ((Q - jnp.eye(3)) @ p["da"].T), p)
    assert float(jnp.abs(eps).max()) < 1e-10 and float(jnp.abs(kappa).max()) < 1e-10


def test_linear_strains_are_the_derivative_of_the_exact_strains():
    problem = _roof_problem()
    exact = jx.KirchhoffLoveShell(problem.space, geometrically_nonlinear=True)
    p = _point(problem)
    rng = np.random.default_rng(0)
    du, d2u = jnp.asarray(rng.normal(size=(3, 2))), jnp.asarray(rng.normal(size=(3, 3)))
    lin = problem.strains(du, d2u, p)
    h = 1e-6
    for k in range(2):
        fd = (exact.strains(h * du, h * d2u, p)[k] - exact.strains(-h * du, -h * d2u, p)[k]) / (2 * h)
        np.testing.assert_allclose(lin[k], fd, rtol=1e-6, atol=1e-9)


def test_scordelis_lo_roof_coarse_regression():
    space, solution = roof.solve_roof(3, 8)
    uz = -roof.displacement_at(space, solution, 0.0, 0.5)[2]
    assert abs(uz - 0.300065) < 2e-5          # converged value 0.30059, reference 0.3024


def test_boundary_integrals_on_surfaces_are_rejected():
    problem = _roof_problem(degree=2, elements=2)
    from jaxiga.space.evaluation import evaluate
    with pytest.raises(NotImplementedError):
        evaluate(problem.space, boundary_gauss(problem.space, "free_left"))
