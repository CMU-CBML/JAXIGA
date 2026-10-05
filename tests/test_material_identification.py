"""The material-identification example: exact edges, adjoint gradient and a short recovery."""

import jax
import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.examples import material_identification as mi


def _solver():
    V, nodes, to_quad = mi.pixel_space()
    bcs = [jx.DirichletBC(mi.measured_edges, where=w) for w in ("top", "bottom", "left", "right")]
    problem = jx.LinearElasticity(V, plane="stress", dirichlet=bcs)

    def solve(phi, log_contrast, edges):
        params = {"E": jx.PointField(jnp.exp(log_contrast * to_quad(phi))), "nu": mi.NU, "edges": edges}
        return jx.solve(problem, params=params, linear=jx.LinearOptions(method="scipy")).at(basis=nodes).reshape(mi.N, mi.N, 2)
    return solve


def test_edges_are_reproduced_and_gradient_matches_finite_differences():
    solve = _solver()
    u, truth, contrast = mi.synthetic_specimen(solve)
    edges = mi.edges_of(jnp.asarray(u))
    field = np.asarray(solve(jnp.asarray(truth, float), np.log(contrast), edges))
    # Control-point interpolation of the (noisy) edge data.
    assert np.abs(field[0] - u[0]).max() < 5e-3 and np.abs(field[:, -1] - u[:, -1]).max() < 5e-3

    target = jnp.asarray(u)
    loss = lambda z: jnp.sum((solve(jax.nn.sigmoid(z), np.log(contrast), edges) - target)[1:-1, 1:-1] ** 2)
    z = jnp.asarray(np.random.default_rng(1).normal(0, .5, (mi.N, mi.N)))
    g = jax.grad(loss)(z)
    k, h = (25, 20), 1e-5
    fd = (loss(z.at[k].add(h)) - loss(z.at[k].add(-h))) / (2 * h)
    np.testing.assert_allclose(float(g[k]), float(fd), rtol=1e-4, atol=1e-12)
