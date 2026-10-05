"""Differentiable inverse problem: recover a permeability field from pressures.

This is where the JAX architecture pays off. `solve` is differentiable, so the
misfit between computed and measured pressures can be minimized directly, and
the adjoint solve happens automatically inside `jax.grad`. Each gradient costs
one forward solve plus one adjoint solve, independent of the number of
parameters.

Deviation from the architecture guide's sketch: it writes the permeability as
`jnp.exp(jx.Solution(Vk, p["log_k"], None, None)(x))` inside the energy kernel.
The kernel receives only `(grad_u, u, x, params)` -- no basis data -- so
evaluating a spline coefficient field there would need a per-point geometry
inversion at every quadrature point. Here the field is instead parametrized by
smooth modes in x, which is differentiable, cheap, and exercises exactly the
same capability: a gradient through the solve with respect to a field
parameter.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.solvers.linear import SOLVE_COUNTER, reset_solve_counter
from jaxiga.solvers.optimize import minimize_lbfgs

N_MODES = 3
ALL = lambda x: jnp.full(x.shape[0], True)


def log_permeability(x, theta):
    """Smooth field: a truncated cosine expansion with coefficients theta."""
    modes = jnp.array([
        jnp.cos(i * jnp.pi * x[0]) * jnp.cos(j * jnp.pi * x[1])
        for i in range(N_MODES)
        for j in range(N_MODES)
    ])
    return jnp.dot(theta, modes)


class Darcy(jx.Problem):
    """Steady Darcy flow: psi = 1/2 k(x) |grad p|^2 with k = exp(log_k)."""

    is_linear = True

    def energy(self, grad_u, u, x, params):
        k = jnp.exp(log_permeability(x, params["log_k"]))
        return 0.5 * k * jnp.sum(grad_u**2)


V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(3))
problem = Darcy(
    V,
    dirichlet=[jx.DirichletBC(0.0, where=ALL)],
    source=lambda x, p: 1.0,  # uniform recharge
)
print(V.summary())

# -- synthetic truth and measurements -------------------------------------

rng = np.random.default_rng(0)
theta_true = jnp.asarray(rng.normal(scale=0.4, size=N_MODES**2))

truth = jx.solve(problem, params={"log_k": theta_true})
x_obs = jnp.asarray(rng.uniform(0.15, 0.85, size=(40, 2)))
# locating the observation points is a host-side search: do it once, so the
# loss below stays jittable
targets = jx.locate_points(V, x_obs)
p_obs = truth.probe(x_obs, targets=targets)[:, 0]
print(f"{len(x_obs)} pressure measurements, range "
      f"[{float(p_obs.min()):.4f}, {float(p_obs.max()):.4f}]")


# -- inversion -------------------------------------------------------------


def loss(theta):
    sol = jx.solve(problem, params={"log_k": theta})
    misfit = jnp.mean((sol.probe(x_obs, targets=targets)[:, 0] - p_obs) ** 2)
    return misfit + 1e-6 * jnp.sum(theta**2)


theta0 = jnp.zeros(N_MODES**2)
print(f"\ninitial loss     {float(loss(theta0)):.6e}")

reset_solve_counter()
g = jax.grad(loss)(theta0)
print(f"solves per gradient: {SOLVE_COUNTER}   (independent of {N_MODES**2} parameters)")

# adjoint gradient against finite differences
h = 1e-5
fd = float((loss(theta0.at[0].add(h)) - loss(theta0.at[0].add(-h))) / (2 * h))
print(f"grad[0]: adjoint {float(g[0]): .6e}   finite difference {fd: .6e}")

result = minimize_lbfgs(jax.jit(loss), theta0, tol=1e-12, max_steps=300)
print(f"\n{result}")
print(f"final loss       {result.value:.6e}")

recovered = jx.solve(problem, params={"log_k": result.x})
pred = recovered.probe(x_obs, targets=targets)[:, 0]
print(f"measurement RMS misfit  {float(jnp.sqrt(jnp.mean((pred - p_obs) ** 2))):.3e}")

# The field, not the coefficients, is what is identifiable from the data.
grid = jnp.asarray([[i / 12, j / 12] for i in range(1, 12) for j in range(1, 12)])
k_true = jnp.exp(jax.vmap(log_permeability, in_axes=(0, None))(grid, theta_true))
k_rec = jnp.exp(jax.vmap(log_permeability, in_axes=(0, None))(grid, result.x))
rel = float(jnp.linalg.norm(k_rec - k_true) / jnp.linalg.norm(k_true))
print(f"permeability field relative error  {rel:.3e}")
