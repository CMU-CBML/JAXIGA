"""Recover k(x) = exp(theta[0] + theta[1]*x) from synthetic 1D pressures."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxiga as jx
from jaxiga.solvers.optimize import minimize_lbfgs


class Darcy(jx.Problem):
    is_linear = True

    def energy(self, grad_u, u, x, params):
        theta = params["theta"]
        permeability = jnp.exp(theta[0] + theta[1] * x[0])
        return 0.5 * permeability * jnp.sum(grad_u**2)


def main():
    space = jx.FunctionSpace(jx.primitives.interval().elevate(3).refine(3))
    problem = Darcy(
        space,
        dirichlet=[jx.DirichletBC(0.0, where="left"),
                   jx.DirichletBC(0.0, where="right")],
        source=lambda x, p: 1.0,
    )
    theta_true = jnp.array([0.4, 0.7])
    x_obs = jnp.linspace(0.1, 0.9, 12)[:, None]
    targets = jx.locate_points(space, x_obs)  # Locate once, outside differentiation.
    truth = jx.solve(problem, params={"theta": theta_true})
    observed = truth.probe(x_obs, targets=targets)

    def loss(theta):
        solution = jx.solve(problem, params={"theta": theta})
        predicted = solution.probe(x_obs, targets=targets)
        return jnp.mean((predicted - observed)**2) / jnp.mean(observed**2)

    fit = minimize_lbfgs(jax.jit(loss), jnp.zeros(2), max_steps=50, tol=1e-9)
    print(f"True parameters: {theta_true}; recovered: {fit.x}")
    print(f"Relative pressure misfit: {fit.value:.3e}; converged: {fit.converged}")

    x = jnp.linspace(0, 1, 100)
    plt.plot(x, jnp.exp(theta_true[0] + theta_true[1] * x), label="true")
    plt.plot(x, jnp.exp(fit.x[0] + fit.x[1] * x), "--", label="recovered")
    plt.xlabel("x")
    plt.ylabel("Permeability k(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
