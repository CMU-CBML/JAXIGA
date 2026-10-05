"""Differentiate J(a) = integral(u**2 dx) through the solve -a*u'' = 1."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxiga as jx


def main():
    space = jx.FunctionSpace(jx.primitives.interval().elevate(2).refine(3))
    problem = jx.Poisson(
        space,
        dirichlet=[jx.DirichletBC(0.0, where="left"),
                   jx.DirichletBC(0.0, where="right")],
        source=lambda x, p: 1.0,
    )
    basis = jx.evaluate(space, jx.gauss(space))

    def objective(a):
        solution = jx.solve(problem, params={"a0": a})
        u = solution.at(basis=basis)[..., 0]
        return jnp.sum(basis.w * u**2)

    value, derivative = jax.value_and_grad(objective)(2.0)
    h = 1e-4
    finite_difference = (objective(2.0 + h) - objective(2.0 - h)) / (2 * h)
    print(f"J(2) = {value:.8f}")
    print(f"dJ/da: autodiff = {derivative:.8f}, finite difference = {finite_difference:.8f}")

    solution = jx.solve(problem, params={"a0": 2.0})
    points, values = solution.sample(n=6)
    order = np.argsort(points[:, 0])
    plt.plot(points[order, 0], values[order, 0])
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"Diffusion a = 2; dJ/da = {derivative:.6f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
