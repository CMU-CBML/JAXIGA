"""Solve -u'' = pi**2*sin(pi*x), u(0) = u(1) = 0, with three methods."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxiga as jx


def main():
    space = jx.FunctionSpace(jx.primitives.interval().elevate(3).refine(3))
    problem = jx.Poisson(
        space,
        dirichlet=[jx.DirichletBC(0.0, where="left"),
                   jx.DirichletBC(0.0, where="right")],
        source=lambda x, p: jnp.pi**2 * jnp.sin(jnp.pi * x[0]),
    )

    # The geometry, PDE and boundary conditions are identical for each method.
    for method, style in zip(("galerkin", "energy", "collocation"), ("-", "--", ":")):
        solution = jx.solve(problem, params={"a0": 1.0}, method=method)
        error = jx.errornorm(solution, lambda x: jnp.sin(jnp.pi * x[0]), "L2")
        print(f"{method:12s}: relative L2 error = {error:.3e}")
        points, values = solution.sample(n=6)
        order = np.argsort(points[:, 0])
        plt.plot(points[order, 0], values[order, 0], style, label=method)

    x = np.linspace(0, 1, 17)
    plt.plot(x, np.sin(np.pi * x), "k.", label="exact")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
