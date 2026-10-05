"""Solve -Laplacian(u) = 2*pi**2*sin(pi*x)*sin(pi*y) on the unit square."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxiga as jx


def main():
    # Cubic splines on an 8 x 8 element mesh.
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(3)
    space = jx.FunctionSpace(patch)

    def exact(x):
        return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])

    problem = jx.Poisson(
        space,
        dirichlet=[jx.DirichletBC(0.0, where=side)
                   for side in ("left", "right", "bottom", "top")],
        source=lambda x, p: 2 * jnp.pi**2 * exact(x),
    )
    solution = jx.solve(problem, params={"a0": 1.0})
    print(f"Relative L2 error: {jx.errornorm(solution, exact, 'L2'):.3e}")

    points, values = solution.sample(n=5)
    plt.tricontourf(points[:, 0], points[:, 1], values[:, 0], levels=20)
    plt.colorbar(label="u")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Poisson equation")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
