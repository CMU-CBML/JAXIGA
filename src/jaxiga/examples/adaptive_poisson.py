"""Refine locally to resolve a concentrated Poisson source on a unit square."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

import jaxiga as jx


def main():
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2)
    space = jx.FunctionSpace(patch)

    # Boundary conditions must be rebuilt for each refined space.
    def make_problem(space):
        return jx.Poisson(
            space,
            dirichlet=[jx.DirichletBC(0.0, where=side)
                       for side in ("left", "right", "bottom", "top")],
            source=lambda x, p: jnp.exp(-80 * ((x[0] - 0.65)**2 + (x[1] - 0.35)**2)),
        )

    solution, space, history = jx.adapt(
        make_problem, params={"a0": 1.0}, space=space,
        n_cycles=3, frac=0.4, verbose=True,
    )
    print(f"Elements: {history[0]['n_elems']} -> {history[-1]['n_elems']}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    points, values = solution.sample(n=5)
    contour = axes[0].tricontourf(points[:, 0], points[:, 1], values[:, 0], levels=20)
    fig.colorbar(contour, ax=axes[0], label="u")
    axes[0].set_title("Poisson solution")
    # This unit-square geometry maps parametric element boxes directly to x, y.
    for x0, y0, x1, y1 in space.elem_vertex:
        axes[1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False))
    axes[1].set_title("Locally refined mesh")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", aspect="equal")
    plt.show()


if __name__ == "__main__":
    main()
