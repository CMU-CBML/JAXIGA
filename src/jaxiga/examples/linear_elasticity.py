"""Stretch a square plate with a circular hole and plot the von Mises stress."""

import matplotlib.pyplot as plt
import numpy as np

import jaxiga as jx


def main():
    # Four NURBS patches represent [-4, 4]^2 with an exact circular hole.
    patches = [patch.elevate(3).refine(3)
               for patch in jx.primitives.plate_with_hole_diagonal(radius=1, side=4)]
    space = jx.FunctionSpace(patches, vec=2)  # Shared patch interfaces are glued.
    problem = jx.LinearElasticity(
        space, plane="stress",
        dirichlet=[
            jx.DirichletBC([0.0, 0.0], where=lambda x: np.isclose(x[:, 1], -4)),
            jx.DirichletBC([0.0, 0.008], where=lambda x: np.isclose(x[:, 1], 4)),
        ],
    )
    # The hole and lateral edges are traction-free (the natural condition).
    solution = jx.solve(problem, params={"E": 1e5, "nu": 0.3})
    n = 5
    basis = jx.evaluate(space, jx.grid(space, n))
    points = np.asarray(basis.x).reshape(-1, n, n, 2)
    stress = np.asarray(solution.field("von_mises", basis=basis)).reshape(-1, n, n)
    print(f"Degrees of freedom: {space.n_dofs}; peak sampled stress: {stress.max():.3f}")

    # Draw each element separately so the plotting never fills the hole.
    for xy, values in zip(points, stress):
        image = plt.pcolormesh(xy[..., 0], xy[..., 1], values, shading="gouraud",
                               vmin=0, vmax=stress.max(), cmap="viridis")
    plt.colorbar(image, label="Von Mises stress")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plate with a hole under vertical tension")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
