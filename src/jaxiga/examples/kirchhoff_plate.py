"""Bend a clamped unit-square plate under a uniform transverse load."""

import matplotlib.pyplot as plt

import jaxiga as jx


def main():
    # Cubic splines provide the C1 continuity needed by this fourth-order PDE.
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(3)
    space = jx.FunctionSpace(patch)
    problem = jx.KirchhoffPlate(
        space,
        dirichlet=[jx.ClampedBC(where=side)
                   for side in ("left", "right", "bottom", "top")],
        source=lambda x, p: 1.0,
    )
    solution = jx.solve(problem, params={"E": 210e3, "nu": 0.3, "t": 0.01})
    centre = solution.probe([[0.5, 0.5]])[0, 0]
    print(f"Centre deflection: {centre:.6e}")

    points, values = solution.sample(n=5)
    plt.tricontourf(points[:, 0], points[:, 1], values[:, 0], levels=20)
    plt.colorbar(label="Deflection w")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Clamped Kirchhoff–Love plate")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
