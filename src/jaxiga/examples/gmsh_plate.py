"""Mesh CAD from a .geo file, then solve a perforated plate in tension.

Install with: pip install -e '.[gmsh,post]' matplotlib
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxiga as jx


def main():
    geometry = Path(__file__).with_name("geometries") / "perforated_plate.geo"
    mesh = jx.read_gmsh(geometry, mesh_size="auto", geometry_order=3)
    mesh = mesh.refine()  # Insert knots and refit new boundary samples to CAD.
    space = jx.FunctionSpace(mesh, vec=2)
    print(f"{mesh.meshing_method}: {len(mesh)} quads; min scaled Jacobian: {mesh.min_scaled_jacobian:.3f}")
    print(f"Sampled CAD boundary error: {mesh.geometry_error:.3g}")
    problem = jx.LinearElasticity(
        space, plane="stress",
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="fixed")],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([10.0, 0.0]), where="loaded")],
    )
    solution = jx.solve(problem, params={"E": 1e5, "nu": 0.3})
    output = solution.to_vtk("gmsh_plate.vtu", n=4, fields=("von_mises",))
    print(f"ParaView: {output}")
    basis = jx.evaluate(space, jx.grid(space, 8))
    points = np.asarray(basis.x).reshape(-1, 8, 8, 2)
    stress = np.asarray(solution.field("von_mises", basis=basis)).reshape(-1, 8, 8)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for xy, values in zip(points, stress):
        edge = np.concatenate((xy[0], xy[1:, -1], xy[-1, -2::-1], xy[-2::-1, 0]))
        axes[0].plot(edge[:, 0], edge[:, 1], "k-", lw=0.4)
        image = axes[1].pcolormesh(xy[..., 0], xy[..., 1], values, shading="gouraud",
                                    vmin=0, vmax=stress.max())
    fig.colorbar(image, ax=axes[1], label="Von Mises stress")
    axes[0].set_title("Gmsh quadrilateral mesh")
    axes[1].set_title("Horizontal tension; left edge fixed")
    for ax in axes:
        ax.set(xlabel="x", ylabel="y", aspect="equal")
    plt.show()


if __name__ == "__main__":
    main()
