"""Perforated plate in tension: curved triangle mesh and plane-stress solution.

Run: python -m jaxiga.examples.gmsh_plate_triangles
Requires: pip install -e '.[gmsh,post]' matplotlib
"""
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import numpy as np

import jaxiga as jx


def main():
    geometry = Path(__file__).with_name("geometries") / "perforated_plate.geo"
    mesh = jx.read_gmsh(geometry, cell_type="simplex", mesh_size=1.0, geometry_order=2)
    space = jx.FunctionSpace(mesh, vec=2)
    problem = jx.LinearElasticity(
        space, plane="stress",
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="fixed")],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([10.0, 0.0]), where="loaded")],
    )
    solution = jx.solve(problem, params={"E": 1e5, "nu": 0.3})
    output = solution.to_vtk("gmsh_plate_triangles.vtu", n=4, fields=("von_mises",))
    print(f"ParaView: {output}")
    print(f"{len(mesh)} quadratic triangles, {space.n_dofs} DOFs; "
          f"min scaled Jacobian: {mesh.min_scaled_jacobian:.3f}")
    print(f"Sampled CAD boundary error: {mesh.geometry_error:.3g}")

    samples = jx.grid(space, 8)
    basis = jx.evaluate(space, samples)
    points = np.asarray(basis.x)
    stress = np.asarray(solution.field("von_mises", basis=basis)).ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Draw the three curved edges of each actual mesh cell.
    ref = np.asarray(samples.ref)
    edges = []
    for mask, axis in ((np.isclose(ref[:, 1], 0), 0),
                       (np.isclose(ref.sum(axis=1), 1), 1),
                       (np.isclose(ref[:, 0], 0), 1)):
        ids = np.flatnonzero(mask)
        ids = ids[np.argsort(ref[ids, axis])]
        edges.extend(points[:, ids])
    axes[0].add_collection(LineCollection(edges, colors="k", linewidths=0.4))
    axes[0].autoscale()

    # Sample triangles stay inside each cell, preserving holes and stress jumps.
    local = mtri.Triangulation(ref[:, 0], ref[:, 1]).triangles
    triangles = np.concatenate([local + e * basis.n_q for e in range(space.n_elems)])
    xy = points.reshape(-1, 2)
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    image = axes[1].tripcolor(triangulation, stress, shading="gouraud", vmin=0)
    fig.colorbar(image, ax=axes[1], label="Von Mises stress")
    axes[0].set_title("Gmsh quadratic triangle mesh")
    axes[1].set_title("Horizontal tension; left edge fixed")
    for ax in axes:
        ax.set(xlabel="x", ylabel="y", aspect="equal")
    plt.show()


if __name__ == "__main__":
    main()
