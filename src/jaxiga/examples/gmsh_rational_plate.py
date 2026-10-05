"""Exact conic hole boundaries and mixed quad/triangle plane-stress elasticity.

Run: python -m jaxiga.examples.gmsh_rational_plate --refine 1
Requires: pip install -e '.[gmsh,post]' matplotlib
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxiga as jx
from jaxiga.post.matplotlib import plot_mesh_field


def main(refinements=0):
    geometry = Path(__file__).with_name("geometries") / "perforated_plate.geo"
    mesh = jx.read_gmsh(
        geometry, mesh_size="auto", geometry_order=2, rational_geometry=True
    )
    patches = []
    for patch in mesh:
        patches.extend(
            jx.split_to_simplices(patch)
            if np.mean(patch.ctrl_pts[:, 0]) > 6
            else [patch]
        )
    patches = jx.refine_cells(patches, n=refinements)
    space = jx.FunctionSpace(patches, vec=2)
    problem = jx.LinearElasticity(
        space,
        plane="stress",
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="fixed")],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([10.0, 0.0]), where="loaded")],
    )
    solution = jx.solve(problem, params={"E": 1e5, "nu": 0.3})
    output = solution.to_vtk("gmsh_rational_plate.vtu", n=4, fields=("von_mises",))
    print(
        f"{space.n_elems} mixed cells; {space.n_dofs} DOFs; CAD distance {mesh.geometry_error:.3g}"
    )
    print(f"ParaView: {output}")
    plot_mesh_field(solution, field="von_mises")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refine", type=int, default=0, help="uniform cell subdivision levels"
    )
    main(parser.parse_args().refine)
