"""Exact cylindrical walls, mixed hex/tet elasticity, Matplotlib and ParaView.

Run: python -m jaxiga.examples.gmsh_cylinder_3d --refine 1
Requires: pip install -e '.[gmsh,post]' matplotlib
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt

import jaxiga as jx
from jaxiga.post.matplotlib import plot_mesh_field


def main(refinements=0, solver="cg", plot=True):
    start = perf_counter()
    print("Loading CAD and building the mixed space...", flush=True)
    geometry = Path(__file__).with_name("geometries") / "quarter_cylinder.geo"
    mesh = jx.read_gmsh(geometry, geometry_order=2, rational_geometry=True)
    # Keep the lower hex; restrict the upper hex to six rational tetrahedra.
    patches = [mesh[0], *jx.split_to_simplices(mesh[1])]
    patches = jx.refine_cells(patches, n=refinements)
    space = jx.FunctionSpace(patches, vec=3)
    print(
        f"{space.n_elems} cells; {space.n_dofs} DOFs; setup {perf_counter() - start:.1f}s",
        flush=True,
    )
    problem = jx.LinearElasticity(
        space,
        dirichlet=[
            jx.DirichletBC(0.0, where="sym_x", component=0),
            jx.DirichletBC(0.0, where="sym_y", component=1),
            jx.DirichletBC(0.0, where="ends", component=2),
        ],
        neumann=[jx.Neumann(lambda x, n, p: -10.0 * n, where="inner")],
    )
    print(f"Assembling and solving with {solver}...", flush=True)
    start = perf_counter()
    solution = jx.solve(
        problem,
        params={"E": 1e5, "nu": 0.3},
        chunk=16,
        linear=jx.LinearOptions(
            method=solver, preconditioner="jacobi", tol=1e-8, maxiter=5000
        ),
    )
    residual = float(solution.stats["relative_residual"])
    print(
        f"Solve {perf_counter() - start:.1f}s; relative residual {residual:.2e}",
        flush=True,
    )
    if not bool(solution.stats["converged"]):
        raise RuntimeError("The linear solver did not converge; no results exported")
    print("Exporting ParaView results...", flush=True)
    output = solution.to_vtk("gmsh_cylinder_3d.vtu", n=5, fields=("von_mises",))
    print(
        f"{space.n_elems} mixed cells; {space.n_dofs} DOFs; CAD distance {mesh.geometry_error:.3g}"
    )
    print(f"ParaView: {output}")
    if plot:
        print("Plotting mesh and stress...", flush=True)
        plot_mesh_field(solution, field="von_mises")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refine", type=int, default=0, help="uniform cell subdivision levels"
    )
    parser.add_argument(
        "--solver",
        choices=("cg", "scipy"),
        default="cg",
        help="Jacobi-preconditioned CG (default) or direct sparse LU",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="write VTK without opening a plot"
    )
    args = parser.parse_args()
    main(args.refine, solver=args.solver, plot=not args.no_plot)
