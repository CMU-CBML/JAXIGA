"""IGAPack turbine-blade geometry: a clamped root and transverse tip traction.

Run: python -m jaxiga.examples.blade_3d --refine 0
Requires: pip install -e '.[post]' 'matplotlib>=3.6'
"""

import argparse
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp

import jaxiga as jx
from jaxiga.post.matplotlib import plot_mesh_field


def geometry(refinements=0):
    path = Path(__file__).with_name("geometries") / "igapack_blade.mat"
    patch = jx.read_matlab_nurbs(path, "blade")[0].reverse(0)
    return patch.elevate(2).refine(refinements).with_labels(v0="fixed", v1="loaded")


def main(refinements=0, solver="cg", plot=True):
    start = perf_counter()
    # The leading/trailing faces collapse to edges. Tie their repeated controls
    # so the displacement has a unique value on each physical edge.
    space = jx.FunctionSpace(geometry(refinements), vec=3, collapse_degenerate=True)
    print(
        f"Blade: {space.n_elems} elements, {space.n_dofs} DOFs; solving with {solver}...",
        flush=True,
    )
    problem = jx.LinearElasticity(
        space,
        dirichlet=[jx.DirichletBC([0, 0, 0], where="fixed")],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([0, -0.001, 0]), where="loaded")],
    )
    solution = jx.solve(
        problem,
        params={"E": 1e5, "nu": 0.3},
        chunk=32,
        linear=jx.LinearOptions(method=solver, tol=1e-6, maxiter=10000),
    )
    print(
        f"Setup/solve: {perf_counter() - start:.1f}s; relative residual {float(solution.stats['relative_residual']):.2e}",
        flush=True,
    )
    if not bool(solution.stats["converged"]):
        raise RuntimeError("Solver did not converge; try --solver scipy")
    # Interior averages avoid singular endpoint derivatives at collapsed edges.
    output = solution.to_vtk("blade_3d.vtu", n=3, cell_fields=("von_mises",))
    print(f"ParaView: {output}", flush=True)
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plot_mesh_field(solution, "von_mises", field_location="cell")
        fig.set_size_inches(12, 3.5)
        for ax in axes:
            ax.view_init(elev=12, azim=-70, roll=90)
            ax.set_axis_off()
        plt.show()
    return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine", type=int, default=0)
    parser.add_argument("--solver", choices=("cg", "scipy"), default="cg")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    main(args.refine, args.solver, not args.no_plot)
