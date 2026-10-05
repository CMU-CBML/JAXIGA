"""IGAPack's 25-patch connecting rod, with optional mixed hex/tet coupling.

Run: python -m jaxiga.examples.connecting_rod_3d --refine 1
Requires: pip install -e '.[post]' matplotlib
"""

import argparse
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp

import jaxiga as jx
from jaxiga.post.matplotlib import plot_mesh_field


def geometry(refinements=0, mixed=False):
    path = Path(__file__).with_name("geometries") / "igapack_connecting_rod.mat"
    names = [f"solid{i}{s}" for i in range(1, 4) for s in "ABC"] + ["solid4"]
    names += [f"solidT{i}{s}" for i in range(1, 6) for s in "ABC"]
    patches = []
    for name, patch in zip(names, jx.read_matlab_nurbs(path, names)):
        if name == "solid4" or name.startswith("solidT"):
            patch = patch.reverse(2)  # positive volume orientation
        if name.startswith("solid1"):
            patch = patch.with_labels(u1="fixed")
        if name.startswith("solid3"):
            patch = patch.with_labels(u0="fixed")
        if name.startswith("solidT"):
            patch = patch.with_labels(v0="loaded")
        # Keep the shaft's native (2,2,1) geometry for conversion: this needs
        # degree-five tets, avoiding unnecessary degree-six elevation.
        patches.extend(
            jx.split_to_simplices(patch)
            if mixed and name == "solid4"
            else [patch.elevate(2)]
        )
    return (
        jx.refine_cells(patches, refinements)
        if mixed
        else [p.refine(refinements) for p in patches]
    )


def main(
    refinements=0, mixed=False, solver="cg", plot=True, *, maxiter=50000, tol=None
):
    linear = jx.LinearOptions(
        method=solver,
        tol=(1e-6 if mixed else 1e-8) if tol is None else tol,
        maxiter=maxiter,
    )
    start = perf_counter()
    space = jx.FunctionSpace(geometry(refinements, mixed), vec=3)
    print(
        f"Connecting rod: {space.n_elems} elements, {space.n_dofs} DOFs; "
        f"solving with {solver} (tol={linear.tol:g}, maxiter={linear.maxiter})...",
        flush=True,
    )
    # Same support/load regions and material as IGAPack's ConnectingRodMP.m:
    # clamp the big-end cuts and apply downward traction on the small-end bore.
    problem = jx.LinearElasticity(
        space,
        dirichlet=[jx.DirichletBC([0, 0, 0], where="fixed")],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([0, 0, -1.0]), where="loaded")],
    )
    solution = jx.solve(
        problem,
        params={"E": 4e5, "nu": 0.3},
        chunk=16,
        linear=linear,
    )
    print(
        f"Setup/solve: {perf_counter() - start:.1f}s; relative residual {float(solution.stats['relative_residual']):.2e}",
        flush=True,
    )
    if not bool(solution.stats["converged"]):
        raise RuntimeError(
            "Solver did not converge; increase --maxiter, adjust --tol, "
            "or use --solver scipy"
        )
    stem = "connecting_rod_mixed_3d" if mixed else "connecting_rod_3d"
    output = solution.to_vtk(stem + ".vtu", n=3, cell_fields=("von_mises",))
    print(f"ParaView: {output}", flush=True)
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plot_mesh_field(solution, "von_mises", field_location="cell")
        fig.set_size_inches(12, 4.5)
        for ax in axes:
            ax.view_init(elev=25, azim=-65)
            ax.set_axis_off()
        plt.show()
    return solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine", type=int, default=0)
    parser.add_argument(
        "--mixed", action="store_true", help="split the shaft into rational tetrahedra"
    )
    parser.add_argument("--solver", choices=("cg", "scipy"), default="cg")
    parser.add_argument(
        "--maxiter", type=int, default=50000, help="CG iteration limit (default: 50000)"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help="relative residual tolerance (default: 1e-6 mixed, 1e-8 tensor)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    main(
        args.refine,
        args.mixed,
        args.solver,
        not args.no_plot,
        maxiter=args.maxiter,
        tol=args.tol,
    )
