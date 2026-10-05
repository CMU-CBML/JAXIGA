"""Compare sparse direct and conjugate-gradient solves on a 3D elasticity system.

Run from the repository root::

    python -m paper.experiments.linear_solver_3d

See paper/README.md for outputs, caching and the suggested reading order.
"""

import time

import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.methods._common import PointField, build_context
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

from ._common import RESULTS, run_example, write_table


def linear_solver_3d():
    """Direct factorisation against Jacobi-CG on the 3D displacement system."""
    print("[9] Linear solver in three dimensions")
    import dataclasses

    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from jaxiga.forms.library import PhaseFieldDisplacement, edge_crack_distance
    from jaxiga.methods.galerkin import assemble_csr_values, rhs
    from jaxiga.solvers.phase_field import seed_refine

    ELL = 1 / 16
    crack = edge_crack_distance((0.5, 0.5), axes=(0, 2))
    rows = []
    for base, levels, extra in [(8, 2, 0), (8, 2, 2), (8, 3, 0)]:
        kx = np.linspace(0, 1, base + 1)[1:-1]
        ky = np.linspace(0, 1, 3)[1:-1]
        patch = jx.primitives.cuboid((0, 0, 0), (1, 0.2, 1)).elevate(2).insert_knots(kx, ky, kx)
        V = seed_refine(jx.refine_elements(jx.FunctionSpace(patch, vec=1), []),
                        crack, ELL, levels)
        if extra:
            V = jx.refine_elements(V, np.arange(0, V.n_elems, extra))
        Vu = dataclasses.replace(V, vec=3)
        disp = PhaseFieldDisplacement(
            Vu, plane="strain",
            dirichlet=[jx.DirichletBC(0.0, where="back", component=2),
                       jx.DirichletBC(1.0, where="front", component=2),
                       jx.DirichletBC(0.0, where="bottom", component=1),
                       jx.DirichletBC(0.0, where="left", component=0)])
        basis = evaluate(Vu, P.gauss(Vu))
        zero = jnp.zeros((basis.n_elems, basis.n_q))
        params = {"E": 20.8e3, "nu": 0.3, "k": 1e-8, "phi": PointField(zero)}
        ctx = build_context(disp, params)
        pattern = ctx.assembly
        b = np.asarray(rhs(disp, params, ctx))
        data = np.asarray(assemble_csr_values(ctx, params))
        A = sp.csr_matrix((data, pattern.column_indices, pattern.indptr),
                          shape=pattern.shape).tocsc()

        t = time.perf_counter()
        lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
        t_lu = time.perf_counter() - t
        fill = (lu.L.nnz + lu.U.nnz) / A.nnz

        M = spla.LinearOperator(A.shape, matvec=lambda v: v / A.diagonal())
        count = [0]
        t = time.perf_counter()
        x, _ = spla.cg(A, b, rtol=1e-9, M=M, maxiter=20000,
                       callback=lambda _: count.__setitem__(0, count[0] + 1))
        t_cg = time.perf_counter() - t
        err = np.linalg.norm(x - lu.solve(b)) / max(np.linalg.norm(lu.solve(b)), 1e-300)
        rows.append((A.shape[0], t_lu, fill, t_cg, count[0], err))
        print(f"    {A.shape[0]:6d} dofs: splu {t_lu:6.2f}s (fill {fill:4.1f}x), "
              f"CG {t_cg:5.2f}s ({count[0]} its), agreement {err:.1e}")

    lines = [r"\begin{tabular}{r r r r r}", r"\hline",
             r"Free dofs & \texttt{splu} & Fill-in & Jacobi-CG & Iterations \\", r"\hline"]
    for n, t_lu, fill, t_cg, its, _ in rows:
        lines.append(f"${n}$ & ${t_lu:.2f}$~s & ${fill:.1f}\\times$ & "
                     f"${t_cg:.2f}$~s & ${its}$ \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write_table("cg_vs_direct", "\n".join(lines))
    RESULTS["linear_solver_3d"] = [
        {"dofs": int(n), "splu_seconds": t_lu, "fill": fill,
         "cg_seconds": t_cg, "cg_iterations": int(its), "agreement": err}
        for n, t_lu, fill, t_cg, its, err in rows]


if __name__ == "__main__":
    run_example("linear_solver_3d", linear_solver_3d)
