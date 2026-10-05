"""Solve the same Poisson problem with Galerkin, energy and collocation methods.

Run from the repository root::

    python -m paper.experiments.method_comparison

See paper/README.md for outputs, caching and the suggested reading order.
"""

import jax.numpy as jnp

import jaxiga as jx

from ._common import RESULTS, run_example, write_table
from .poisson_convergence import poisson_problem


def method_comparison():
    print("[2] Method comparison")
    rows = []
    for deg in (2, 3, 4):
        for r in (3, 4, 5):
            problem, exact = poisson_problem(2, deg, r)
            g = jx.solve(problem, params={"a0": 1.0})
            e = jx.solve(problem, params={"a0": 1.0}, method="energy", max_steps=4000)
            c = jx.solve(problem, params={"a0": 1.0}, method="collocation")
            scale = float(jnp.max(jnp.abs(g.u)))
            rows.append((
                deg, r, problem.space.n_dofs,
                jx.errornorm(g, exact, "L2"),
                jx.errornorm(e, exact, "L2"),
                jx.errornorm(c, exact, "L2"),
                float(jnp.max(jnp.abs(e.u - g.u))) / scale,
            ))
            print(f"    p={deg} r={r}: gal {rows[-1][3]:.2e} em {rows[-1][4]:.2e} "
                  f"col {rows[-1][5]:.2e} agree {rows[-1][6]:.1e}")

    lines = [
        r"\begin{tabular}{c c r l l l l}", r"\hline",
        r"$p$ & ref. & dofs & Galerkin & energy min. & collocation & "
        r"$\|u_{\mathrm{EM}}-u_{\mathrm{G}}\|_\infty/\|u\|_\infty$ \\",
        r"\hline",
    ]
    for deg, r, n, eg, ee, ec, agree in rows:
        lines.append(f"{deg} & {r} & {n} & {eg:.3e} & {ee:.3e} & {ec:.3e} & {agree:.1e} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write_table("method_comparison", "\n".join(lines))
    RESULTS["max_em_galerkin_disagreement"] = max(r[6] for r in rows)


if __name__ == "__main__":
    run_example("method_comparison", method_comparison)
