"""Solve simply supported and clamped fourth-order plate problems with smooth splines.

Run from the repository root::

    python -m paper.experiments.kirchhoff_plate

See paper/README.md for outputs, caching and the suggested reading order.
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx

from ._common import ALL, RESULTS, rate, run_example, write_table


PLATE_E, PLATE_NU, PLATE_T, PLATE_Q = 210e3, 0.3, 0.01, 1.0
PLATE_D = PLATE_E * PLATE_T**3 / (12 * (1 - PLATE_NU**2))
PLATE_PARAMS = {"E": PLATE_E, "nu": PLATE_NU, "t": PLATE_T}


def kirchhoff_plate():
    print("[9] Kirchhoff-Love plate")
    navier = lambda x: PLATE_Q * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) / (
        4 * jnp.pi**4 * PLATE_D
    )
    rows = []
    for deg in (2, 3):
        errs = []
        for r in (3, 4, 5):
            V = jx.FunctionSpace(
                jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0).elevate(deg).refine(r)
            )
            problem = jx.KirchhoffPlate(
                V,
                dirichlet=[jx.DirichletBC(0.0, where=ALL)],
                source=lambda x, p: PLATE_Q * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
            )
            sol = jx.solve(problem, params=PLATE_PARAMS)
            errs.append(jx.errornorm(sol, navier, "L2"))
            rows.append((deg, 2**r, V.n_dofs, errs[-1], rate(errs)[-1],
                         min(deg + 1, 2 * deg - 2)))
        print(f"    p={deg}: rates {[f'{v:.2f}' for v in rate(errs)[1:]]}, "
              f"theory {min(deg + 1, 2 * deg - 2)}")
        RESULTS[f"plate_p{deg}_rate"] = float(rate(errs)[-1])

    # clamped plate against Timoshenko's tabulated centre deflection
    clamped = []
    for r in (3, 4, 5):
        V = jx.FunctionSpace(jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0).elevate(3).refine(r))
        problem = jx.KirchhoffPlate(
            V,
            dirichlet=[jx.ClampedBC(where=s) for s in ("left", "right", "bottom", "top")],
            source=lambda x, p: PLATE_Q,
        )
        sol = jx.solve(problem, params=PLATE_PARAMS)
        clamped.append((V.n_dofs, float(sol.probe(np.array([[0.5, 0.5]]))[0, 0])))
    reference = 0.0012653 * PLATE_Q / PLATE_D
    print(f"    clamped w_max {clamped[-1][1]:.6e} vs Timoshenko {reference:.6e} "
          f"(ratio {clamped[-1][1] / reference:.5f})")
    RESULTS["plate_clamped_ratio"] = clamped[-1][1] / reference

    lines = [
        r"\begin{tabular}{c c r l c}", r"\hline",
        r"$p$ & mesh & dofs & rel. $L^2$ error & rate \\", r"\hline",
    ]
    for deg, n, dofs, err, rt, theory in rows:
        r_txt = "---" if np.isnan(rt) else f"{rt:.2f}"
        lines.append(f"{deg} & ${n}\\times{n}$ & {dofs} & {err:.4e} & {r_txt} \\\\")
    lines += [
        r"\hline",
        r"\multicolumn{5}{l}{\footnotesize Theoretical rate $\min(p+1,\,2p-2)$: "
        r"2 for $p=2$, 4 for $p=3$.} \\",
        r"\multicolumn{5}{l}{\footnotesize Clamped plate, $32\times32$, $p=3$: "
        rf"$w_{{\max}} = {clamped[-1][1]:.6f}$ vs Timoshenko ${reference:.6f}$ "
        rf"(ratio ${clamped[-1][1] / reference:.5f}$).}} \\",
        r"\hline", r"\end{tabular}",
    ]
    write_table("kirchhoff_plate", "\n".join(lines))


if __name__ == "__main__":
    run_example("kirchhoff_plate", kirchhoff_plate)
