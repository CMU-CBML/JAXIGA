"""Resolve a corner singularity using local refinement on an L-shaped domain.

Run from the repository root::

    python -m paper.experiments.adaptive_lshape

See paper/README.md for outputs, caching and the suggested reading order.
"""

import os

import jax.numpy as jnp
import matplotlib
import numpy as np

import jaxiga as jx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import FIGURES, RESULTS, run_example, write_table


LSHAPE_ALPHA = 2.0 / 3.0


def _lshape_theta(x, y):
    t = jnp.arctan2(y, x)
    return jnp.where(t < -1e-12, t + 2 * jnp.pi, t)


def lshape_exact(x):
    r = jnp.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-300
    return r**LSHAPE_ALPHA * jnp.sin(LSHAPE_ALPHA * _lshape_theta(x[0], x[1]))


def lshape_exact_grad(x):
    r = jnp.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-300
    t = _lshape_theta(x[0], x[1])
    c = LSHAPE_ALPHA * r ** (LSHAPE_ALPHA - 1.0)
    return jnp.array(
        [c * jnp.sin((LSHAPE_ALPHA - 1.0) * t), c * jnp.cos((LSHAPE_ALPHA - 1.0) * t)]
    )


def lshape_patches(deg, refine):
    quadrants = [
        jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0),
        jx.primitives.rectangle(-1.0, 0.0, 0.0, 1.0),
        jx.primitives.rectangle(-1.0, -1.0, 0.0, 0.0),
    ]
    return [q.elevate(deg).refine(refine) for q in quadrants]


def lshape_problem(space):
    return jx.Poisson(
        space,
        dirichlet=[
            jx.DirichletBC(lambda x, p: lshape_exact(x), where=lambda X: np.ones(len(X), bool))
        ],
    )


def _tail_slope(n_dofs, errs, window=8):
    n = np.asarray(n_dofs, dtype=float)
    e = np.asarray(errs, dtype=float)
    k = max(0, len(n) - window)
    return float(np.polyfit(np.log(n[k:]), np.log(e[k:]), 1)[0])


def adaptive_lshape():
    print("[8] adaptive refinement, L-shaped domain")
    params = {"a0": 1.0}
    runs = {}

    for deg, start, n_cycles, n_uniform in ((2, 1, 16, 5), (3, 3, 12, 4)):
        V0 = jx.FunctionSpace(lshape_patches(deg, start))
        _, V, hist = jx.adapt(
            lshape_problem, params, space=V0, n_cycles=n_cycles, frac=0.5,
            exact=lshape_exact, exact_grad=lshape_exact_grad,
        )
        uni = []
        for r in range(start, start + n_uniform):
            Vu = jx.FunctionSpace(lshape_patches(deg, r))
            sol = jx.solve(lshape_problem(Vu), params=params)
            uni.append((Vu.n_dofs, jx.errornorm(sol, lshape_exact, "H1",
                                                exact_grad=lshape_exact_grad)))
        runs[deg] = {
            "adaptive": [(h["n_dofs"], h["H1"]) for h in hist],
            "uniform": uni,
            "levels": V.hierarchy.n_levels,
            "n_local": V.n_local,
            "n_bernstein": V.n_bernstein,
            "space": V,
        }
        s_a = _tail_slope(*zip(*runs[deg]["adaptive"]))
        s_u = _tail_slope(*zip(*uni), window=3)
        runs[deg]["slope_adaptive"] = s_a
        runs[deg]["slope_uniform"] = s_u
        print(f"    p={deg}: adaptive slope {s_a:.3f} (optimal {-deg/2:.2f}), "
              f"uniform {s_u:.3f}, {V.hierarchy.n_levels} levels, "
              f"n_local_max {V.n_local}/{V.n_bernstein}")

    # -- figure: error vs dofs, plus the adapted mesh --------------------
    fig = plt.figure(figsize=(6.6, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    colours = {2: "tab:blue", 3: "tab:red"}
    for deg, run in runs.items():
        na, ea = zip(*run["adaptive"])
        nu, eu = zip(*run["uniform"])
        ax.loglog(na, ea, "o-", ms=3, color=colours[deg], label=f"adaptive, $p={deg}$")
        ax.loglog(nu, eu, "s--", ms=3, mfc="none", color=colours[deg],
                  label=f"uniform, $p={deg}$")
    n_ref = np.array([2e2, 4e3])
    ax.loglog(n_ref, 3.0e-2 * (n_ref / n_ref[0]) ** (-1.0), ":", c="0.4", lw=1)
    ax.loglog(n_ref, 6.0e-3 * (n_ref / n_ref[0]) ** (-1.5), ":", c="0.4", lw=1)
    ax.loglog(n_ref, 9.0e-2 * (n_ref / n_ref[0]) ** (-1 / 3), ":", c="0.4", lw=1)
    ax.text(n_ref[1], 3.0e-2 * (n_ref[1] / n_ref[0]) ** (-1.0), r"$N^{-1}$",
            fontsize=7, va="bottom", ha="right", color="0.4")
    ax.text(n_ref[1], 6.0e-3 * (n_ref[1] / n_ref[0]) ** (-1.5), r"$N^{-3/2}$",
            fontsize=7, va="bottom", ha="right", color="0.4")
    ax.text(n_ref[1], 9.0e-2 * (n_ref[1] / n_ref[0]) ** (-1 / 3), r"$N^{-1/3}$",
            fontsize=7, va="bottom", ha="right", color="0.4")
    ax.set_xlabel("degrees of freedom $N$")
    ax.set_ylabel(r"$H^1$ error")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.24))
    ax.grid(True, which="both", alpha=0.25, lw=0.4)

    # the adapted mesh: element boxes mapped to physical space
    ax2 = fig.add_subplot(gs[0, 1])
    V = runs[2]["space"]
    boxes = V.elem_vertex
    patch_of = np.asarray(V.elem_patch)
    origin = {0: (0.0, 0.0), 1: (-1.0, 0.0), 2: (-1.0, -1.0)}
    for e in range(V.n_elems):
        ox, oy = origin[int(patch_of[e])]
        x0, y0, x1, y1 = boxes[e, 0] + ox, boxes[e, 1] + oy, boxes[e, 2] + ox, boxes[e, 3] + oy
        ax2.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    lw=0.25, ec="0.25"))
    ax2.set_xlim(-1.02, 1.02)
    ax2.set_ylim(-1.02, 1.02)
    ax2.set_aspect("equal")
    ax2.set_xticks([-1, 0, 1])
    ax2.set_yticks([-1, 0, 1])
    ax2.tick_params(labelsize=7)
    ax2.set_title(f"$p=2$, {V.n_elems:,} elements, {runs[2]['levels']} levels", fontsize=7)
    fig.savefig(os.path.join(FIGURES, "lshape_adaptive.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/lshape_adaptive.pdf")

    # -- table ------------------------------------------------------------
    lines = [
        r"\begin{tabular}{c c r l r l c c}", r"\hline",
        r"& \multicolumn{3}{c}{adaptive} & \multicolumn{2}{c}{uniform} & & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"$p$ & dofs & $H^1$ error & rate & dofs & $H^1$ error & optimal & "
        r"$n_{\mathrm{loc}}^{\max}$ \\",
        r"\hline",
    ]
    for deg, run in runs.items():
        na, ea = run["adaptive"][-1]
        nu, eu = run["uniform"][-1]
        lines.append(
            f"{deg} & {na} & {ea:.3e} & ${run['slope_adaptive']:.3f}$ & "
            f"{nu} & {eu:.3e} & ${-deg/2:.1f}$ & "
            f"{run['n_local']} ({run['n_bernstein']}) \\\\"
        )
    lines += [
        r"\hline",
        r"\multicolumn{8}{l}{\footnotesize Uniform refinement rates: "
        + ", ".join(f"${runs[d]['slope_uniform']:.3f}$ at $p={d}$" for d in sorted(runs))
        + r", against the regularity limit $-1/3$.} \\",
        r"\hline", r"\end{tabular}",
    ]
    write_table("lshape_adaptive", "\n".join(lines))

    for deg, run in runs.items():
        RESULTS[f"lshape_p{deg}_slope_adaptive"] = run["slope_adaptive"]
        RESULTS[f"lshape_p{deg}_slope_uniform"] = run["slope_uniform"]
        RESULTS[f"lshape_p{deg}_levels"] = run["levels"]
        RESULTS[f"lshape_p{deg}_n_local"] = run["n_local"]
        RESULTS[f"lshape_p{deg}_gain"] = run["uniform"][-1][1] / run["adaptive"][-1][1]


if __name__ == "__main__":
    run_example("adaptive_lshape", adaptive_lshape)
