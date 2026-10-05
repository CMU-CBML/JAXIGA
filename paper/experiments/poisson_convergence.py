"""Solve manufactured Poisson problems and measure convergence in 1D, 2D and 3D.

Run from the repository root::

    python -m paper.experiments.poisson_convergence

See paper/README.md for outputs, caching and the suggested reading order.
"""

import os
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

import jaxiga as jx
from jaxiga.space.evaluation import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import ALL, FIGURES, RESULTS, rate, run_example, write_table
from ._plotting import slice_pointset


# Manufactured solution u = prod_i sin(2 pi x_i). The two full periods per
# direction put several elements per wavelength only after a few refinements,
# so the coarse levels are genuinely pre-asymptotic and the rate is measured on
# a resolved mesh rather than read off a curve that was never in doubt.
KWAVE = 2.0


def poisson_problem(dim, deg, refine):
    k = KWAVE * jnp.pi
    if dim == 1:
        patch = jx.primitives.interval(0.0, 1.0)
        exact = lambda x: jnp.sin(k * x[0])
    elif dim == 2:
        patch = jx.primitives.rectangle(0, 0, 1, 1)
        exact = lambda x: jnp.sin(k * x[0]) * jnp.sin(k * x[1])
    else:
        patch = jx.primitives.cuboid([0, 0, 0], [1, 1, 1])
        exact = lambda x: jnp.sin(k * x[0]) * jnp.sin(k * x[1]) * jnp.sin(k * x[2])
    src = lambda x, p: dim * k**2 * exact(x)

    V = jx.FunctionSpace(patch.elevate(deg).refine(refine))
    return jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=src), exact


def poisson_solution_figure():
    # 3D solution, shown as two plane sections rather than a rendered volume.
    # A 3D axes spends most of its bounding box on empty corners and reads the
    # field through a perspective projection, which is exactly the wrong trade
    # for a quantitative figure; a section is drawn to scale and to size.
    # The section is taken at z = 1/4, where sin(2 pi z) attains its maximum,
    # so nothing of the field is lost to the choice of plane.
    from scipy.interpolate import griddata

    problem, exact = poisson_problem(3, 3, 3)
    sol = jx.solve(problem, params={"a0": 1.0})
    l2 = jx.errornorm(sol, exact, "L2")

    n_slice, n_grid, cut = 7, 220, 0.25
    ps, others = slice_pointset(problem.space, 2, cut, n_slice)
    b = evaluate(problem.space, ps)
    x = np.asarray(b.x).reshape(-1, 3)
    uh = np.asarray(sol.at(basis=b)).reshape(-1)
    ue = np.asarray(jax.vmap(exact)(jnp.asarray(x)))

    g = np.linspace(0.0, 1.0, n_grid)
    X, Y = np.meshgrid(g, g, indexing="ij")
    gu = griddata(x[:, others], uh, (X, Y), method="linear")
    ge = griddata(x[:, others], np.abs(uh - ue), (X, Y), method="linear")

    umax = np.nanmax(np.abs(gu))
    # Carry the error's order of magnitude in the panel title rather than as a
    # colorbar offset: matplotlib parks that offset directly under the title,
    # where the two collide at this figure width.
    expo = int(np.floor(np.log10(np.nanmax(ge))))
    ge, emax = ge / 10.0**expo, np.nanmax(ge) / 10.0**expo
    n_e_dir = int(round(problem.space.n_elems ** (1 / 3)))
    # Sized to the 390 pt text width so that \includegraphics[width=\textwidth]
    # reproduces it at 1:1 and the tick labels keep the size they were set at.
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.35))
    fig.subplots_adjust(wspace=0.32)
    panels = [
        (gu, r"computed solution $u_h$", "RdBu_r",
         matplotlib.colors.Normalize(-umax, umax)),
        (ge, rf"pointwise error $|u_h - u| \;/\; 10^{{{expo}}}$", "magma",
         matplotlib.colors.Normalize(0.0, emax)),
    ]
    for ax, (field, label, cmap, norm) in zip(axes, panels):
        im = ax.pcolormesh(X, Y, field, cmap=cmap, norm=norm, shading="gouraud",
                           rasterized=True)
        # element boundaries, so the error lattice can be read against the mesh
        for e in np.linspace(0.0, 1.0, n_e_dir + 1)[1:-1]:
            ax.axvline(e, color="0.5", lw=0.3, alpha=0.6)
            ax.axhline(e, color="0.5", lw=0.3, alpha=0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_title(label, fontsize=9, pad=4)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    axes[0].set_ylabel("$y$")
    fig.savefig(os.path.join(FIGURES, "poisson_solution.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/poisson_solution.pdf  "
          f"(section z={cut}, p=3, 3 refinements, {problem.space.n_dofs} dofs, "
          f"L2 err {l2:.3e})")
    RESULTS["poisson_3d_figure"] = {
        "dofs": problem.space.n_dofs, "elems": problem.space.n_elems,
        "elems_per_dir": n_e_dir, "cut": cut, "l2": float(l2),
        "max_err": float(emax * 10.0**expo), "umax": float(umax),
    }


def poisson_convergence():
    print("[1] Poisson convergence")
    grid = {1: (2, 3, 4, 5, 6, 7), 2: (1, 2, 3, 4, 5, 6), 3: (1, 2, 3, 4)}
    data = {}
    rows = []

    for dim in (1, 2, 3):
        for deg in (2, 3, 4):
            errs, dofs, times = [], [], []
            for r in grid[dim]:
                problem, exact = poisson_problem(dim, deg, r)
                t0 = time.perf_counter()
                sol = jx.solve(problem, params={"a0": 1.0})
                sol.u.block_until_ready()
                times.append(time.perf_counter() - t0)
                errs.append(jx.errornorm(sol, exact, "L2"))
                dofs.append(problem.space.n_dofs)
            data[(dim, deg)] = (dofs, errs, rate(errs), times)
            print(f"    dim={dim} p={deg}: rates {[f'{x:.2f}' for x in rate(errs)[1:]]}")

    # Compact rate summary. The full error-vs-refinement data is shown in the
    # figure; tabulating it as well would be redundant and long, so only the
    # observed asymptotic rates are given.
    lines = [
        r"\begin{tabular}{c c c c c}", r"\hline",
        r"& \multicolumn{3}{c}{observed rate} & \\",
        r"$p$ & $d=1$ & $d=2$ & $d=3$ & theory \\", r"\hline",
    ]
    for deg in (2, 3, 4):
        cells = [f"{data[(dim, deg)][2][-1]:.2f}" for dim in (1, 2, 3)]
        lines.append(f"{deg} & " + " & ".join(cells) + f" & {deg + 1} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write_table("poisson_convergence", "\n".join(lines))

    # Figure, sized to the 390 pt text width so that it is included at 1:1 and
    # the tick labels reach print at the size they were set at.
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.1))
    for ax, dim in zip(axes, (1, 2, 3)):
        for deg, mark in zip((2, 3, 4), ("o", "s", "^")):
            dofs, errs, _, _ = data[(dim, deg)]
            h = np.array(dofs, float) ** (-1.0 / dim)
            ax.loglog(h, errs, marker=mark, ms=3.2, lw=1.0, label=f"$p={deg}$")
            ref = errs[-1] * (h / h[-1]) ** (deg + 1)
            ax.loglog(h, ref, "k:", lw=0.7)
        ax.set_title(f"{dim}D", fontsize=9, pad=3)
        ax.grid(True, which="major", alpha=0.3)
        # log minor-tick labels collide on these narrow panels
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=4))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=100, numticks=6))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel(r"relative $L^2$ error")
    axes[1].set_xlabel(r"$h \sim \mathrm{dofs}^{-1/d}$")
    # One legend for all three panels, below them, so that it applies to the
    # 2D and 3D panels as well and cannot cover a curve.
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], color="k", ls=":", lw=0.7))
    labels.append(r"rate $h^{p+1}$")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08), handlelength=1.8, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.34)
    fig.savefig(os.path.join(FIGURES, "poisson_convergence.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/poisson_convergence.pdf")
    poisson_solution_figure()

    RESULTS["poisson"] = {f"{d}-{p}": data[(d, p)][2][-1] for d in (1, 2, 3) for p in (2, 3, 4)}


if __name__ == "__main__":
    run_example("poisson_convergence", poisson_convergence)
