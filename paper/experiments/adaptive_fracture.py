"""Compare crack-following refinement with meshes laid out in advance.

Run from the repository root::

    python -m paper.experiments.adaptive_fracture

See paper/README.md for outputs, caching and the suggested reading order.
"""

import os

import matplotlib
import numpy as np

import jaxiga as jx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import FIGURES, RESULTS, figure_data, run_example
from ._fracture_common import (
    _crack_panel,
    _crack_panel_data,
    _fracture_case,
    _run_fracture,
    _summary_only,
)


def _adaptive_fracture_compute():
    """Crack-following adaptivity, against meshes laid out in advance."""
    from jaxiga.solvers.phase_field import Material

    tension = _fracture_case(
        "tension", 0.0125,
        [jx.DirichletBC([0.0, 0.0], where="bottom"),
         jx.DirichletBC([0.0, 1.0], where="top")],
        ("top", 1), (0, 1), ((0.0, 0.0), (1.0, 1.0)), (1, 1),
        Material(E=210e3, nu=0.3, plane="strain"), [(45, 1e-4), (1155, 1e-6)])
    tension["Gc"] = 2.7

    shear = _fracture_case(
        "shear", 0.015,
        [jx.DirichletBC([0.0, 0.0], where="bottom"),
         jx.DirichletBC(1.0, where="top", component=0),
         jx.DirichletBC(0.0, where="top", component=1),
         jx.DirichletBC(0.0, where="left", component=1),
         jx.DirichletBC(0.0, where="right", component=1)],
        ("top", 0), (0, 1), ((0.0, 0.0), (1.0, 1.0)), (1, 1),
        Material(E=210e3, nu=0.3, plane="strain"), [(80, 1e-4), (620, 1e-5)])
    shear["Gc"] = 2.7

    runs = {}
    # Tension: the crack path is known, so a band can be laid out by hand.
    runs["tension_band"] = _run_fracture(
        tension, 20, 3, band=2.0, adaptive=False, label="tension, hand-laid band")
    runs["tension_adaptive"] = _run_fracture(
        tension, 20, 3, adaptive=True, dilate=1.0, label="tension, adaptive",
        capture_final=True)
    # Shear: it cannot, so the honest reference is a uniform mesh at the same h.
    runs["shear_uniform"] = _run_fracture(
        shear, 160, 0, adaptive=False, label="shear, uniform 160^2")
    runs["shear_adaptive"] = _run_fracture(
        shear, 20, 3, adaptive=True, dilate=1.0, label="shear, adaptive",
        capture_final=True)

    results = {k: _summary_only(v[0]) for k, v in runs.items()}
    for pair, ref, adp in [("tension", "tension_band", "tension_adaptive"),
                           ("shear", "shear_uniform", "shear_adaptive")]:
        r, a = runs[ref][0], runs[adp][0]
        results[f"{pair}_speedup"] = r["seconds"] / a["seconds"]
        results[f"{pair}_element_ratio"] = r["elems_end"] / a["elems_end"]

    record = {"results": results,
              "curves": {k: {"u": v[1].tolist(), "f": v[2].tolist(),
                             "elems": np.asarray(v[3]).tolist()}
                         for k, v in runs.items()}}
    arrays = {}
    for key in ("tension_adaptive", "shear_adaptive"):
        info = runs[key][0]
        for name, value in _crack_panel_data(
                info["final_space_d"], info["final_sol_d"]).items():
            arrays[f"{key}__{name}"] = value
    return record, arrays


def adaptive_fracture():
    """Crack-following adaptivity, against meshes laid out in advance."""
    print("[7] Adaptive phase-field fracture (2D)")
    record, arrays = figure_data("adaptive_fracture", _adaptive_fracture_compute)
    RESULTS["adaptive_fracture"] = record["results"]
    _adaptive_fracture_figure(record, arrays)

def _adaptive_fracture_figure(record, arrays):
    """Adaptive against pre-laid meshes, the element count each carries, and
    the final damage field with the mesh that resolved it."""
    info_of = {k: dict(v, label=v["label"].replace("160^2", r"$160^2$"))
               if isinstance(v, dict) and "label" in v else v
               for k, v in record["results"].items()}
    curve_of = record["curves"]

    def curve(key):
        c = curve_of[key]
        return (info_of[key], np.asarray(c["u"]), np.asarray(c["f"]),
                np.asarray(c["elems"]))

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.6))
    panels = [
        (0, "tension_band", "tension_adaptive", "tab:blue",
         "tension: the band can be laid out by hand"),
        (1, "shear_uniform", "shear_adaptive", "tab:red",
         "shear: it cannot"),
    ]
    for i, ref, adp, colour, title in panels:
        ax = axes[0, i]
        (ri, ur, fr, _), (ai, ua, fa, _) = curve(ref), curve(adp)
        ax.plot(ur * 1e3, fr, "-", lw=3.2, color="0.78", solid_capstyle="round",
                label=f"{ri['label']} ({ri['elems_end']:,} el., {ri['seconds']:,.0f} s)")
        ax.plot(ua * 1e3, fa, "--", lw=1.1, color=colour,
                label=f"adaptive ({ai['elems_start']:,}$\\to${ai['elems_end']:,} el., "
                      f"{ai['seconds']:.0f} s)")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(r"prescribed displacement [$\mu$m]")
        ax.set_ylabel("reaction [N]")
        ax.grid(alpha=0.3)
        # below the axes, so that the box cannot cover the curves
        ax.legend(fontsize=8, loc="upper center", frameon=False,
                  bbox_to_anchor=(0.5, -0.17))
        # Above the rising branch, the one region neither response crosses:
        # the shear curve drops almost vertically through the lower right.
        inset = ax.inset_axes([0.13, 0.66, 0.29, 0.25])
        n = min(len(fr), len(fa))
        inset.plot(ua[:n] * 1e3, np.abs(fa[:n] - fr[:n]) / fr.max() * 100,
                   lw=0.8, color=colour)
        inset.set_title("difference [% of peak]", fontsize=7.5, pad=2)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.3)

    for key, colour, style in [("tension_adaptive", "tab:blue", "-"),
                               ("shear_adaptive", "tab:red", "-")]:
        info, _, _, elems = curve(key)
        axes[0, 2].plot(np.arange(len(elems)), elems, style, lw=1.3, color=colour,
                        label=info["label"])
    for key, colour, style in [("tension_band", "tab:blue", "--"),
                               ("shear_uniform", "tab:red", ":")]:
        axes[0, 2].axhline(info_of[key]["elems_end"], color=colour, ls=style, lw=1.1,
                           alpha=0.7, label=info_of[key]["label"])
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlabel("load increment")
    axes[0, 2].set_ylabel("elements")
    axes[0, 2].set_title("elements the solver carries", fontsize=9.5)
    axes[0, 2].grid(alpha=0.3, which="both")
    axes[0, 2].legend(fontsize=8, ncol=2, loc="upper center", frameon=False,
                      bbox_to_anchor=(0.5, -0.17), columnspacing=1.0,
                      handlelength=1.6)

    # Bottom row: the crack the adaptive run actually found, and the mesh that
    # followed it there. The third slot is unused -- the top row has three
    # cases, the bottom only two adaptive runs to show.
    sc = None
    for j, key, colour in [(0, "tension_adaptive", "tab:blue"),
                           (1, "shear_adaptive", "tab:red")]:
        info = info_of[key]
        panel = {n: arrays[f"{key}__{n}"]
                 for n in ("x", "dmg", "mesh", "n_elems", "n_samp")}
        sc = _crack_panel(axes[1, j], panel)
        axes[1, j].set_title(
            f"{info['label']}: damage + mesh ({info['elems_end']:,} el.)",
            fontsize=9.5, color=colour)
        axes[1, j].set_xlabel("$x$")
        axes[1, j].set_ylabel("$y$")
    axes[1, 2].axis("off")
    fig.colorbar(sc, ax=axes[1, 2], fraction=0.6, label=r"damage $\phi$")

    for ax in axes.flat:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "adaptive_fracture.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/adaptive_fracture.pdf")


if __name__ == "__main__":
    run_example("adaptive_fracture", adaptive_fracture)
