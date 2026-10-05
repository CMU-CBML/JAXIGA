"""Simulate a notched 3D specimen and compare its two-dimensional limits.

Run from the repository root::

    python -m paper.experiments.fracture_3d

See paper/README.md for outputs, caching and the suggested reading order.
"""

import os

import matplotlib
import numpy as np

import jaxiga as jx
from jaxiga.space.evaluation import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import FIGURES, RESULTS, figure_data, run_example
from ._fracture_common import _PLOT_DTYPE, _fracture_case, _run_fracture, _summary_only
from ._plotting import box_slice_lines, slice_pointset


def _fracture_3d_compute():
    """The notched cube, its two-dimensional limits, and the solver comparison."""
    from jaxiga.solvers.phase_field import Material

    def cube(ell):
        c = _fracture_case(
            "cube", ell,
            [jx.DirichletBC(0.0, where="back", component=2),
             jx.DirichletBC(1.0, where="front", component=2),
             jx.DirichletBC(0.0, where="bottom", component=1),
             jx.DirichletBC(0.0, where="left", component=0)],
            ("front", 2), (0, 2), ((0.0, 0.0, 0.0), (1.0, 0.2, 1.0)), (1, 0.25, 1),
            Material(E=20.8e3, nu=0.3, plane="strain"),
            [(12, 5e-4), (20, 1e-4), (600, 2.5e-5)], stop_below=0.05)
        c["Gc"] = 0.5
        return c

    def plate(ell, plane):
        c = _fracture_case(
            f"plate-{plane}", ell,
            [jx.DirichletBC([0.0, 0.0], where="bottom"),
             jx.DirichletBC([0.0, 1.0], where="top")],
            ("top", 1), (0, 1), ((0.0, 0.0), (1.0, 1.0)), (1, 1),
            Material(E=20.8e3, nu=0.3, plane=plane),
            [(12, 5e-4), (20, 1e-4), (600, 2.5e-5)], stop_below=0.05)
        c["Gc"] = 0.5
        return c

    runs = {}
    # The two-dimensional limits, scaled by the thickness, must bracket the cube.
    for plane in ("stress", "strain"):
        runs[f"plate_{plane}"] = _run_fracture(
            plate(1 / 16, plane), 8, 2, dilate=1.0, label=f"2D plane {plane} x 0.2")
    # The mesh/crack panel is taken from the coarser case: it carries the same
    # message -- refinement follows the damage -- at a size that fits
    # comfortably on a 40 GB device, where l = 1/32 is at the edge of it.
    runs["cube_16"] = _run_fracture(cube(1 / 16), 8, 2, dilate=1.0,
                                    label="3D cube, l = 1/16", capture_final=True)
    runs["cube_32"] = _run_fracture(cube(1 / 32), 8, 3, dilate=1.5,
                                    label="3D cube, l = 1/32")

    THICK = 0.2
    for key in ("plate_stress", "plate_strain"):
        info, u, f, elems = runs[key]
        runs[key] = (dict(info, peak_force=info["peak_force"] * THICK), u, f * THICK, elems)

    results = {k: _summary_only(v[0]) for k, v in runs.items()}
    results["peak_ratio_l16_to_l32"] = (
        runs["cube_32"][0]["peak_force"] / runs["cube_16"][0]["peak_force"])
    results["brackets"] = bool(
        runs["plate_stress"][0]["peak_force"]
        < runs["cube_16"][0]["peak_force"]
        < runs["plate_strain"][0]["peak_force"])

    record = {"results": results,
              "curves": {k: {"u": v[1].tolist(), "f": v[2].tolist(),
                             "elems": np.asarray(v[3]).tolist()}
                         for k, v in runs.items()},
              "box": [list(b) for b in runs["cube_16"][0]["box"]]}
    info = runs["cube_16"][0]
    arrays = _cube3d_crack_panel_data(
        info["final_space_d"], info["final_sol_d"], info["box"])
    return record, arrays


def fracture_3d():
    """The notched cube, its two-dimensional limits, and the solver comparison."""
    print("[8] Phase-field fracture in three dimensions")
    record, arrays = figure_data("fracture_3d", _fracture_3d_compute)
    RESULTS["fracture_3d"] = record["results"]
    _cube3d_figure(record, arrays)
    print(f"    two-dimensional limits bracket the cube: "
          f"{record['results']['brackets']}")


def _cube3d_crack_panel_data(space_d, sol_d, box, n_slice=9, n_grid=200):
    """Damage field and mesh on the slice y = mid-thickness, x-z plane.

    A slice rather than a 3D perspective plot, for the reason given at
    :func:`poisson_solution_figure`: a section is drawn to scale, a projection
    is not. The crack in this benchmark is unbroken through the thickness (see
    :func:`jaxiga.forms.library.edge_crack_distance`), so the mid-thickness
    plane shows it at its most developed.
    """
    from scipy.interpolate import griddata

    lower, upper = box
    axis, cut = 1, 0.5 * (lower[1] + upper[1])
    p = (cut - lower[axis]) / (upper[axis] - lower[axis])
    ps, others = slice_pointset(space_d, axis, p, n_slice)
    b = evaluate(space_d, ps)
    x = np.asarray(b.x).reshape(-1, 3)
    dmg = np.clip(np.asarray(sol_d.at(basis=b)[..., 0]).reshape(-1), 0.0, 1.0)

    g0 = np.linspace(lower[others[0]], upper[others[0]], n_grid)
    g1 = np.linspace(lower[others[1]], upper[others[1]], n_grid)
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    field = griddata(x[:, others], dmg, (G0, G1), method="linear")

    # G0/G1 are a regular meshgrid over ``extent``, so they are rebuilt when
    # drawing rather than stored -- two n_grid^2 arrays saved for four numbers.
    return {
        "field": np.asarray(field, dtype=_PLOT_DTYPE),
        "mesh": np.asarray(box_slice_lines(space_d, box, axis, cut),
                           dtype=_PLOT_DTYPE),
        "extent": np.array([lower[others[0]], upper[others[0]],
                            lower[others[1]], upper[others[1]]], dtype=float),
        "cut": np.array(cut, dtype=float),
    }


def _cube3d_crack_panel(ax, data):
    """Draw the mid-thickness damage slice from :func:`_cube3d_crack_panel_data`."""
    from matplotlib.collections import LineCollection

    x0, x1, y0, y1 = data["extent"]
    field = data["field"]
    G0, G1 = np.meshgrid(np.linspace(x0, x1, field.shape[0]),
                         np.linspace(y0, y1, field.shape[1]), indexing="ij")
    sc = ax.pcolormesh(G0, G1, field, cmap="inferno",
                       vmin=0.0, vmax=1.0, shading="gouraud", rasterized=True)
    ax.add_collection(LineCollection(list(data["mesh"]), lw=0.2,
                                     colors="tab:cyan", alpha=0.5))
    ax.set_aspect("equal")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    return sc


def _cube3d_figure(record, arrays):
    info_of, curve_of = record["results"], record["curves"]

    def curve(key):
        c = curve_of[key]
        return (info_of[key], np.asarray(c["u"]), np.asarray(c["f"]),
                np.asarray(c["elems"]))

    def pretty(label):
        return label.replace("l = ", r"$\ell$ = ").replace(" x ", r" $\times$ ")

    # Drawn narrower than before, so that the type is not scaled down as far
    # when the figure is set at the text width.
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.9))
    for key, style, colour in [("plate_strain", "-", "0.45"),
                               ("plate_stress", "-", "0.72")]:
        info, u, f, _ = curve(key)
        ax[0].plot(u * 1e3, f, style, lw=1.1, color=colour,
                   label=f"{pretty(info['label'])}  (peak {f.max():.1f} N)")
    for key, colour, style in [("cube_16", "0.2", "--"), ("cube_32", "tab:purple", "-")]:
        info, u, f, _ = curve(key)
        ax[0].plot(u * 1e3, f, style, lw=1.7, color=colour,
                   label=f"{pretty(info['label'])}  (peak {f.max():.1f} N)")
    ax[0].set_xlabel(r"prescribed displacement [$\mu$m]")
    ax[0].set_ylabel(r"reaction $F_z$ [N]")
    ax[0].set_title(r"notched cube, $1\times0.2\times1$", fontsize=10)

    for key, colour in [("cube_16", "0.2"), ("cube_32", "tab:purple")]:
        info, _, _, elems = curve(key)
        ax[1].plot(np.arange(len(elems)), elems, lw=1.4, color=colour,
                   label=f"{pretty(info['label'])}  ({elems[0]:,}$\\to${elems[-1]:,})")
    ax[1].axhline(8 * 4 * 2 * 4 * 8 * 4, color="0.4", ls=":", lw=1.2,
                  label=r"uniform at the same $h$ (8,192 el.)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("load increment")
    ax[1].set_ylabel("elements")
    ax[1].set_title("elements the solver carries", fontsize=10)
    for a in ax[:2]:
        a.grid(alpha=0.3, which="both"); a.tick_params(labelsize=9)
        # below the axes: the curves fill both panels
        a.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.24),
                 frameon=False)

    sc = _cube3d_crack_panel(ax[2], arrays)
    ax[2].set_title(rf"damage + mesh, $y={float(arrays['cut']):.1f}$"
                    rf" ($\ell=1/16$)", fontsize=10)
    ax[2].tick_params(labelsize=9)
    fig.colorbar(sc, ax=ax[2], shrink=0.9, label=r"damage $\phi$")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "cube3d.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/cube3d.pdf")


if __name__ == "__main__":
    run_example("fracture_3d", fracture_3d)
