"""Kirchhoff-Love shell: flat-plate consistency and the Scordelis-Lo roof benchmark.

Run from the repository root::

    python -m paper.experiments.kirchhoff_love_shell

Writes tables/kirchhoff_love_shell.tex, figures/shell_roof.pdf and
figures/shell_convergence.pdf. The solves are cached in figdata/.
See paper/README.md for outputs, caching and the suggested reading order.
"""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxiga as jx
from jaxiga.examples import scordelis_lo_roof as roof
from jaxiga.geometry.nurbs import Patch
from jaxiga.space.pointset import PointSet

from ._common import FIGURES, RESULTS, figure_data, run_example, write_table

DEGREES, MESHES = (2, 3, 4), (4, 8, 16, 32)
LITERATURE = 0.3024  # MacNeal and Harder (1985)


def _structured(space, solution, nu_, nv_):
    """Geometry and displacement on a structured parametric grid (for surface plots)."""
    from jaxiga.space.evaluation import evaluate

    boxes = np.asarray(space.elem_vertex)
    uu, vv = np.meshgrid(np.linspace(0, 1, nu_), np.linspace(0, 1, nv_))
    elems, refs = [], []
    for u, v in zip(uu.ravel(), vv.ravel()):
        e = int(np.flatnonzero((boxes[:, 0] <= u + 1e-12) & (u - 1e-12 <= boxes[:, 2])
                               & (boxes[:, 1] <= v + 1e-12) & (v - 1e-12 <= boxes[:, 3]))[0])
        lo, hi = boxes[e, :2], boxes[e, 2:]
        elems.append(e); refs.append(np.clip(2 * (np.array([u, v]) - lo) / (hi - lo) - 1, -1, 1))
    ps = PointSet(elems=np.array(elems), ref=np.array(refs)[:, None, :], tag="grid")
    x = np.asarray(evaluate(space, ps).x)[:, 0].reshape(nv_, nu_, 3)
    u = np.asarray(solution.at(ps))[:, 0].reshape(nv_, nu_, 3)
    return x, u


def _compute():
    rows = []
    for p in DEGREES:
        for n in MESHES:
            space, sol = roof.solve_roof(p, n)
            uz = -float(roof.displacement_at(space, sol, 0.0, 0.5)[2])
            rows.append({"p": p, "elements": n, "dofs": int(space.n_dofs), "uz_A": uz})
            print(f"    p={p} {n:2d}x{n:<2d} dofs={space.n_dofs:5d} u_z(A)={uz:.6f}")

    # Consistency on flat geometry: the shell must reproduce the plate exactly.
    E, nu, t, q = 210e3, 0.3, 0.01, 1.0
    flat = jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(4)
    plate = jx.KirchhoffPlate(jx.FunctionSpace(flat), source=lambda x, p: q,
                              dirichlet=[jx.ClampedBC(where=s) for s in ("left", "right", "bottom", "top")])
    w_plate = jx.solve(plate, params={"E": E, "nu": nu, "t": t})
    surface = Patch.create(flat.knots, flat.degree, np.c_[np.asarray(flat.ctrl_pts), np.zeros(len(flat.ctrl_pts))],
                           np.asarray(flat.weights), labels=dict(flat.labels))
    Vs = jx.FunctionSpace(surface, vec=3)
    shell = jx.KirchhoffLoveShell(Vs, source=lambda x, p: jnp.array([0.0, 0.0, q]),
                                  dirichlet=[jx.ClampedBC(where=s) for s in ("left", "right", "bottom", "top")])
    w_shell = jx.solve(shell, params=shell.parameters(E, nu, t))
    from jaxiga.space.pointset import grid
    a = np.asarray(w_plate.at(grid(w_plate.space, 5)))[..., 0]
    b = np.asarray(w_shell.at(grid(Vs, 5)))
    plate_diff = float(np.abs(a - b[..., 2]).max() / np.abs(a).max())
    centre = float(roof.displacement_at(Vs, w_shell, 0.5, 0.5)[2])
    timoshenko = 0.0012653 * q / (E * t**3 / (12 * (1 - nu**2)))

    space, sol = roof.solve_roof(3, 16)
    x, u = _structured(space, sol, 41, 41)
    record = {"rows": rows, "plate_relative_difference": plate_diff,
              "clamped_centre": centre, "timoshenko": timoshenko}
    return record, {"x": x, "u": u}


def _schematic(ax):
    """Geometry and boundary conditions of the Scordelis-Lo roof."""
    R, L, phi = roof.RADIUS, roof.LENGTH, roof.HALF_ANGLE
    th = np.linspace(-phi, phi, 40)
    ys = np.linspace(0, L, 20)
    T, Y = np.meshgrid(th, ys)
    ax.plot_surface(R * np.sin(T), Y, R * np.cos(T), color="0.85", alpha=0.6, linewidth=0, shade=False)
    for y in (0.0, L):  # rigid diaphragms
        ax.plot(R * np.sin(th), np.full_like(th, y), R * np.cos(th), color="C0", lw=2.5)
    for s in (-1, 1):  # free edges
        ax.plot([s * R * np.sin(phi)] * 2, [0, L], [R * np.cos(phi)] * 2, color="k", lw=1.2, ls="--")
    for tt in np.linspace(-0.7 * phi, 0.7 * phi, 3):
        for y in (0.3 * L, 0.7 * L):
            x0, z0 = R * np.sin(tt), R * np.cos(tt)
            ax.quiver(x0, y, z0 + 6, 0, 0, -5, color="C3", arrow_length_ratio=0.35, lw=0.8)
    xa, za = -R * np.sin(phi), R * np.cos(phi)
    ax.scatter([xa], [L / 2], [za], color="k", s=22, depthshade=False, zorder=10)
    ax.set_xlabel("$x$", labelpad=-8); ax.set_ylabel("$y$", labelpad=-8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=22, azim=-58); ax.set_box_aspect((2 * R * np.sin(phi), L, 0.6 * R))
    ax.figure.canvas.draw()  # fix the projection before anchoring 2D labels

    from mpl_toolkits.mplot3d import proj3d

    def label(point, text, offset, color="k"):
        x2, y2, _ = proj3d.proj_transform(*point, ax.get_proj())
        ax.annotate(text, xy=(x2, y2), xytext=offset, textcoords="offset points", fontsize=7,
                    color=color, ha="center", va="center",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6})

    t20 = np.deg2rad(20.0)
    label((-R * np.sin(t20), 0.0, R * np.cos(t20)), "rigid diaphragm\n$u_x=u_z=0$", (-28, -38), "C0")
    label((0.0, L, R), "rigid diaphragm\n$u_x=u_z=0$", (22, 32), "C0")
    label((R * np.sin(phi), 0.3 * L, R * np.cos(phi)), "free edge", (46, -16))
    label((xa, L / 2, za), "A", (-4, 20))
    label((0.0, 0.5 * L, R + 6), "self-weight", (-62, 34), "C3")
    return "(a) geometry and boundary conditions"


def _deformed(ax, fig, x, u):
    scale = 10.0
    d = x + scale * u
    uz = u[..., 2]
    norm = plt.Normalize(uz.min(), uz.max())
    ax.plot_surface(d[..., 0], d[..., 1], d[..., 2], facecolors=plt.cm.viridis(norm(uz)),
                    linewidth=0, antialiased=False, shade=False, rstride=1, cstride=1)
    ax.plot_wireframe(x[..., 0], x[..., 1], x[..., 2], color="0.6", lw=0.3, rstride=5, cstride=5)
    m = plt.cm.ScalarMappable(norm=norm, cmap="viridis"); m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.02, label="$u_z$")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # Look along the axis (from y < 0): both free edges are seen symmetrically.
    ax.view_init(elev=20, azim=-90); ax.set_box_aspect((2 * 25 * np.sin(np.deg2rad(40)), 50, 0.6 * 25), zoom=1.3)
    return f"(b) deformed shape ($\\times{scale:.0f}$), $p=3$, $16\\times16$"


def kirchhoff_love_shell():
    print("[9] Kirchhoff-Love shell: Scordelis-Lo roof")
    record, arrays = figure_data("kirchhoff_love_shell", _compute)
    rows = record["rows"]
    converged = rows[-1]["uz_A"]
    RESULTS["shell_uz_converged"] = converged
    RESULTS["shell_plate_relative_difference"] = record["plate_relative_difference"]
    RESULTS["shell_clamped_ratio"] = record["clamped_centre"] / record["timoshenko"]
    print(f"    converged u_z(A) = {converged:.5f}; flat shell vs plate "
          f"{record['plate_relative_difference']:.1e}")

    fig = plt.figure(figsize=(7.2, 3.3))
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2, projection="3d")
    title_a = _schematic(ax_a)
    title_b = _deformed(ax_b, fig, arrays["x"], arrays["u"])
    fig.subplots_adjust(wspace=0.02)
    # The colourbar shrinks the second 3D axes, so per-axes titles end up at
    # different heights; both are set on one baseline of the figure instead.
    for a, title, shift in ((ax_a, title_a, -0.02), (ax_b, title_b, 0.03)):
        box = a.get_position()
        fig.text(0.5 * (box.x0 + box.x1) + shift, 0.9, title, ha="center", va="bottom", fontsize=9)
    fig.savefig(os.path.join(FIGURES, "shell_roof.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for k, p in enumerate(DEGREES):
        sel = [r for r in rows if r["p"] == p]
        ax.semilogx([r["dofs"] for r in sel], [r["uz_A"] for r in sel], "o-", ms=3.5, color=f"C{k}", label=f"$p={p}$")
    ax.axhline(LITERATURE, color="0.3", ls=":", lw=1, label="0.3024 (MacNeal\u2013Harder)")
    ax.axhline(converged, color="0.3", ls="--", lw=1, label=f"{converged:.4f} (converged)")
    ax.set_ylim(0.225, 0.31); ax.set_xlabel("degrees of freedom"); ax.set_ylabel("$-u_z$ at A")
    # below the axes: inside them the box covers the p=2 curve
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(FIGURES, "shell_convergence.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/shell_roof.pdf, shell_convergence.pdf")

    lines = [r"\begin{tabular}{c r r r r}", r"\hline",
             r" & \multicolumn{4}{c}{$-u_z$ at A for $n\times n$ elements} \\",
             r"$p$ & " + " & ".join(f"${n}\\times{n}$" for n in MESHES) + r" \\", r"\hline"]
    for p in DEGREES:
        vals = [r["uz_A"] for r in rows if r["p"] == p]
        lines.append(f"{p} & " + " & ".join(f"{v:.4f}" for v in vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    write_table("kirchhoff_love_shell", "\n".join(lines))


if __name__ == "__main__":
    run_example("kirchhoff_love_shell", kirchhoff_love_shell)
