"""Shared mesh construction and load stepping for the fracture examples."""

import gc
import time

import jax
import matplotlib
import numpy as np

import jaxiga as jx
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

from ._plotting import element_triangulation, mesh_lines


def _fracture_case(name, ell, dirichlet, reaction, crack_axes, box, aspect,
                   material, schedule, stop_below=None):
    return dict(name=name, ell=ell, dirichlet=dirichlet, reaction=reaction,
                crack_axes=crack_axes, box=box, aspect=aspect, material=material,
                schedule=schedule, stop_below=stop_below)


def _fracture_mesh(case, base, levels, band=None):
    """Base mesh, refined either around the notch or across a whole band."""
    from jaxiga.forms.library import edge_crack_distance
    from jaxiga.solvers.phase_field import seed_refine

    lower, upper = case["box"]
    counts = [max(1, round(base * a)) for a in case["aspect"]]
    knots = [np.linspace(0.0, 1.0, n + 1)[1:-1] for n in counts]
    primitive = (jx.primitives.rectangle(*lower, *upper) if len(lower) == 2
                 else jx.primitives.cuboid(lower, upper))
    patch = primitive.elevate(2).insert_knots(*knots)
    V = jx.refine_elements(jx.FunctionSpace(patch, vec=1), [])
    crack = edge_crack_distance((0.5, 0.5), axes=case["crack_axes"])

    if band is None:
        return seed_refine(V, crack, case["ell"], levels), crack, counts

    axis = case["crack_axes"][1]
    for _ in range(levels):
        box = np.asarray(V.elem_vertex)
        dim = V.dim
        near = (box[:, dim + axis] >= 0.5 - band * case["ell"]) & (
            box[:, axis] <= 0.5 + band * case["ell"])
        marked = np.flatnonzero(near & (np.asarray(V.elem_key)[:, 1] < levels))
        if not marked.size:
            break
        V = jx.refine_elements(V, marked)
    return V, crack, counts


def _run_fracture(case, base, levels, *, band=None, adaptive=True, dilate=1.0,
                  n_steps=None, label="", capture_final=False):
    """One load path; returns the curve and what it cost.

    ``capture_final`` keeps the damage field and mesh of the last converged
    step (for the figure that shows the crack the run actually found), at the
    cost of building one extra ``Solution`` wrapper per step -- cheap, since it
    wraps arrays already computed for the reaction. Only the last step's
    objects are kept alive; earlier ones are collected as the loop proceeds.
    """
    from jaxiga.solvers.phase_field import (Adaptivity, Fracture, Material,
                                            StaggeredSolver)

    V, crack, counts = _fracture_mesh(case, base, levels, band=band)
    solver = StaggeredSolver(
        V, case["material"], Fracture(Gc=case["Gc"], ell=case["ell"]),
        dirichlet=case["dirichlet"], reaction=case["reaction"], crack=crack,
        adaptivity=Adaptivity(max_level=levels, dilate=dilate, enabled=adaptive))

    steps = np.concatenate([np.full(n, d) for n, d in case["schedule"]])
    loads = np.cumsum(steps if n_steps is None else steps[:n_steps])

    peak_force, stopped = 0.0, None
    u, f, elems = [], [], []
    final_space_d, final_sol_d = None, None
    t0 = time.perf_counter()
    for step in solver.run(loads, keep_solutions=capture_final):
        u.append(step.load); f.append(abs(step.force)); elems.append(step.n_elems)
        peak_force = max(peak_force, abs(step.force))
        if capture_final:
            final_space_d, final_sol_d = step.space_d, step.sol_d
        if (case["stop_below"] is not None and peak_force > 0.0
                and abs(step.force) < case["stop_below"] * peak_force):
            stopped = step.index
            break
    elapsed = time.perf_counter() - t0
    remeshes = solver.remeshes
    sweeps = solver.stats["sweeps"]
    remesh_seconds = solver.stats["remesh"]

    # Every remesh compiles five kernels at shapes never seen again, and XLA
    # folds the basis arrays into each as constants. Across several cases in one
    # process that accumulation is enough to exhaust memory, so the cache goes
    # when the run that filled it does. Everything returned below is NumPy.
    n_dofs_end = int(solver.mesh.Vu.n_dofs)
    del solver
    gc.collect()
    jax.clear_caches()

    u, f, elems = np.array(u), np.array(f), np.array(elems)
    peak = int(np.argmax(f))
    h = 1.0 / (counts[0] * 2 ** levels)
    out = {"label": label, "peak_force": float(f[peak]), "peak_disp": float(u[peak]),
           "final_force": float(f[-1]), "steps": len(u), "stopped_at": stopped,
           "elems_start": int(elems[0]), "elems_end": int(elems[-1]),
           "dofs_end": n_dofs_end, "remeshes": remeshes,
           "sweeps": sweeps, "remesh_seconds": remesh_seconds,
           "seconds": elapsed, "l_over_h": case["ell"] / h}
    if capture_final:
        out["final_space_d"] = final_space_d
        out["final_sol_d"] = final_sol_d
        out["box"] = case["box"]
        # Figure plumbing, not results: see _summary_only, which strips these
        # before RESULTS is serialised.
    print(f"    {label:34s} peak {f[peak]:7.2f} N at u={u[peak]:.5f}, "
          f"{elems[0]}->{elems[-1]} el, {len(u)} steps, {elapsed:.0f}s")
    return out, u, f, elems

#: What ``capture_final`` adds to a run's dict for the figures to draw. These
#: are live objects, not numbers, and must not reach ``results_summary.json``.
_CAPTURE_KEYS = ("final_space_d", "final_sol_d", "box")


def _summary_only(info):
    """A run's dict with the figure-only captures removed, so it serialises.

    ``json.dump(..., default=float)`` calls ``float()`` on anything it does not
    recognise, so a ``FunctionSpace`` left in ``RESULTS`` aborts the write
    *after* the file has been truncated -- losing the whole summary at the very
    end of a run that has already done all its work.
    """
    return {k: v for k, v in info.items() if k not in _CAPTURE_KEYS}


#: Plot data is stored at single precision. It exists only to be drawn -- a
#: figure resolves nothing near float32 -- and it halves what goes into git.
#: Quantities the paper *quotes* live in the JSON record at full precision.
_PLOT_DTYPE = np.float32


def _crack_panel_data(space_d, sol_d, n_samp=4):
    """Sample a 2D damage field and its mesh into plain arrays.

    Split from :func:`_crack_panel` so the figure can be drawn from committed
    data without the space and solution objects, which only a full solve
    produces. See :func:`figure_data`.

    The triangulation is not stored: ``element_triangulation`` is a pure
    function of the element and sample counts, so it is rebuilt when drawing
    rather than committed (it is the largest array here by some margin).
    """
    b = evaluate(space_d, P.grid(space_d, n_samp))
    return {
        "x": np.asarray(b.x, dtype=_PLOT_DTYPE).reshape(-1, 2),
        "dmg": np.clip(np.asarray(sol_d.at(basis=b)[..., 0], dtype=_PLOT_DTYPE)
                       .reshape(-1), 0.0, 1.0),
        "mesh": np.asarray(mesh_lines(space_d, n=2), dtype=_PLOT_DTYPE),
        "n_elems": np.asarray(b.n_elems),
        "n_samp": np.asarray(n_samp),
    }


def _crack_panel(ax, data):
    """Final damage field with the mesh that resolved it overlaid.

    One panel rather than the two separate ones used for the hand-laid mesh in
    Figure~\\ref{fig:phase_field}: here the point being illustrated is that the
    refinement *follows* the damage, and that is only visible with both drawn
    together.
    """
    from matplotlib.collections import LineCollection

    x = data["x"]
    tri = element_triangulation(int(data["n_elems"]), int(data["n_samp"]))
    triang = matplotlib.tri.Triangulation(x[:, 0], x[:, 1], tri)
    sc = ax.tripcolor(triang, data["dmg"], shading="gouraud", cmap="inferno",
                      vmin=0.0, vmax=1.0)
    ax.add_collection(LineCollection(list(data["mesh"]), lw=0.15,
                                     colors="tab:cyan", alpha=0.5))
    ax.set_aspect("equal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return sc
