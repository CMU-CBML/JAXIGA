"""Notched specimens on a mesh that follows the crack: tension, shear, and a cube.

Three standard benchmarks, solved with the adaptive staggered scheme in
:mod:`jaxiga.solvers.phase_field`. The mesh starts coarse away from the notch
and is refined during the run wherever damage appears, so the cost of resolving
the damage band is paid only along the crack that actually forms.

``tension`` is the case where the crack path is known and a band can be
pre-refined by hand -- adaptivity rediscovers that band rather than beating it.
``shear`` is the case where it cannot: the crack curves to the lower right
corner, and the alternative is a uniform mesh six times larger. ``cube`` is the
tension test given a thickness, in 3D.

Usage::

    python adaptive_fracture.py tension --movie
    python adaptive_fracture.py shear --base 20 --levels 3
    python adaptive_fracture.py cube --base 8 --levels 2

``--movie`` renders one frame per output step and encodes them to MP4: in 2D the
mesh, stress and damage, the layout the IGAPack MATLAB code writes; in 3D the
crack surface in space together with the mesh and damage on a mid-thickness
slice. ``--vtk`` writes a ``.vtu`` per step and a ``.pvd`` collection, which is
what to open in ParaView and where the full 3D fields live.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

import jaxiga as jx
from jaxiga.forms.library import edge_crack_distance
from jaxiga.solvers.phase_field import (
    Adaptivity,
    Fracture,
    Material,
    StaggeredSolver,
    seed_refine,
)


def band_refine(space, levels, half_width, axis=1, centre=0.5):
    """Pre-refine a straight band across the whole specimen.

    The reference mesh an analyst lays out by hand when the crack path is known
    in advance -- which for the tension test it is. It is the mesh adaptivity
    has to beat, and the mesh adaptivity should *rediscover*.
    """
    for _ in range(levels):
        box = np.asarray(space.elem_vertex)
        dim = space.dim
        lo, hi = box[:, axis], box[:, dim + axis]
        near = (hi >= centre - half_width) & (lo <= centre + half_width)
        level = np.asarray(space.elem_key)[:, 1]
        marked = np.flatnonzero(near & (level < levels))
        if not marked.size:
            break
        space = jx.refine_elements(space, marked)
    return space

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# the two benchmarks
# --------------------------------------------------------------------------


def tension(ell=0.0125):
    """Miehe's tension test: the notched half is pulled apart vertically.

    The crack runs straight ahead of the notch, so the response is the classic
    brittle one -- linear to a peak, then an almost vertical drop.
    """
    return dict(
        name="tension",
        fracture=Fracture(Gc=2.7, ell=ell),
        dirichlet=[
            jx.DirichletBC([0.0, 0.0], where="bottom"),
            jx.DirichletBC([0.0, 1.0], where="top"),
        ],
        reaction=("top", 1),
        schedule=[(45, 1e-4), (1155, 1e-6)],
    )


def shear(ell=0.015):
    """Miehe's shear test: the top edge slides horizontally.

    The crack curves down towards the lower right corner rather than running
    straight, which is the point of the test -- it is the case a fixed
    pre-refined band cannot be laid out in advance, so it is the case adaptivity
    is actually needed for. The vertical edges are held against transverse
    motion, as in the reference implementation.

    The reported force is the horizontal reaction on the driven edge. The
    reference code measures it on the fixed edge instead, which by equilibrium
    is the same magnitude with the opposite sign, since the only other
    constrained direction is vertical.
    """
    return dict(
        name="shear",
        fracture=Fracture(Gc=2.7, ell=ell),
        dirichlet=[
            jx.DirichletBC([0.0, 0.0], where="bottom"),
            jx.DirichletBC(1.0, where="top", component=0),
            jx.DirichletBC(0.0, where="top", component=1),
            jx.DirichletBC(0.0, where="left", component=1),
            jx.DirichletBC(0.0, where="right", component=1),
        ],
        reaction=("top", 0),
        schedule=[(80, 1e-4), (620, 1e-5)],
    )


def cube(ell=1.0 / 32.0):
    """The notched cube: the tension test given a thickness.

    A 1 x 0.2 x 1 block with a crack through the thickness at ``z = 0.5``,
    ``x < 0.5``, pulled apart in ``z``. Held against rigid-body motion by one
    face per remaining direction, as in the reference implementation, so the
    lateral faces are free to contract.

    In 3D the solver switches to conjugate gradients on its own; a direct
    factorisation of these systems costs more than the assembly that builds
    them.
    """
    return dict(
        name="cube",
        dim=3,
        box=((0.0, 0.0, 0.0), (1.0, 0.2, 1.0)),
        aspect=(1, 0.25, 1),          # elements per side, relative to --base
        material=Material(E=20.8e3, nu=0.3, plane="strain"),
        fracture=Fracture(Gc=0.5, ell=ell),
        crack_axes=(0, 2),
        dirichlet=[
            jx.DirichletBC(0.0, where="back", component=2),    # z = 0 held
            jx.DirichletBC(1.0, where="front", component=2),   # z = 1 driven
            jx.DirichletBC(0.0, where="bottom", component=1),  # one y face
            jx.DirichletBC(0.0, where="left", component=0),    # one x face
        ],
        reaction=("front", 2),
        # Ramp quickly through the elastic range, then crawl.
        #
        # Measured, halving l0 from 1/16 to 1/32 at l/h = 2 throughout: the peak
        # rises from 11.68 N to 14.87 N, a factor of 1.27 against the 1.41 that
        # pure AT2 strength scaling (sigma_c ~ l0^-1/2) would give, while the
        # displacement at the peak does not move at all -- 0.00800 in both. So
        # this benchmark is *not* converged in l0: the initial crack here is a
        # seeded damage band of width l0 rather than a geometric notch, so
        # material strength still governs the peak and does not hand over to a
        # Gc-controlled Griffith load. Quoting a peak load means quoting l0 with
        # it. What is stable is where the peak sits, and that the drop after it
        # sharpens as l0 falls -- so the increments through it stay small.
        schedule=[(12, 5e-4), (20, 1e-4), (600, 2.5e-5)],
        stop_below=0.05,
    )


CASES = {"tension": tension, "shear": shear, "cube": cube}


def default_case(case):
    """Fill in the 2D defaults the plate cases do not bother to state."""
    case.setdefault("stop_below", None)
    case.setdefault("dim", 2)
    case.setdefault("box", ((0.0, 0.0), (1.0, 1.0)))
    case.setdefault("aspect", (1, 1))
    case.setdefault("material", Material(E=210e3, nu=0.3, plane="strain"))
    case.setdefault("crack_axes", (0, 1))
    return case


def resident_gb():
    """Resident set size, in GB.

    Reported alongside the load steps because in 3D memory, not time, is what
    ends a run: a remesh at 16000 elements peaks near 8 GB, and a machine that
    runs out kills the process without a traceback. Watching it climb is the
    difference between diagnosing that and guessing at it.
    """
    try:
        out = os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()
        return int(out) / 1024.0 / 1024.0
    except (ValueError, OSError):  # pragma: no cover - platform dependent
        return float("nan")


def has_broken(force, peak, stop_below):
    """Has the specimen lost its load-carrying capacity?

    True once the reaction is below ``stop_below`` of the largest it reached.
    A brittle crack that has run through leaves a residual of a few per cent --
    the degraded stiffness of the broken band -- and every increment after that
    repeats the same picture, so a run that keeps going is only burning time.

    ``stop_below`` of ``None`` disables the test, which is what the cases whose
    load path is meant to be followed to its end use.
    """
    return (
        stop_below is not None
        and peak > 0.0
        and abs(force) < stop_below * peak
    )


def load_path(schedule, n_steps=None):
    """Absolute prescribed displacements from ``[(count, increment), ...]``."""
    steps = np.concatenate([np.full(n, d) for n, d in schedule])
    if n_steps is not None:
        steps = steps[:n_steps]
    return np.cumsum(steps)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def sample_fields(step, n):
    """Points, displacement, damage and the stress actually carried, on a grid.

    The stress is the *degraded* stress ``g(phi) sigma``, so a formed crack
    reads as the stress-free surface it is rather than as the band of large
    strain it also is. Dimension-generic: the same call feeds the 2D figures
    and the 3D VTU files.

    Returns arrays shaped ``(n_elems, n_points_per_elem, ...)``.
    """
    import jax
    from jaxiga.forms import materials
    from jaxiga.space import pointset as P
    from jaxiga.space.evaluation import evaluate

    space = step.space_u
    basis = evaluate(space, P.grid(space, n))
    params = step.sol_u.params
    C = materials.constitutive(params["E"], params["nu"], space.dim,
                               step.sol_u.problem.plane)

    x = np.asarray(basis.x)
    u = np.asarray(step.sol_u.at(basis=basis))
    phi = np.clip(np.asarray(step.sol_d.at(basis=basis)[..., 0]), 0.0, 1.0)

    strain = jax.vmap(jax.vmap(materials.strain_voigt))(step.sol_u.grad(basis=basis))
    stress = ((1.0 - phi) ** 2)[..., None] * (strain @ np.asarray(C).T)
    von_mises = np.asarray(jax.vmap(jax.vmap(materials.von_mises))(stress))
    return basis, x, u, phi, von_mises


class VtkSeries:
    """One ``.vtu`` per output step, plus a ``.pvd`` collection tying them to time.

    The route for 3D, where a flat figure cannot show a crack surface. Each file
    carries the displacement, the damage, the degraded von Mises stress and the
    refinement level of every element, so the adapted mesh is paintable
    alongside the fields that drove it. Open the ``.pvd``, not the individual
    files, and the run plays as an animation.
    """

    def __init__(self, outdir, n=2):
        from jaxiga.post.vtk import write_series, write_vtu

        self._write_vtu, self._write_series = write_vtu, write_series
        self.outdir, self.n = outdir, n
        os.makedirs(outdir, exist_ok=True)
        self.paths, self.times = [], []

    def write(self, step):
        _, _, u, phi, vm = sample_fields(step, self.n)
        level = np.asarray(step.space_d.elem_key)[:, 1]
        path = self._write_vtu(
            step.space_u,
            os.path.join(self.outdir, f"step_{len(self.paths):05d}"),
            n=self.n - 1,
            point_data={"displacement": u, "damage": phi, "von_mises": vm},
            cell_data={"level": level},
        )
        self.paths.append(path)
        self.times.append(step.load)
        return path

    def close(self, name):
        return self._write_series(self.paths, self.times,
                                  os.path.join(self.outdir, name))


def _triangulation(x, n):
    """Corner-connected triangles for a per-element ``n x n`` sample grid."""
    n_e = x.shape[0]
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    c = (i * n + j).ravel()
    quad = np.stack([c, c + 1, c + n + 1, c + n], axis=1)
    tris = np.concatenate([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])
    offs = (np.arange(n_e) * n * n)[:, None, None]
    return (tris[None] + offs).reshape(-1, 3)


class Frames:
    """Renders one figure per output step: mesh, stress, damage.

    The layout follows ``plotDispPhase2D`` in the MATLAB reference: the mesh
    across the top, the deformed configuration coloured by von Mises stress
    below left, the damage field below right.

    The stress shown is the *degraded* stress ``g(phi) * sigma``, the stress the
    material actually carries, so a formed crack reads as a stress-free surface
    rather than as the band of large strain it also is.
    """

    def __init__(self, outdir, n=3, magnify=10.0, dpi=110, smax=None,
                 slice_axis=1, slice_value=0.1, slice_tol=1e-9):
        import matplotlib

        matplotlib.use("Agg")
        self.outdir, self.n, self.magnify, self.dpi = outdir, n, magnify, dpi
        os.makedirs(outdir, exist_ok=True)
        self.paths = []
        self.fixed_smax = smax
        self.smax = smax or 1.0
        self.slice_axis, self.slice_value = slice_axis, slice_value
        self.slice_tol = slice_tol

    def _fields(self, step):
        _, x, u, phi, vm = sample_fields(step, self.n)
        return (x.reshape(-1, 2), u.reshape(-1, 2), phi.reshape(-1),
                vm.reshape(-1), x)

    def draw(self, step, mesh_lines):
        if step.space_u.dim == 3:
            return self._draw3d(step)
        return self._draw2d(step, mesh_lines)

    def _draw2d(self, step, mesh_lines):
        import matplotlib.pyplot as plt
        import matplotlib.tri
        from matplotlib.collections import LineCollection

        x, u, phi, vm, gridded = self._fields(step)
        tri = _triangulation(gridded, self.n)
        if self.fixed_smax is None:
            self.smax = max(self.smax, float(np.percentile(vm, 99.5)))

        fig = plt.figure(figsize=(7.2, 6.4))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.28, wspace=0.28)

        ax = fig.add_subplot(gs[0, :])
        ax.add_collection(LineCollection(mesh_lines, lw=0.25, colors="0.35"))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"mesh: {step.n_elems} elements, {step.n_dofs} dofs", fontsize=9)

        d = np.stack([x[:, 0] + self.magnify * u[:, 0],
                      x[:, 1] + self.magnify * u[:, 1]], axis=1)
        axs = fig.add_subplot(gs[1, 0])
        t = matplotlib.tri.Triangulation(d[:, 0], d[:, 1], tri)
        sc = axs.tripcolor(t, vm, shading="gouraud", cmap="viridis",
                           vmin=0.0, vmax=self.smax)
        axs.set_title(rf"$\sigma_{{VM}}$, displacement $\times${self.magnify:g}",
                      fontsize=9)
        fig.colorbar(sc, ax=axs)

        axp = fig.add_subplot(gs[1, 1])
        t2 = matplotlib.tri.Triangulation(x[:, 0], x[:, 1], tri)
        sc2 = axp.tripcolor(t2, phi, shading="gouraud", cmap="inferno",
                            vmin=0.0, vmax=1.0)
        axp.set_title(r"damage $\phi$", fontsize=9)
        fig.colorbar(sc2, ax=axp)

        for a in (ax, axs, axp):
            a.set_aspect("equal")
            a.tick_params(labelsize=7)
        self._finish(fig, step)

    def _draw3d(self, step, threshold=0.4):
        """Crack surface in space, plus the mesh and damage on a mid-thickness slice.

        A 3D mesh drawn as a wireframe is unreadable and an isosurface needs a
        renderer this file does not have, so the crack is drawn as the cloud of
        sample points that have crossed the damage threshold -- for a band a few
        elements thick that reads as the surface it is -- and the slice panels
        give the same view as the 2D cases for comparison.
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri
        from matplotlib.collections import LineCollection

        _, x, _, phi, _ = sample_fields(step, self.n)
        axis = self.slice_axis
        value = self.slice_value
        flat_x, flat_phi = x.reshape(-1, 3), phi.reshape(-1)

        fig = plt.figure(figsize=(9.6, 4.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.35, wspace=0.2)

        ax = fig.add_subplot(gs[:, 0], projection="3d")
        hot = flat_phi > threshold
        if hot.any():
            ax.scatter(flat_x[hot, 0], flat_x[hot, 1], flat_x[hot, 2],
                       c=flat_phi[hot], cmap="inferno", vmin=0.0, vmax=1.0,
                       s=3, alpha=0.55, linewidths=0)
        lo, hi = flat_x.min(axis=0), flat_x.max(axis=0)
        for a in range(3):
            for corner in range(4):
                p0, p1 = lo.copy(), lo.copy()
                others = [d for d in range(3) if d != a]
                for k, d in enumerate(others):
                    if (corner >> k) & 1:
                        p0[d] = p1[d] = hi[d]
                p1[a] = hi[a]
                ax.plot(*zip(p0, p1), color="0.7", lw=0.6)
        ax.set_box_aspect(hi - lo)
        ax.set_title(rf"crack surface ($\phi > {threshold:g}$)", fontsize=9)
        # A thin direction crowds its tick labels into an unreadable smear, so
        # every axis gets the two ends and nothing between.
        for setter, l, h in zip((ax.set_xticks, ax.set_yticks, ax.set_zticks), lo, hi):
            setter([l, h])
        for label, setter in zip("xyz", (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)):
            setter(f"${label}$", fontsize=7, labelpad=-6)
        ax.tick_params(labelsize=5.5, pad=-2)
        ax.view_init(elev=20, azim=-62)

        names = ["x", "y", "z"]
        others = [d for d in range(3) if d != axis]
        near = np.abs(flat_x[:, axis] - value) < self.slice_tol
        axm = fig.add_subplot(gs[0, 1])
        hit, blo, bhi = slice_elements(step.space_d, axis, value, self.n)
        axm.add_collection(
            LineCollection(boxes_to_segments(blo, bhi), lw=0.25, colors="0.35")
        )
        axm.set_xlim(lo[others[0]], hi[others[0]])
        axm.set_ylim(lo[others[1]], hi[others[1]])
        axm.set_title(f"mesh on {names[axis]} = {value:g}: "
                      f"{int(hit.sum())} of {step.n_elems} elements", fontsize=8)

        axp = fig.add_subplot(gs[1, 1])
        if near.sum() > 3:
            pts = flat_x[near][:, others]
            triang = matplotlib.tri.Triangulation(pts[:, 0], pts[:, 1])
            sc = axp.tripcolor(triang, flat_phi[near], shading="gouraud",
                               cmap="inferno", vmin=0.0, vmax=1.0)
            fig.colorbar(sc, ax=axp)
        axp.set_xlim(lo[others[0]], hi[others[0]])
        axp.set_ylim(lo[others[1]], hi[others[1]])
        axp.set_title(rf"damage $\phi$ on {names[axis]} = {value:g}", fontsize=8)

        for a in (axm, axp):
            a.set_aspect("equal")
            a.set_xlabel(f"${names[others[0]]}$", fontsize=7)
            a.set_ylabel(f"${names[others[1]]}$", fontsize=7)
            a.tick_params(labelsize=6)
        self._finish(fig, step)

    def _finish(self, fig, step):
        import matplotlib.pyplot as plt

        fig.suptitle(f"step {step.index}:  $u$ = {step.load * 1e3:.4f} mm,"
                     f"  $F$ = {step.force:.1f} N", fontsize=10)
        path = os.path.join(self.outdir, f"frame_{len(self.paths):05d}.png")
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        self.paths.append(path)

    def encode(self, target, fps=15):
        """Stitch the frames into an MP4 (falling back to an animated GIF)."""
        import shutil
        import subprocess

        if shutil.which("ffmpeg") is None:
            return self._gif(os.path.splitext(target)[0] + ".gif", fps)
        pattern = os.path.join(self.outdir, "frame_%05d.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", target],
            check=True,
        )
        return target

    def _gif(self, target, fps):
        from PIL import Image

        images = [Image.open(p) for p in self.paths]
        images[0].save(target, save_all=True, append_images=images[1:],
                       duration=int(1000 / fps), loop=0)
        return target


def slice_elements(space, axis, value, n=2):
    """Elements the plane ``x[axis] = value`` passes through.

    Returns their boxes in the two remaining coordinates, read off the sampled
    physical points so curved geometry stays honest. A 3D mesh drawn as a
    wireframe is unreadable; a slice through it is exactly the 2D picture, and
    for a crack that runs across the thickness it is the informative one.
    """
    from jaxiga.space import pointset as P
    from jaxiga.space.evaluation import evaluate

    x = np.asarray(evaluate(space, P.grid(space, n)).x)   # (n_e, n_q, dim)
    lo, hi = x.min(axis=1), x.max(axis=1)
    hit = (lo[:, axis] <= value) & (hi[:, axis] >= value)
    others = [d for d in range(x.shape[-1]) if d != axis]
    return hit, lo[hit][:, others], hi[hit][:, others]


def boxes_to_segments(lo, hi):
    """Rectangle outlines as segments for a LineCollection."""
    segments = []
    for (x0, y0), (x1, y1) in zip(lo, hi):
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        segments.extend([corners[k], corners[k + 1]] for k in range(4))
    return np.asarray(segments)


def mesh_lines(space, n=2):
    """Element boundaries in physical space, as segments for a LineCollection."""
    from jaxiga.space import pointset as P
    from jaxiga.space.evaluation import evaluate

    b = evaluate(space, P.grid(space, n))
    x = np.asarray(b.x).reshape(b.n_elems, n, n, -1)
    segs = []
    for corner in (x[:, 0, :, :], x[:, -1, :, :], x[:, :, 0, :], x[:, :, -1, :]):
        segs.append(np.stack([corner[:, :-1], corner[:, 1:]], axis=2).reshape(-1, 2, 2))
    return np.concatenate(segs)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=sorted(CASES), nargs="?", default="tension")
    ap.add_argument("--base", type=int, default=20, help="elements per side of the base mesh")
    ap.add_argument("--levels", type=int, default=3, help="refinement levels above the base mesh")
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--dilate", type=float, default=1.0, help="refine within this many l of damage")
    ap.add_argument("--steps", type=int, default=None, help="truncate the load path")
    ap.add_argument("--ell", type=float, default=None, metavar="L0",
                    help="regularisation length, overriding the case default; a "
                         "smaller one is a sharper crack and needs more levels "
                         "under it to keep l/h")
    ap.add_argument("--stop-below", type=float, default=None, metavar="FRACTION",
                    help="stop once the reaction falls below this fraction of its "
                         "peak, i.e. once the crack has run through; the "
                         "increments after that carry no information")
    ap.add_argument("--fixed", action="store_true", help="freeze the mesh (reference run)")
    ap.add_argument("--band", type=float, default=None, metavar="HALFWIDTH_IN_L",
                    help="pre-refine a straight band of this half-width instead of "
                         "only the notch (the hand-laid reference mesh)")
    ap.add_argument("--movie", action="store_true")
    ap.add_argument("--vtk", action="store_true",
                    help="write one .vtu per output step plus a .pvd collection")
    ap.add_argument("--vtk-samples", type=int, default=2, metavar="N",
                    help="sample points per element and direction in the .vtu "
                         "(2 = element corners; 3 subdivides each element)")
    ap.add_argument("--every", type=int, default=5, help="steps between movie frames")
    ap.add_argument("--vtk-every", type=int, default=1, metavar="N",
                    help="steps between .vtu files (default: every step). A .vtu "
                         "is a thousand times the size of a rendered frame, so "
                         "raise this if disk is short -- but a time series with "
                         "gaps in it is a poor record of a propagating crack")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--smax", type=float, default=None,
                    help="fix the stress colour scale (default: running max)")
    ap.add_argument("--linear", default="auto", choices=["auto", "direct", "cg"],
                    help="linear solver (auto: direct in 2D, CG in 3D)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    case = default_case(CASES[args.case](**({"ell": args.ell} if args.ell else {})))
    if args.stop_below is not None:
        case["stop_below"] = args.stop_below
    out = args.out or os.path.join(HERE, f"adaptive_{case['name']}")
    os.makedirs(out, exist_ok=True)
    frac, material = case["fracture"], case["material"]
    crack = edge_crack_distance((0.5, 0.5), axes=case["crack_axes"])

    lower, upper = case["box"]
    counts = [max(1, round(args.base * a)) for a in case["aspect"]]
    knots = [np.linspace(0.0, 1.0, n + 1)[1:-1] for n in counts]
    primitive = (jx.primitives.rectangle(*lower, *upper) if case["dim"] == 2
                 else jx.primitives.cuboid(lower, upper))
    patch = primitive.elevate(args.degree).insert_knots(*knots)
    V = jx.refine_elements(jx.FunctionSpace(patch, vec=1), [])
    if args.band is not None:
        V = band_refine(V, args.levels, args.band * frac.ell,
                        axis=case["crack_axes"][1])
    else:
        V = seed_refine(V, crack, frac.ell, args.levels)
    h = 1.0 / (counts[0] * 2**args.levels)
    print(f"[{case['name']}] base {'x'.join(map(str, counts))}, {args.levels} levels: "
          f"{V.n_elems} elements to start, l/h = {frac.ell / h:.1f}")



    solver = StaggeredSolver(
        V, material, frac,
        dirichlet=case["dirichlet"], reaction=case["reaction"], crack=crack,
        adaptivity=Adaptivity(max_level=args.levels, dilate=args.dilate,
                              enabled=not args.fixed),
        linear=args.linear, verbose=True,
    )

    loads = load_path(case["schedule"], args.steps)
    frames = (Frames(os.path.join(out, "frames"), smax=args.smax)
              if args.movie else None)
    vtk = (VtkSeries(os.path.join(out, "vtu"), n=args.vtk_samples)
           if args.vtk else None)
    # Rendering is kept out of the reported time: the point of the run is what
    # the solve costs, and a movie is an output choice, not part of it.
    rows, drawing, t0 = [], 0.0, time.perf_counter()
    stop_below, peak_force, stopped = case["stop_below"], 0.0, None
    last, written = None, {"frame": None, "vtu": None}

    def emit(step, want_frame, want_vtu):
        """Render and write, skipping anything already done for this step."""
        start = time.perf_counter()
        if frames is not None and want_frame and written["frame"] != step.index:
            frames.draw(step, mesh_lines(step.space_d))
            written["frame"] = step.index
        if vtk is not None and want_vtu and written["vtu"] != step.index:
            vtk.write(step)
            written["vtu"] = step.index
        return time.perf_counter() - start
    for step in solver.run(loads, keep_solutions=args.movie or args.vtk):
        rows.append((step.load, step.force, step.n_elems, step.n_dofs, step.sweeps))
        peak_force = max(peak_force, abs(step.force))
        drawing += emit(step, step.index % args.every == 0 or step.index == 1,
                        step.index % args.vtk_every == 0 or step.index == 1)
        last = step
        if step.index % 50 == 0 or step.remeshed:
            print(f"    step {step.index:5d} u={step.load:.6f} F={step.force:8.1f} N "
                  f"sweeps={step.sweeps:3d} {step.n_elems:6d} elems "
                  f"[{time.perf_counter() - t0 - drawing:6.0f}s, {resident_gb():.1f} GB]")

        # The specimen has broken: the reaction is a small fraction of what it
        # carried at its peak, and nothing further happens. Drawn out of the
        # generator rather than out of the solver, which has no business
        # deciding when a problem is over.
        if has_broken(step.force, peak_force, stop_below):
            stopped = step.index
            print(f"    stopping at step {step.index}: reaction is "
                  f"{abs(step.force) / peak_force:.1%} of its peak, "
                  f"the crack has run through")
            break

    # The step a run ends on is the one worth having, and it lands on the output
    # cadence only by luck -- an early stop fires wherever the crack finishes.
    if last is not None:
        drawing += emit(last, True, True)
    elapsed = time.perf_counter() - t0 - drawing

    u = np.array([r[0] for r in rows])
    f = np.array([r[1] for r in rows])
    peak = int(np.argmax(f))
    st = solver.stats
    print(f"  peak {f[peak]:.1f} N at u={u[peak]:.5f}, final {f[-1]:.1f} N, "
          f"{len(rows)} steps{'' if stopped is None else f' (stopped at {stopped} of {len(loads)})'}"
          f", {solver.mesh.Vd.n_elems} elements at the end, {elapsed:.0f}s")
    print(f"  {st['sweeps']} staggered sweeps; {solver.remeshes} remeshes costing "
          f"{st['remesh']:.0f}s ({st['rebuild']:.0f}s rebuilding, "
          f"{st['transfer']:.0f}s transferring), {elapsed - st['remesh']:.0f}s solving")
    print(f"  displacement solves: {solver.linear_report()}")
    if frames is not None:
        print(f"  plus {drawing:.0f}s rendering {len(frames.paths)} frames")

    with open(os.path.join(out, "load_displacement.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["displacement", "force", "elements", "dofs", "sweeps"])
        w.writerows(rows)
    np.savez(os.path.join(out, "curve.npz"), u=u, f=f,
             elems=np.array([r[2] for r in rows]),
             sweeps=np.array([r[4] for r in rows]))

    if frames is not None:
        path = frames.encode(os.path.join(out, f"{case['name']}.mp4"), fps=args.fps)
        print(f"  wrote {path} ({len(frames.paths)} frames)")
    if vtk is not None:
        print(f"  wrote {vtk.close(case['name'] + '.pvd')} "
              f"({len(vtk.paths)} steps; open the .pvd to animate)")
    print(f"  wrote {out}/load_displacement.csv")


if __name__ == "__main__":
    main()
