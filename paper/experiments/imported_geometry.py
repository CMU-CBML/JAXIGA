"""Quad/hex CAD examples; cache mesh and stress samples for paper figures.

Run: python -m paper.experiments.imported_geometry
Set JAXIGA_RECOMPUTE_FIGURES=1 to rebuild the geometries and solve again.
"""

from pathlib import Path
import hashlib

import jax.numpy as jnp
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.tri import Triangulation

import jaxiga as jx
from jaxiga.examples.connecting_rod_3d import geometry as rod_geometry
from jaxiga.geometry.nurbs import evaluate_patch
from jaxiga.geometry.mixed import facets
from jaxiga.space.evaluation import evaluate
from jaxiga.space.pointset import PointSet
from ._common import FIGURES, RESULTS, figure_data, plt, provenance, run_example


def _pointwise(solution, elems, refs, chunk=256):
    """von Mises stress at the unit-square face samples ``refs`` of ``elems``."""
    values = []
    for start in range(0, len(elems), chunk):
        # the evaluator's reference cell is [-1, 1]^d, the Bezier cells' [0, 1]^d
        ps = PointSet(
            elems=elems[start : start + chunk],
            ref=jnp.asarray(2.0 * refs[start : start + chunk] - 1.0),
            weights=None,
            tag="grid",
        )
        basis = evaluate(solution.space, ps)
        values.append(np.asarray(solution.field("von_mises", basis=basis)))
    values = np.concatenate(values).reshape(len(elems), -1)
    assert np.isfinite(values).all()
    return values


# Uniform refinement levels of both geometries, as reported in the paper.
LEVELS = 3


def _samples(solution, n=7, pointwise=False):
    """Sample only tensor-cell edges/faces.

    Colors are physical cell averages by default. With ``pointwise`` the stress
    is instead evaluated at every face sample, so that it varies inside each
    element; samples are kept per face, and the field is therefore allowed to
    jump between elements wherever the discrete stress does.
    """
    space = solution.space
    assert space.cell_type == "tensor"
    patches = jx.bezier_cells(space.patches)
    averages = np.asarray(solution.cell_average("von_mises"))
    assert np.isfinite(averages).all()
    t = np.linspace(0, 1, n)
    grid = np.array([[u, v] for v in t for u in t])
    quads = np.array(
        [
            [i + n * j, i + 1 + n * j, i + 1 + n * (j + 1), i + n * (j + 1)]
            for j in range(n - 1)
            for i in range(n - 1)
        ]
    )
    edges, polygons, values = [], [], []
    if space.dim == 2:
        records = [(e, grid) for e in range(space.n_elems)]
    else:
        boundary = {
            (int(e), int(s))
            for b in space.boundaries.values()
            for e, s in zip(b.elems, b.sides)
        }
        records = []
        for face in facets(patches):
            if (face.cell, face.code) not in boundary:
                continue
            axis, end = divmod(face.code, 2)
            ref = np.full((len(grid), 3), end, dtype=float)
            ref[:, np.arange(3) != axis] = grid
            records.append((face.cell, ref))
    points = []
    for e, ref in records:
        xy = evaluate_patch(patches[e], ref)
        image = xy.reshape(n, n, space.dim)
        edges.extend([image[0], image[-1], image[:, 0], image[:, -1]])
        points.append(xy)
        polygons.extend(xy[quads])
        values.extend([averages[e]] * len(quads))
    if pointwise:
        elems = np.array([e for e, _ in records])
        refs = np.array([ref for _, ref in records])
        return {
            "edges": np.asarray(edges),
            "points": np.asarray(points),
            "stress": _pointwise(solution, elems, refs),
            "cell_average_max": np.asarray(averages.max()),
        }
    return {
        "edges": np.asarray(edges),
        "polygons": np.asarray(polygons),
        "stress": np.asarray(values),
    }


def _compute():
    source = Path(jx.__file__).parent / "examples/geometries/perforated_plate.geo"
    mesh = jx.read_gmsh(source, mesh_size="auto", geometry_order=3)
    initial_error = mesh.geometry_error
    mesh = mesh.refine(LEVELS)
    cases = [("plate", list(mesh)), ("rod", rod_geometry(refinements=LEVELS))]
    record = {
        "provenance": provenance(),
        "gmsh": mesh.gmsh_version,
        "geometry_order": 3,
        "mesh_size_request": "auto",
        "mesh_size": mesh.mesh_size,
        "refinement_levels": LEVELS,
        "plate_min_scaled_jacobian": mesh.min_scaled_jacobian,
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source, source.with_name("igapack_connecting_rod.mat"))
        },
        "plate_meshing_method": mesh.meshing_method,
        "plate_cad_error_before": initial_error,
        "plate_cad_error_after": mesh.geometry_error,
        "cases": {},
    }
    arrays = {}
    for name, patches in cases:
        dim = patches[0].dim
        space = jx.FunctionSpace(patches, vec=dim)
        traction = [10.0, 0.0] if dim == 2 else [0.0, 0.0, -1.0]
        problem = jx.LinearElasticity(
            space,
            plane="stress" if dim == 2 else "strain",
            dirichlet=[jx.DirichletBC([0.0] * dim, where="fixed")],
            neumann=[jx.Neumann(lambda x, n, p: jnp.array(traction), where="loaded")],
        )
        print(
            f"  {name}: {space.n_elems} tensor cells, {space.n_dofs} DOFs", flush=True
        )
        solution = jx.solve(
            problem,
            params={"E": 1e5 if dim == 2 else 4e5, "nu": 0.3},
            chunk=16,
            linear=jx.LinearOptions(method="cg", tol=1e-8, maxiter=50000),
        )
        if not bool(solution.stats["converged"]):
            raise RuntimeError(f"{name} failed convergence: {solution.stats}")
        # The rod is shown with the stress varying inside each element; the
        # plate keeps cell averages.
        samples = _samples(solution, pointwise=dim == 3)
        arrays.update({f"{name}_{key}": value for key, value in samples.items()})
        record["cases"][name] = {
            "patches": len(patches),
            "elements": space.n_elems,
            "dofs": space.n_dofs,
            "degree": list(space.degree),
            "relative_residual": float(solution.stats["relative_residual"]),
            "E": 1e5 if dim == 2 else 4e5,
            "nu": 0.3,
            "traction": traction,
            "solver": "cg",
            "preconditioner": "jacobi",
            "tol": 1e-8,
            "maxiter": 50000,
            "stress_sampling": "pointwise" if dim == 3 else "cell average",
            "von_mises_min": float(samples["stress"].min()),
            "von_mises_max": float(samples["stress"].max()),
        }
    return record, arrays


def _view():
    """Orthographic camera: viewing direction and the image-plane projection."""
    elev, azim = np.deg2rad([25.0, -65.0])
    view = np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )
    right = np.array([-np.sin(azim), np.cos(azim), 0.0])
    return view, np.column_stack([right, np.cross(view, right)])


def _plot_pointwise(ax, points, stress, norm):
    """Smoothly shaded boundary faces of a 3D solid, from per-face samples.

    Each face carries an ``n x n`` grid of samples. Its triangles interpolate
    the vertex values, and all triangles are drawn from back to front, like the
    flat-shaded polygons of the cell-average plot.
    """
    n_faces, n_pts, _ = points.shape
    n = int(round(np.sqrt(n_pts)))
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1))
    a = (i + n * j).ravel()
    cell = np.concatenate(
        [np.stack([a, a + 1, a + n + 1], 1), np.stack([a, a + n + 1, a + n], 1)]
    )
    triangles = (cell[None] + n_pts * np.arange(n_faces)[:, None, None]).reshape(-1, 3)
    view, projection = _view()
    flat = points.reshape(-1, 3)
    order = np.argsort((flat @ view)[triangles].mean(axis=1))
    xy = flat @ projection
    ax.tripcolor(
        Triangulation(xy[:, 0], xy[:, 1], triangles[order]),
        stress.ravel(),
        shading="gouraud",
        cmap="viridis",
        norm=norm,
        rasterized=True,
    )


def _plot(name, data):
    dim = data[f"{name}_edges"].shape[-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.09, top=0.88, wspace=0.12)
    edges, stress = data[f"{name}_edges"], data[f"{name}_stress"]
    pointwise = f"{name}_points" in data
    norm = colors.Normalize(0, stress.max())
    if pointwise:
        # Pointwise values peak sharply at re-entrant edges. The scale stops at
        # the 99.9th percentile, rounded down to a multiple of ten, and the
        # colorbar marks the clipped range; otherwise a few samples would set
        # the scale for the whole surface.
        norm = colors.Normalize(0, 10 * np.floor(np.percentile(stress, 99.9) / 10))
        _plot_pointwise(axes[1], data[f"{name}_points"], stress, norm)
        polygons = None
    else:
        polygons = data[f"{name}_polygons"]
    if dim == 3:
        # Orthographic camera, retaining physical aspect and sorting faces from
        # back to front. A 2D viewport avoids square Axes3D boxes clipping a rod.
        view, projection = _view()
        edges = edges @ projection
        if polygons is not None:
            order = np.argsort(polygons.mean(axis=1) @ view)
            polygons, stress = polygons[order] @ projection, stress[order]
    axes[0].add_collection(
        LineCollection(edges, colors="black", linewidths=0.35 if dim == 2 else 0.1)
    )
    if polygons is not None:
        surface = PolyCollection(
            polygons,
            facecolors=plt.cm.viridis(norm(stress)),
            edgecolors="none",
            antialiaseds=False,
            rasterized=True,
        )
        axes[1].add_collection(surface)
    pts = edges.reshape(-1, 2)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    for ax in axes:
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_axis_off()
    axes[0].set_title(
        "(a) Perforated plate" if dim == 2 else "(b) Connecting rod", loc="left"
    )
    axes[1].set_title(
        "von Mises stress" if pointwise else "Cell-average von Mises stress"
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
        cax=fig.add_axes([0.92, 0.2, 0.018, 0.6]),
        extend="max" if stress.max() > norm.vmax else "neither",
    )
    fig.savefig(Path(FIGURES) / f"imported_{name}.pdf", dpi=300)
    plt.close(fig)


def imported_geometry():
    record, arrays = figure_data("imported_geometry", _compute)
    for name in ("plate", "rod"):
        _plot(name, arrays)
    RESULTS["imported_geometry"] = record


if __name__ == "__main__":
    run_example("imported_geometry", imported_geometry)
