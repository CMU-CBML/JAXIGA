"""Small Matplotlib mesh/field plots for tensor, simplex, and mixed Bezier cells."""

from itertools import combinations

import numpy as np

from jaxiga.geometry.mixed import interfaces
from jaxiga.geometry.nurbs import evaluate_patch
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate
from jaxiga.space.simplex import face_points, lattice, sample_cells


def plot_mesh_field(solution, field=None, n=7, *, field_location="point"):
    """Mesh and field side by side; 3D plots show exterior faces only.

    Uses the undeformed geometry, including tensor patches with multiple spans.
    Separate samples retain physical derivative jumps at cell interfaces.
    Returns the figure and its two plotting axes.
    ``field_location="cell"`` displays volume-weighted averages from interior
    quadrature, so singular derivatives at collapsed CAD edges are not sampled.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from jaxiga.post.vtk import _cells
    from jaxiga.geometry.simplex import SimplexPatch
    from jaxiga.geometry.gmsh_geometry import tensor_grid
    from jaxiga.geometry.mixed import corners

    space = solution.space
    dim = space.dim
    if dim not in (2, 3) or space.patches is None:
        raise ValueError("mesh/field plotting requires 2D or 3D cell patches")
    if field_location not in {"point", "cell"}:
        raise ValueError("field_location must be 'point' or 'cell'")
    if space.collapse_degenerate and field and field_location == "point":
        raise ValueError(
            "collapsed boundaries require field_location='cell' for derived fields"
        )
    if n < 2:
        raise ValueError("mesh/field plotting requires at least two samples per edge")
    projection = {"projection": "3d"} if dim == 3 else {}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6 if dim == 3 else 5),
        subplot_kw=projection,
        constrained_layout=dim == 2,
    )
    if dim == 3:
        # Matplotlib's constrained layout misses some projected 3D labels.
        fig.subplots_adjust(left=0.04, right=0.94, bottom=0.15, top=0.92, wspace=0.12)
    patches = list(space.patches)
    if space.cell_type == "tensor":
        from jaxiga.geometry.refinement import bezier_cells

        patches = bezier_cells(patches)
    if space.n_elems != len(patches):
        raise ValueError(
            "mesh/field plotting requires ordinary tensor or Bezier cell spaces"
        )
    cell_values = None
    if field_location == "cell":
        cell_values = np.asarray(solution.cell_average(field)).reshape(
            space.n_elems, -1
        )
        cell_values = (
            cell_values[:, 0]
            if cell_values.shape[1] == 1
            else np.linalg.norm(cell_values, axis=1)
        )
    if space.cell_type == "mixed":
        _, faces = interfaces(patches)
    else:
        from jaxiga.geometry.mixed import facets

        exterior = {
            (int(e), int(s))
            for b in space.boundaries.values()
            for e, s in zip(b.elems, b.sides)
        }
        faces = [f for f in facets(patches) if (f.cell, f.code) in exterior]
    edges = []
    if dim == 2:
        # All cell edges, including interior interfaces.
        records = []
        for e, p in enumerate(patches):
            ref, _ = corners(p)
            for a, b in combinations(range(len(ref)), 2):
                if (
                    isinstance(p, SimplexPatch)
                    or np.count_nonzero(ref[a] != ref[b]) == 1
                ):
                    records.append((e, ref[a], ref[b]))
    else:
        records = []
        for face in faces:
            for a, b in combinations(range(len(face.reference)), 2):
                ra, rb = face.reference[[a, b]]
                if len(face.vertices) == 3 or np.count_nonzero(ra != rb) == 1:
                    records.append((face.cell, ra, rb))
    t = np.linspace(0, 1, n)[:, None]
    for e, a, b in records:
        edges.append(evaluate_patch(patches[e], (1 - t) * a + t * b))
    axes[0].add_collection(
        (Line3DCollection if dim == 3 else LineCollection)(
            edges, colors="k", linewidths=0.5
        )
    )
    polygons, colors = [], []
    if dim == 2:
        ps = P.grid(space, n)
        basis = evaluate(space, ps)
        values = (
            np.repeat(cell_values[:, None], basis.n_q, axis=1)
            if cell_values is not None
            else (
                solution.field(field, basis=basis)
                if field
                else solution.at(basis=basis)
            )
        )
        values = np.asarray(values).reshape(basis.n_elems, basis.n_q, -1)
        values = (
            values[..., 0] if values.shape[-1] == 1 else np.linalg.norm(values, axis=-1)
        )
        for e, p in enumerate(patches):
            if isinstance(p, SimplexPatch):
                local = sample_cells(2, n - 1)
            else:
                q = _cells(1, n - 1, 2)
                local = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
            xy = np.asarray(basis.x[e])
            triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], local)
            axes[1].tripcolor(
                triangulation,
                values[e],
                shading="gouraud",
                vmin=float(values.min()),
                vmax=float(values.max()),
            )
        scalar = plt.cm.ScalarMappable(
            norm=plt.Normalize(values.min(), values.max()), cmap="viridis"
        )
    else:
        for face in faces:
            patch = patches[face.cell]
            simplex = isinstance(patch, SimplexPatch)
            if simplex:
                q = lattice(2, n - 1)
                ref = face_points(3, face.code, q)
                local = sample_cells(2, n - 1)
            else:
                q = tensor_grid([np.linspace(0, 1, n)] * 2)
                d, end = divmod(face.code, 2)
                ref = np.full((len(q), 3), end, dtype=float)
                ref[:, np.arange(3) != d] = q
                quads = _cells(1, n - 1, 2)
                local = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
            if space.cell_type != "simplex":
                ref = 2 * ref - 1
            if cell_values is not None:
                local_ref = (
                    ref if simplex and space.cell_type == "simplex" else (ref + 1) / 2
                )
                polygons.extend(evaluate_patch(patch, local_ref)[local])
                colors.extend([cell_values[face.cell]] * len(local))
                continue
            ps = P.PointSet(np.array([face.cell]), ref, tag="grid")
            basis = evaluate(space, ps)
            value = np.asarray(
                solution.field(field, basis=basis)
                if field
                else solution.at(basis=basis)
            ).reshape(len(ref), -1)
            value = (
                value[:, 0] if value.shape[1] == 1 else np.linalg.norm(value, axis=1)
            )
            polygons.extend(np.asarray(basis.x[0])[local])
            colors.extend(value[local].mean(axis=1))
        scalar = plt.cm.ScalarMappable(
            norm=plt.Normalize(min(colors), max(colors)), cmap="viridis"
        )
        axes[1].add_collection3d(
            Poly3DCollection(
                polygons,
                facecolors=scalar.to_rgba(colors),
                edgecolors="none",
                linewidths=0,
                antialiased=False,
            )
        )
    xyz = np.concatenate(edges)
    for ax in axes:
        ax.set(xlabel="x", ylabel="y")
        if dim == 3:
            ax.set(
                zlabel="z",
                xlim=(xyz[:, 0].min(), xyz[:, 0].max()),
                ylim=(xyz[:, 1].min(), xyz[:, 1].max()),
                zlim=(xyz[:, 2].min(), xyz[:, 2].max()),
            )
            ax.set_box_aspect(np.ptp(xyz, axis=0))
            ax.view_init(elev=24, azim=-135)
        else:
            ax.autoscale()
            ax.set_aspect("equal")
    axes[0].set_title("Analysis mesh")
    label = field.replace("_", " ").title() if field else "Solution"
    if field_location == "cell":
        label = "Cell-average " + label
    axes[1].set_title(label)
    fig.colorbar(
        scalar,
        ax=axes[1],
        shrink=0.7,
        label=label,
    )
    return fig, axes
