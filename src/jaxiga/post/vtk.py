"""VTK output.

One dimension-generic writer replaces the legacy ``plot_sol2D``,
``plot_sol3D``, ``plot_sol2D_elast``, ``plot_sol2D_vector`` and their error
variants.

Each element is sampled on a uniform ``(n+1)^dim`` grid and emitted as a block
of linear cells. Points are duplicated between neighbouring elements, which
keeps the writer simple and renders correctly; the sampling density ``n``
controls how faithfully curved NURBS geometry is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

VTK_LINE = 3
VTK_QUAD = 9
VTK_HEXAHEDRON = 12


def _cells(n_e, n, dim):
    """Connectivity of the per-element sample grids."""
    per_elem = (n + 1) ** dim
    quads = []

    if dim == 1:
        base = np.arange(n)
        local = np.stack([base, base + 1], axis=1)
    elif dim == 2:
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        i, j = i.ravel(), j.ravel()
        c = lambda a, b: a + b * (n + 1)  # noqa: E731
        local = np.stack([c(i, j), c(i + 1, j), c(i + 1, j + 1), c(i, j + 1)], axis=1)
    elif dim == 3:
        i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        c = lambda a, b, d: a + b * (n + 1) + d * (n + 1) ** 2  # noqa: E731
        local = np.stack(
            [
                c(i, j, k), c(i + 1, j, k), c(i + 1, j + 1, k), c(i, j + 1, k),
                c(i, j, k + 1), c(i + 1, j, k + 1), c(i + 1, j + 1, k + 1), c(i, j + 1, k + 1),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"cannot write VTK for dim={dim}")

    for e in range(n_e):
        quads.append(local + e * per_elem)
    return np.concatenate(quads, axis=0)


@dataclass
class MixedCells:
    connectivity: np.ndarray
    offsets: np.ndarray
    types: np.ndarray
    owners: np.ndarray


def _mixed_cells(space, n, stride):
    from jaxiga.space.simplex import sample_cells
    connectivity, sizes, types, owners = [], [], [], []
    for e,kind in enumerate(np.asarray(space.element_types)):
        local = sample_cells(space.dim,n) if kind else _cells(1,n,space.dim)
        connectivity.extend((local+e*stride).ravel())
        sizes.extend([local.shape[1]]*len(local))
        types.extend([(5 if space.dim==2 else 10) if kind else (9 if space.dim==2 else 12)]*len(local))
        owners.extend([e]*len(local))
    return MixedCells(np.asarray(connectivity),np.cumsum(sizes),np.asarray(types,dtype='uint8'),np.asarray(owners))


def _grid(space, n):
    """Sample grid for a space: basis, VTK coordinates, connectivity, cell type."""
    basis = evaluate(space, P.grid(space, n + 1))
    x = np.asarray(basis.x).reshape(-1, basis.x.shape[-1])
    coords = [np.ascontiguousarray(x[:, d]) for d in range(space.dim)]
    while len(coords) < 3:
        coords.append(np.zeros(len(x)))
    if space.cell_type == "mixed":
        cells = _mixed_cells(space,n,basis.n_q)
        return basis,coords,cells,cells.types
    if space.cell_type == "simplex":
        from jaxiga.space.simplex import sample_cells
        local = sample_cells(space.dim, n)
        conn = np.concatenate([local + e * basis.n_q for e in range(basis.n_elems)])
        return basis, coords, conn, {2: 5, 3: 10}[space.dim]
    conn = _cells(basis.n_elems, n, space.dim)
    return basis, coords, conn, {1: VTK_LINE, 2: VTK_QUAD, 3: VTK_HEXAHEDRON}[space.dim]


def _require_pyevtk():
    try:
        from pyevtk.hl import unstructuredGridToVTK
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "writing VTK needs pyevtk; install it with `pip install pyevtk` "
            "or `pip install jaxiga[post]`"
        ) from exc
    return unstructuredGridToVTK


def _strip_suffix(path):
    path = str(path)
    for suffix in (".vtu", ".vtk"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def write_vtu(space, path: str, *, n: int = 4, point_data=None, cell_data=None,
              dtype=np.float32):
    """Write named fields on a space's sample grid to a ``.vtu`` file.

    The general form of :func:`write_vtk`, for when the fields to write do not
    all come from one :class:`~jaxiga.post.solution.Solution` -- a phase-field
    step, for instance, carries a displacement and a damage field that live on
    different spaces over the same mesh, and belong in one file so that a viewer
    can show them together.

    ``point_data`` values are arrays over the sample points, shaped
    ``(n_elems, n_points_per_elem)`` or flat; a trailing component axis is split
    into ``name_0``, ``name_1``, ... ``cell_data`` values are per *element* and
    are repeated across the cells each element is subdivided into, which is what
    makes a refinement level or an element index paintable.

    Fields are written as ``dtype``, single precision by default: these files
    are for looking at, and a load-stepping run writes one per step, so halving
    them is worth more than digits no viewer will show. Pass ``np.float64`` if
    the file is going somewhere that will compute with it.
    """
    unstructured_grid_to_vtk = _require_pyevtk()
    _, coords, conn, cell_type = _grid(space, n)
    owners = conn.owners if isinstance(conn,MixedCells) else np.repeat(np.arange(space.n_elems),len(conn)//space.n_elems)

    def flatten(name, value, into):
        if isinstance(value,tuple) and len(value)==3:
            into[name] = tuple(np.ascontiguousarray(np.asarray(v).ravel().astype(dtype)) for v in value)
            return
        value = np.asarray(value)
        if value.ndim > 1 and value.shape[-1] > 1 and value.size != len(coords[0]):
            for c in range(value.shape[-1]):
                into[f"{name}_{c}"] = np.ascontiguousarray(
                    value[..., c].reshape(-1).astype(dtype)
                )
        else:
            into[name] = np.ascontiguousarray(value.reshape(-1).astype(dtype))

    points, cells = {}, {}
    for name, value in (point_data or {}).items():
        flatten(name, value, points)
    for name, value in (cell_data or {}).items():
        flatten(name, np.asarray(value)[owners], cells)

    if isinstance(conn,MixedCells):
        connectivity,offsets,types = conn.connectivity,conn.offsets,conn.types
    else:
        connectivity,offsets = conn.ravel(),np.arange(1,len(conn)+1)*conn.shape[1]
        types = np.full(len(conn),cell_type,dtype="uint8")
    stem = _strip_suffix(path)
    unstructured_grid_to_vtk(
        stem,
        *coords,
        connectivity=np.ascontiguousarray(connectivity),
        offsets=np.ascontiguousarray(offsets),
        cell_types=types,
        pointData=points or None,
        cellData=cells or None,
    )
    return stem + ".vtu"


def write_series(paths, times, path: str):
    """Tie a sequence of ``.vtu`` files to times, as a ParaView ``.pvd``.

    Opening the collection instead of the individual files is what makes a run
    play as an animation rather than a pile of snapshots. Paths are stored
    relative to the collection, so the directory stays movable.
    """
    import os
    import xml.etree.ElementTree as ET

    path = str(path)
    if not path.endswith(".pvd"):
        path += ".pvd"
    root = ET.Element("VTKFile", type="Collection", version="0.1",
                      byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    base = os.path.dirname(os.path.abspath(path))
    for file, time in zip(paths, times):
        ET.SubElement(
            collection,
            "DataSet",
            timestep=repr(float(time)),
            group="",
            part="0",
            file=os.path.relpath(os.path.abspath(str(file)), base),
        )
    ET.ElementTree(root).write(path, xml_declaration=True, encoding="utf-8")
    return path


def write_vtk(sol, path: str, n: int = 10, fields=(), dtype=np.float32, cell_fields=()):
    """Write a solution to a ``.vtu`` file.

    Parameters
    ----------
    sol : Solution
    path : str
        Output path; a ``.vtu`` suffix is added if missing.
    n : int
        Sampling subdivisions per element and direction.
    fields : sequence of str
        Derived field names registered by the problem, e.g. ``"von_mises"``.
    dtype : numpy dtype
        Precision of the written fields; see :func:`write_vtu`.
    cell_fields : sequence of str
        Volume-weighted averages computed at interior quadrature points and
        stored as cell data. Avoids endpoint derivative evaluation at collapsed
        CAD edges; point coordinates and displacement still use the exact map.
    """
    space = sol.space
    if space.collapse_degenerate and fields:
        raise ValueError("pointwise endpoint derivatives are singular on collapsed boundaries; use cell_fields instead")
    basis, *_ = _grid(space, n)
    vals = np.asarray(sol.at(basis=basis)).reshape(-1, space.vec)

    point_data = {}
    if space.vec == 1:
        point_data["solution"] = vals[:, 0]
    else:
        for c, name in enumerate(["u", "v", "w"][: space.vec]):
            point_data[name] = vals[:, c]

    if space.vec in (2,3):
        point_data["displacement"] = tuple(vals[:,i] if i<space.vec else np.zeros(len(vals)) for i in range(3))

    for name in fields:
        f = np.asarray(sol.field(name, basis=basis))
        f = f.reshape(f.shape[0] * f.shape[1], -1)
        if f.shape[1] == 1:
            point_data[name] = f[:, 0]
        else:
            for c in range(f.shape[1]):
                point_data[f"{name}_{c}"] = f[:, c]

    cell_data = {"element_id":np.arange(space.n_elems)}
    cell_data.update({name: np.asarray(sol.cell_average(name)) for name in cell_fields})
    return write_vtu(space, path, n=n, point_data=point_data,
                     cell_data=cell_data, dtype=dtype)
