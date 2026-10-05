"""Geometry IO: geomdl JSON, G+Smo XML, MATLAB NURBS and optional Gmsh input.

The JSON/XML readers run in NumPy. Those formats describe the same object this package
calls a :class:`~jaxiga.geometry.nurbs.Patch` -- a tensor-product NURBS with
open knot vectors -- so the readers are mostly a matter of getting two
conventions right.

Control-point ordering
----------------------
This package flattens the control net with the **first parametric direction
fastest**. geomdl flattens with the *last* direction fastest (its
``ctrlpts`` list runs ``for u: for v:``), so its data is transposed on the way
in and out. G+Smo's ``<coefs>`` block already runs with the first direction
fastest and needs no transpose.

Weights
-------
Both formats optionally store weighted (homogeneous) control points. geomdl's
``ctrlptsw`` is weighted and its ``ctrlpts`` is not; G+Smo stores unweighted
coefficients with a separate ``<weights>`` block. Patches here hold unweighted
points and separate weights, so anything weighted is divided through on read.

Knot vectors are normalized to ``[0, 1]`` by ``Patch.create``, which changes
nothing geometrically.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from jaxiga.geometry.nurbs import Patch
from jaxiga.geometry.gmsh import GmshImportError, GmshMesh, read_gmsh  # noqa: F401


class GeometryFormatError(Exception):
    """A geometry file could not be interpreted."""


def read_matlab_nurbs(path, variables=None) -> list:
    """Read MATLAB NURBS-toolbox structs (``coefs``, ``knots``, ``order``).

    Returns patches in file order, or in the order of explicit variable names.
    Coefficients are homogeneous, with their coordinate index first and the
    first parametric direction fastest. No MATLAB code is executed. MATLAB
    v7.3/HDF5 files are unsupported; save geometry with ``save -v7`` instead.
    The original parametrization, including its orientation, is preserved.
    """
    from scipy.io import loadmat

    try:
        records = loadmat(path, simplify_cells=True)
    except (ValueError, NotImplementedError) as exc:
        raise GeometryFormatError(f"cannot read MATLAB geometry: {exc}") from exc
    if isinstance(variables, str):
        variables = [variables]
    if variables is None:
        variables = [name for name, value in records.items()
                     if isinstance(value, dict) and {"coefs", "knots", "order"} <= value.keys()]
    patches = []
    for name in variables:
        try:
            data = records[name]
            order = np.atleast_1d(data["order"]).astype(float)
            if not np.isfinite(order).all() or np.any(order < 2) or np.any(order != order.astype(int)):
                raise ValueError("orders must be integers at least two")
            degree = tuple(order.astype(int) - 1)
            if not 1 <= len(degree) <= 3:
                raise ValueError("only curves, surfaces and volumes are supported")
            knots = ([np.asarray(data["knots"], dtype=float)] if len(degree) == 1
                     else [np.asarray(k, dtype=float) for k in data["knots"]])
            sizes = tuple(len(k) - p - 1 for k, p in zip(knots, degree))
            if any(k.ndim != 1 or k.size < 2 or not np.isfinite(k).all()
                   or np.any(np.diff(k) < 0) or k[-1] <= k[0] for k in knots):
                raise ValueError("knots must be finite, nondecreasing and span a nonempty interval")
            coefs = np.asarray(data["coefs"], dtype=float)
            if len(knots) != len(degree) or coefs.shape != (4, *sizes):
                raise ValueError("coefs must have shape (4, *control_net_sizes)")
            if "number" in data and not np.array_equal(np.atleast_1d(data["number"]), sizes):
                raise ValueError("number disagrees with knots/order")
            H = coefs.reshape(4, -1, order="F").T
            if not np.isfinite(H).all() or np.any(H[:, -1] <= 0):
                raise ValueError("coefficients must be finite and weights positive")
            xyz = H[:, :3] / H[:, -1:]
            while xyz.shape[1] > len(degree) and np.all(xyz[:, -1] == 0):
                xyz = xyz[:, :-1]
            patches.append(Patch.create(knots, degree, xyz, H[:, -1]))
        except (KeyError, TypeError, ValueError) as exc:
            raise GeometryFormatError(f"invalid NURBS variable {name!r}: {exc}") from exc
    if not patches:
        raise GeometryFormatError(f"{path}: no NURBS structs found")
    return patches


def _reorder(points, n_per_dir, to_first_fastest: bool):
    """Convert a flat control net between the two flattening conventions.

    Built as an explicit permutation rather than a reshape chain, because the
    two orders differ by reversing the index axes and a silently transposed
    control net is the kind of bug that only shows up on a non-square grid.
    """
    n = int(np.prod(n_per_dir))
    if to_first_fastest:
        # source index runs last-fastest (C order); enumerate first-fastest
        order = np.arange(n).reshape(n_per_dir).ravel(order="F")
    else:
        order = np.arange(n).reshape(n_per_dir, order="F").ravel(order="C")
    return points[order]


# --------------------------------------------------------------------------
# geomdl JSON
# --------------------------------------------------------------------------


def _patch_from_geomdl_dict(data, labels=None) -> Patch:
    degree, knots, n_per_dir = [], [], []
    for suffix in ("u", "v", "w"):
        if f"degree_{suffix}" not in data:
            break
        degree.append(int(data[f"degree_{suffix}"]))
        knots.append(np.asarray(data[f"knotvector_{suffix}"], dtype=float))
        n_per_dir.append(int(data[f"size_{suffix}"]))
    if not degree:
        # geomdl curves use scalar keys
        if "degree" not in data:
            raise GeometryFormatError("no degree found; is this a geomdl shape record?")
        degree = [int(data["degree"])]
        knots = [np.asarray(data["knotvector"], dtype=float)]
        n_per_dir = [len(data["control_points"]["points"])]

    cp = data["control_points"]
    points = np.asarray(cp["points"], dtype=float)
    if points.shape[0] != int(np.prod(n_per_dir)):
        raise GeometryFormatError(
            f"control-point count {points.shape[0]} does not match the declared "
            f"sizes {tuple(n_per_dir)}"
        )
    weights = np.asarray(cp.get("weights", np.ones(len(points))), dtype=float)

    for d, (kv, n, p) in enumerate(zip(knots, n_per_dir, degree)):
        if len(kv) != n + p + 1:
            raise GeometryFormatError(
                f"direction {d}: knot vector of length {len(kv)} is inconsistent with "
                f"{n} control points of degree {p} (expected {n + p + 1})"
            )

    if len(n_per_dir) > 1:
        points = _reorder(points, n_per_dir, to_first_fastest=True)
        weights = _reorder(weights, n_per_dir, to_first_fastest=True)

    # Drop trailing all-zero coordinate columns. Both formats pad geometry out
    # to 3D, and carrying a column of zeros would make every geometry Jacobian
    # rectangular and singular. The parametric dimension is the floor, so a
    # surface never collapses to a curve even if it happens to lie in a
    # coordinate plane.
    keep = points.shape[1]
    while keep > len(degree) and np.allclose(points[:, keep - 1], 0.0):
        keep -= 1
    points = points[:, :keep]

    return Patch.create(
        knots=knots, degree=tuple(degree), ctrl_pts=points, weights=weights, labels=labels
    )


def read_geomdl_json(path, labels=None) -> list:
    """Read a geomdl JSON export; returns a list of patches.

    Accepts both the ``{"shape": {"data": [...]}}`` wrapper geomdl writes and a
    bare record or list of records.
    """
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "shape" in raw:
        records = raw["shape"].get("data", [])
        if isinstance(records, dict):
            records = [records]
    elif isinstance(raw, list):
        records = raw
    else:
        records = [raw]
    if not records:
        raise GeometryFormatError(f"{path}: no shape records found")
    return [_patch_from_geomdl_dict(r, labels) for r in records]


def write_geomdl_json(path, patches, dim_out: int = 3):
    """Write patches in geomdl's JSON layout.

    ``dim_out`` pads 2D control points with a zero z-component, which is what
    geomdl and most viewers expect of a surface.
    """
    if isinstance(patches, Patch):
        patches = [patches]

    records = []
    for patch in patches:
        n_per_dir = patch.n_cp_per_dir
        pts = np.asarray(patch.ctrl_pts, dtype=float)
        if pts.shape[1] < dim_out:
            pts = np.pad(pts, ((0, 0), (0, dim_out - pts.shape[1])))
        wts = np.asarray(patch.weights, dtype=float)

        if patch.dim > 1:
            pts = _reorder(pts, n_per_dir, to_first_fastest=False)
            wts = _reorder(wts, n_per_dir, to_first_fastest=False)

        record = {"control_points": {"points": pts.tolist(), "weights": wts.tolist()}}
        for d, suffix in zip(range(patch.dim), ("u", "v", "w")):
            record[f"degree_{suffix}"] = patch.degree[d]
            record[f"knotvector_{suffix}"] = list(patch.knots[d])
            record[f"size_{suffix}"] = n_per_dir[d]
        records.append(record)

    kind = {1: "curve", 2: "surface", 3: "volume"}[patches[0].dim]
    Path(path).write_text(
        json.dumps({"shape": {"type": kind, "count": len(records), "data": records}}, indent=1)
    )


# --------------------------------------------------------------------------
# G+Smo XML
# --------------------------------------------------------------------------


def _floats(text):
    return np.asarray([float(v) for v in text.split()], dtype=float)


def _read_gismo_geometry(node) -> Patch:
    knot_nodes = node.findall(".//KnotVector")
    if not knot_nodes:
        raise GeometryFormatError("no <KnotVector> under this <Geometry> element")

    degree = tuple(int(k.attrib["degree"]) for k in knot_nodes)
    knots = [_floats(k.text) for k in knot_nodes]
    n_per_dir = [len(kv) - p - 1 for kv, p in zip(knots, degree)]

    coefs_node = node.find(".//coefs")
    if coefs_node is None:
        raise GeometryFormatError("no <coefs> under this <Geometry> element")
    dim_phys = int(coefs_node.attrib.get("geoDim", 3))
    points = _floats(coefs_node.text).reshape(-1, dim_phys)

    weights_node = node.find(".//weights")
    if weights_node is not None:
        weights = _floats(weights_node.text)
    else:
        weights = np.ones(len(points))

    if points.shape[0] != int(np.prod(n_per_dir)):
        raise GeometryFormatError(
            f"{points.shape[0]} coefficients do not match the basis sizes "
            f"{tuple(n_per_dir)}"
        )
    # G+Smo already runs the first direction fastest, matching this package.
    return Patch.create(knots=knots, degree=degree, ctrl_pts=points, weights=weights)


def read_gismo_xml(path) -> list:
    """Read the ``<Geometry>`` elements of a G+Smo XML file.

    Handles ``TensorBSpline`` and ``TensorNurbs`` of any dimension, plus the
    1D ``BSpline``/``Nurbs`` spellings. ``MultiPatch`` containers are read as
    the list of geometries they reference.
    """
    root = ET.parse(Path(path)).getroot()
    nodes = root.findall(".//Geometry")
    if not nodes:
        raise GeometryFormatError(f"{path}: no <Geometry> elements")
    return [_read_gismo_geometry(n) for n in nodes]


def write_gismo_xml(path, patches):
    """Write patches as a G+Smo XML file.

    Rational patches are written as ``TensorNurbs<d>`` with a ``<weights>``
    block; patches with unit weights are written as ``TensorBSpline<d>``.
    """
    if isinstance(patches, Patch):
        patches = [patches]

    root = ET.Element("xml")
    for index, patch in enumerate(patches):
        dim = patch.dim
        wts = np.asarray(patch.weights, dtype=float)
        rational = not np.allclose(wts, 1.0)
        kind = "Nurbs" if rational else "BSpline"
        prefix = "Tensor" if dim > 1 else ""

        geo = ET.SubElement(
            root, "Geometry", {"type": f"{prefix}{kind}{dim}", "id": str(index)}
        )
        outer = ET.SubElement(geo, "Basis", {"type": f"{prefix}{kind}Basis{dim}"})
        holder = (
            ET.SubElement(outer, "Basis", {"type": f"TensorBSplineBasis{dim}"})
            if rational and dim > 1
            else outer
        )
        for d in range(dim):
            b = ET.SubElement(holder, "Basis", {"type": "BSplineBasis", "index": str(d)})
            kv = ET.SubElement(b, "KnotVector", {"degree": str(patch.degree[d])})
            kv.text = " ".join(repr(float(v)) for v in patch.knots[d])
        if rational:
            w = ET.SubElement(outer, "weights")
            w.text = "\n".join(repr(float(v)) for v in wts)

        pts = np.asarray(patch.ctrl_pts, dtype=float)
        coefs = ET.SubElement(geo, "coefs", {"geoDim": str(pts.shape[1])})
        coefs.text = "\n".join(" ".join(repr(float(v)) for v in row) for row in pts)

    ET.indent(root, space=" ")
    ET.ElementTree(root).write(Path(path), encoding="unicode", xml_declaration=True)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

# Spec-named aliases (section 2.12 of the architecture guide).
read_json = read_geomdl_json
read_gismo = read_gismo_xml
write_json = write_geomdl_json
write_gismo = write_gismo_xml

_READERS = {".json": read_geomdl_json, ".xml": read_gismo_xml,
            **{suffix: read_gmsh for suffix in (".geo", ".step", ".stp", ".msh")}}
_WRITERS = {".json": write_geomdl_json, ".xml": write_gismo_xml}


def read(path, **kwargs):
    """Read a sequence of patches; Gmsh input also carries mesh metadata."""
    suffix = Path(path).suffix.lower()
    if suffix not in _READERS:
        raise GeometryFormatError(
            f"unknown geometry format {suffix!r}; expected one of {sorted(_READERS)}"
        )
    return _READERS[suffix](path, **kwargs)


def write(path, patches, **kwargs):
    """Write patches to a file, choosing the writer by extension."""
    suffix = Path(path).suffix.lower()
    if suffix not in _WRITERS:
        raise GeometryFormatError(
            f"unknown geometry format {suffix!r}; expected one of {sorted(_WRITERS)}"
        )
    return _WRITERS[suffix](path, patches, **kwargs)
