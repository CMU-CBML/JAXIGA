"""Gmsh CAD/mesh input as conforming tensor-product or simplex patches.

Gmsh is optional. High-order Lagrange geometry is converted to Bernstein form.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations, product
from pathlib import Path

import numpy as np

from jaxiga.geometry.nurbs import Patch, side_name
from jaxiga.geometry.gmsh_geometry import bezier_jacobians, lagrange_to_bernstein, spline_quality


class GmshImportError(ValueError):
    """Unsupported mesh, ambiguous boundary labels, or unacceptable geometry."""


class _MeshingFailure(GmshImportError):
    """Generation or cell-quality failure that can justify a CAD remesh."""


@dataclass
class GmshMesh(Sequence):
    """Sequence of patches plus source mesh labels and quality diagnostics.

    ``cell_groups`` maps physical names to patch indices. ``boundary_groups``
    maps names to (patch index, parametric side) pairs, including tagged internal
    interfaces. Only exterior sides become FunctionSpace boundary conditions.
    ``physical_names`` retains the original (dimension, physical tag) mapping.
    Quality values refer to the imported or CAD-refitted polynomial geometry.
    ``meshing_method``, ``subdivided`` and ``fallback_reason`` describe how
    the accepted mesh was generated and whether a retry was necessary.
    """

    patches: list
    element_tags: np.ndarray
    cell_groups: dict
    boundary_groups: dict
    physical_names: dict
    scaled_jacobians: np.ndarray
    jacobian_lower_bounds: np.ndarray
    source: str
    gmsh_version: str
    geometry_representation: str = "polynomial"
    reconstruction_report: tuple = ()
    meshing_method: str = "existing_mesh"
    subdivided: bool = False
    fallback_reason: str | None = None
    mesh_size: float | None = None
    geometry_error: float | None = None
    sizing_history: tuple = ()
    _cad_constraints: tuple = field(default=(), repr=False)
    _corner_tags: np.ndarray | None = field(default=None, repr=False)
    _source_digest: str | None = field(default=None, repr=False)
    _min_quality: float = field(default=0.05, repr=False)
    _plane_z: float = field(default=0.0, repr=False)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        return self.patches[index]

    @property
    def min_scaled_jacobian(self):
        return float(self.scaled_jacobians.min())

    @property
    def cell_type(self):
        from jaxiga.geometry.simplex import SimplexPatch
        families={isinstance(p,SimplexPatch) for p in self.patches}
        return "mixed" if len(families)>1 else ("simplex" if True in families else "tensor")

    def _update_geometry(self, patches, to_cad):
        if self.cell_type in {"simplex", "mixed"}:
            if to_cad and self.geometry_representation != "rational_cad":
                raise NotImplementedError(
                    "CAD refitting is not implemented for simplices; use elevate(..., to_cad=False) "
                    "to preserve geometry, or reimport CAD at a higher geometry_order")
            from dataclasses import replace
            return replace(self, patches=patches)
        from jaxiga.geometry.gmsh_cad import update_geometry
        return update_geometry(self, patches, to_cad)

    def refine(self, n=1, *, to_cad=True):
        """Bisect knot spans and refit new boundary samples to the source CAD.

        Use ``to_cad=False`` for ordinary, shape-preserving knot insertion.
        CAD refitting changes the geometry; rebuild the FunctionSpace afterward.
        """
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("n must be a non-negative integer")
        return self._update_geometry([p.refine(n) for p in self], to_cad) if n else self

    def elevate(self, target_degree, *, to_cad=True):
        """Raise spline degree and optionally refit boundary samples to CAD."""
        return self._update_geometry([p.elevate(target_degree) for p in self], to_cad)

    def insert_knots(self, *knots_per_dir, to_cad=True):
        """Insert knots on every patch, checking that shared traces still match."""
        return self._update_geometry([p.insert_knots(*knots_per_dir) for p in self], to_cad)


def _grid(values, dim):
    return np.stack(np.meshgrid(*[values] * dim, indexing="ij"), axis=-1).reshape(
        -1, dim, order="F"
    )


def _elements(gmsh, dim, entity=-1):
    types, tags, nodes = gmsh.model.mesh.getElements(dim, entity)
    return [(int(kind), np.asarray(ids, dtype=np.int64), np.asarray(conn, dtype=np.int64))
            for kind, ids, conn in zip(types, tags, nodes) if len(ids)]


def _convert(gmsh, dim, source, min_scaled_jacobian):
    blocks = _elements(gmsh, dim)
    if not blocks:
        raise _MeshingFailure(f"no {dim}D elements were generated")
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    nodes = dict(zip(map(int, node_tags), np.asarray(coordinates).reshape(-1, 3)))
    if dim == 2:
        xyz = np.asarray(coordinates).reshape(-1, 3)
        scale = max(float(np.ptp(xyz, axis=0).max()), 1e-30)
        if np.ptp(xyz[:, 2]) > 1e-10 * scale:
            raise GmshImportError("2D meshes must lie in a plane parallel to XY; shells are unsupported")
    tags, corners, controls, degrees, qualities, bounds, face_nodes = [], [], [], [], [], [], []
    for kind, ids, conn in blocks:
        try:
            degree, transform, corner_order, reference = lagrange_to_bernstein(gmsh, kind)
        except ValueError as exc:
            raise GmshImportError(str(exc)) from exc
        cells = conn.reshape(len(ids), -1)
        if any(len(set(cell)) != len(cell) for cell in cells):
            raise GmshImportError("a cell has repeated vertex tags")
        try:
            xyz = np.array([[nodes[int(tag)] for tag in cell] for cell in cells])
        except KeyError as exc:
            raise GmshImportError(f"cell references missing node {exc}") from exc
        if not np.isfinite(xyz).all():
            raise GmshImportError("non-finite mesh coordinates")
        if dim == 2:
            scale = max(float(np.ptp(xyz.reshape(-1, 3), axis=0).max()), 1e-30)
            if np.ptp(xyz[..., 2]) > 1e-10 * scale:
                raise GmshImportError("2D meshes must lie in a plane parallel to XY; shells are unsupported")
            xyz = xyz[..., :2]
        points = np.einsum("ij,ejd->eid", transform, xyz)
        primary = cells[:, corner_order].copy()
        det, scaled = bezier_jacobians(points, degree)
        reverse = np.all(det < 0, axis=1)
        # Reverse u in the full Bernstein net, not only its corners.
        permutation = np.arange((degree + 1) ** dim).reshape((degree + 1,) * dim, order="F")
        permutation = permutation[::-1].ravel(order="F")
        points[reverse] = points[reverse][:, permutation]
        primary[reverse] = primary[reverse][:, np.arange(2 ** dim) ^ 1]
        det, scaled = bezier_jacobians(points, degree)
        lower = det.min(axis=1)
        if degree > 1 or dim == 3:
            # Validate the converted polynomial itself, using the same bounds
            # as CAD refitting instead of relying on a mesher-specific metric.
            for i, control in enumerate(points):
                patch = Patch.create([[0] * (degree + 1) + [1] * (degree + 1)] * dim,
                                     (degree,) * dim, control)
                try:
                    _, bound = spline_quality(patch)
                except ValueError as exc:
                    raise _MeshingFailure(f"element {ids[i]}: {exc}") from exc
                lower[i] = min(lower[i], bound)
        quality = scaled.min(axis=1)
        bad = (~np.isfinite(lower) | ~np.isfinite(quality) | (lower <= 0)
               | (quality < min_scaled_jacobian))
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise _MeshingFailure(
                f"element {ids[i]} is inverted, degenerate or too distorted: "
                f"minimum Jacobian={lower[i]:.3g}, sampled scaled Jacobian={quality[i]:.3g} "
                f"(required >= {min_scaled_jacobian:g}); improve or remesh the geometry"
            )
        for cell, flipped in zip(cells, reverse):
            refs = reference.copy()
            if flipped:
                refs[:, 0] *= -1
            face_nodes.append({(d, end): tuple(sorted(cell[np.isclose(refs[:, d], 2 * end - 1)]))
                               for d in range(dim) for end in (0, 1)})
        tags.extend(ids)
        corners.extend(primary)
        controls.extend(points)
        degrees.extend([degree] * len(ids))
        qualities.extend(quality)
        bounds.extend(lower)
    tags, cells = np.asarray(tags), np.asarray(corners)
    if len({tuple(sorted(cell)) for cell in cells}) != len(cells):
        raise GmshImportError("duplicate cells in the mesh")
    # Face identities come from node tags, not a coordinate tolerance.
    faces = defaultdict(list)
    corner_grid = _grid([0, 1], dim)
    traces = {}
    for i, cell in enumerate(cells):
        for d in range(dim):
            for end in (0, 1):
                key = tuple(sorted(cell[corner_grid[:, d] == end]))
                faces[key].append((i, side_name(d, end)))
                if key in traces and traces[key] != face_nodes[i][d, end]:
                    raise GmshImportError("nonconforming high-order nodes on a shared face")
                traces[key] = face_nodes[i][d, end]
    if any(len(owners) > 2 for owners in faces.values()):
        raise GmshImportError("non-manifold mesh: more than two cells share a face")

    physical_names, cell_groups, boundary_groups = {}, defaultdict(set), defaultdict(set)
    index = {int(tag): i for i, tag in enumerate(tags)}
    labels = [{} for _ in cells]
    for group_dim, physical_tag in gmsh.model.getPhysicalGroups():
        name = (gmsh.model.getPhysicalName(group_dim, physical_tag)
                or f"physical_{group_dim}_{physical_tag}")
        physical_names[group_dim, physical_tag] = name
        entities = gmsh.model.getEntitiesForPhysicalGroup(group_dim, physical_tag)
        for entity in entities:
            for kind, ids, conn in _elements(gmsh, group_dim, int(entity)):
                if group_dim == dim:
                    cell_groups[name].update(index[int(tag)] for tag in ids)
                elif group_dim == dim - 1:
                    _, _, _, count, _, primary = gmsh.model.mesh.getElementProperties(kind)
                    if primary != 2 ** (dim - 1):
                        raise GmshImportError(f"physical boundary {name!r} has unsupported element type {kind}")
                    for face in conn.reshape(-1, count)[:, :primary]:
                        key = tuple(sorted(face))
                        if key not in faces:
                            raise GmshImportError(f"physical boundary {name!r} does not match a cell face")
                        for i, side in faces[key]:
                            boundary_groups[name].add((i, side))
                            previous = labels[i].get(side)
                            if previous is not None and previous != name:
                                raise GmshImportError(
                                    f"side belongs to overlapping physical groups {previous!r} and {name!r}; "
                                    "a patch side supports one boundary name"
                                )
                            labels[i][side] = name
    patches = [Patch.create([[0] * (p + 1) + [1] * (p + 1)] * dim,
                            (p,) * dim, xyz, labels=label)
               for xyz, p, label in zip(controls, degrees, labels)]
    mesh = GmshMesh(
        patches, tags, {k: tuple(sorted(v)) for k, v in cell_groups.items()},
        {k: tuple(sorted(v)) for k, v in boundary_groups.items()}, physical_names,
        np.asarray(qualities), np.asarray(bounds), str(source), gmsh.__version__,
    )
    mesh._corner_tags = cells
    mesh._min_quality = min_scaled_jacobian
    if dim == 2:
        mesh._plane_z = float(np.asarray(coordinates).reshape(-1, 3)[0, 2])
    # Retain classification on *all* CAD curves/faces, including untagged ones.
    supports = defaultdict(list)
    for i, cell in enumerate(cells):
        for count in range(1, dim):
            for axes in combinations(range(dim), count):
                for ends in product((0, 1), repeat=count):
                    mask = np.all(corner_grid[:, axes] == ends, axis=1)
                    key = tuple(sorted(cell[mask]))
                    supports[key].append((i, tuple(zip(axes, ends))))
    constraints = [set() for _ in cells]
    for entity_dim, entity_tag in gmsh.model.getEntities():
        if not 0 < entity_dim < dim:
            continue
        for kind, _, conn in _elements(gmsh, entity_dim, entity_tag):
            _, _, _, count, _, primary = gmsh.model.mesh.getElementProperties(kind)
            for cell in conn.reshape(-1, count)[:, :primary]:
                for i, fixed in supports.get(tuple(sorted(cell)), ()):
                    constraints[i].add((entity_dim, entity_tag, fixed))
    mesh._cad_constraints = tuple(tuple(sorted(c, reverse=True)) for c in constraints)
    return mesh


_QUAD_METHODS = (
    ("quasi_structured", 11, 1),
    ("blossom", 8, 1),
    ("frontal_delaunay", 6, 0),
)


def _generate(gmsh, dim, mesh_size, algorithm, recombination, cell_type="tensor"):
    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)
    if mesh_size is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    if cell_type == "simplex":
        gmsh.model.mesh.removeConstraints(gmsh.model.getEntities(2) + gmsh.model.getEntities(3))
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Recombine3DAll", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    elif cell_type == "mixed":
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
    elif dim == 2:
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", recombination)
        for _, tag in gmsh.model.getEntities(2):
            gmsh.model.mesh.setAlgorithm(2, tag, algorithm)
    try:
        gmsh.model.mesh.generate(dim)
    except Exception as exc:
        raise _MeshingFailure(f"Gmsh generation failed: {exc}") from exc


def _finish_mesh(gmsh, dim, path, subdivide, min_scaled_jacobian, geometry_order, cell_type="tensor"):
    if cell_type in {"simplex", "mixed"}:
        from jaxiga.geometry.gmsh_simplex import convert_simplex
        if cell_type == "mixed":
            from jaxiga.geometry.gmsh_mixed import convert_mixed as convert_simplex
        if path.suffix.lower() != ".msh":
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
            try:
                gmsh.model.mesh.setOrder(geometry_order or 1)
                if dim == 2 and geometry_order is not None and geometry_order > 1:
                    gmsh.model.mesh.optimize("HighOrder")
            except Exception as exc:
                raise _MeshingFailure(f"Gmsh high-order geometry failed: {exc}") from exc
        return convert_simplex(gmsh, dim, path, min_scaled_jacobian)
    blocks = _elements(gmsh, dim)
    family = "Quadrilateral" if dim == 2 else "Hexahedron"
    mixed = any(not gmsh.model.mesh.getElementProperties(kind)[0].startswith(family)
                for kind, _, _ in blocks)
    if mixed and any(gmsh.model.mesh.getElementProperties(kind)[2] != 1 for kind, _, _ in blocks):
        raise GmshImportError("high-order mixed/simplex meshes cannot be subdivided without losing geometry")
    split = subdivide and mixed
    if split:
        allowed = {2, 3} if dim == 2 else {4, 5, 6, 7}
        if any(kind not in allowed for kind, _, _ in blocks):
            raise GmshImportError("unsupported cell family for quad/hex subdivision")
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1 if dim == 2 else 2)
        if path.suffix.lower() == ".msh":
            # Without CAD, project nothing: default projection can collapse
            # newly created points in standalone meshes.
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        try:
            gmsh.model.mesh.refine()
        except Exception as exc:
            raise _MeshingFailure(f"Gmsh subdivision failed: {exc}") from exc
    if path.suffix.lower() != ".msh":
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        try:
            gmsh.model.mesh.setOrder(geometry_order or 1)
            if dim == 2 and geometry_order is not None and geometry_order > 1:
                gmsh.model.mesh.optimize("HighOrder")
        except Exception as exc:
            raise _MeshingFailure(f"Gmsh high-order geometry failed: {exc}") from exc
    mesh = _convert(gmsh, dim, path, min_scaled_jacobian)
    mesh.subdivided = split
    return mesh


def read_gmsh(path, *, mesh_size=None, subdivide=True, min_scaled_jacobian=0.05,
              quad_algorithm="quasi_structured", geometry_order=None,
              geometry_tolerance=None, max_sizing_steps=6, verbose=False, cell_type="tensor", rational_geometry=False):
    """Mesh a .geo/.step/.stp file, or read .msh, as analysis patches.

    ``rational_geometry=True`` reconstructs conic curves and aligned cylindrical
    faces from CAD, validating rational Jacobians. Other CAD families are
    rejected. Exact rational refinement preserves geometry without projection.
    ``cell_type="mixed"`` retains quad/triangle or hex/tetra blocks and couples
    matching traces (including a quad face covered by two triangles). Slave
    traces are elevated/reconstructed when needed; pyramids/prisms are rejected.

    ``cell_type="simplex"`` imports triangles/tetrahedra directly as
    SimplexPatch cells (complete polynomial degrees 1--4). CAD uses ordinary
    unstructured Frontal-Delaunay/Delaunay meshing without recombination or
    tensor subdivision, ignoring transfinite/recombination constraints.
    Mixed simplex/tensor cells are rejected in this mode. Degree elevation
    with ``to_cad=False`` preserves geometry; simplex CAD refitting and knot
    refinement are not supported. Automatic CAD sizing is supported.

    Returns a GmshMesh, usable as a sequence of Patch or SimplexPatch objects. Physical curve
    names in 2D and surface names in 3D label patch sides; physical domain
    groups are retained in ``cell_groups``. Unnamed groups get stable names
    ``physical_<dimension>_<tag>``. Overlapping boundary names are rejected.

    In the default tensor mode, CAD topology is meshed at order one, then curved to ``geometry_order``
    (default 1). Existing .msh polynomial quad/hex geometry is preserved exactly
    through a Lagrange-to-Bernstein transformation, including serendipity cells.
    Geometry degrees 1 through 8 are supported. ``mesh_size`` sets a uniform
    target size, otherwise the file/Gmsh sizing settings apply. In 2D,
    Quasi-structured Quad is tried first, followed by Frontal-Delaunay for
    Quads with Blossom recombination, then ordinary Frontal-Delaunay with
    simple recombination. Generation or quality failure triggers the next
    method in a fresh session. ``quad_algorithm`` selects the first method:
    "quasi_structured" (default), "blossom", or "frontal_delaunay".
    Quasi-structured meshing can adjust transfinite counts; select "blossom"
    to retain prescribed counts. In 3D, file/Gmsh meshing settings
    apply (unstructured Delaunay tetrahedra by default).
    If needed, ``subdivide=True`` splits a linear mixed mesh to all
    quads/hexes. Set it False to require the mesh to already have these types.
    High-order mixed/simplex .msh input is rejected, never silently linearized.
    ``meshing_method``, ``subdivided`` and ``fallback_reason`` record the route
    taken. Existing .msh files are never remeshed on failure.

    ``mesh_size="auto"`` starts at one quarter of the longest bounding-box
    extent and halves the target until quality and sampled CAD-distance checks
    pass, up to ``max_sizing_steps``. ``geometry_tolerance`` is an absolute
    distance in model units (default: 0.001 times the longest extent in auto
    mode). This is a bounded heuristic, not a globally optimal mesh search.

    Positive Jacobians are required. The scaled-Jacobian threshold is sampled
    on a max(5, 2*p+3)^dim grid (0 disables only this distortion threshold).
    Polynomial determinants are bounded in Bernstein form with subdivision.
    This is cell validation, not a global self-intersection certificate.

    Only planar XY 2D domains and 3D solids are supported. CAD boundaries are
    approximated by polynomial patches; rational CAD reconstruction and C1
    coupling are not performed. Patch.refine/elevate preserve that polynomial.
    GmshMesh.refine/elevate/insert_knots instead refit new boundary interpolation
    points to source CAD by default, improving its approximation while checking
    Jacobians and shared interfaces. Rebuild the FunctionSpace after refitting.

    This function owns a temporary Gmsh session. An already-initialized Gmsh
    session is rejected, preserving its models and options. No GUI is opened.
    """
    if rational_geometry and geometry_order is None:
        geometry_order = 2
    if cell_type not in {"tensor", "simplex", "mixed"}:
        raise ValueError("cell_type must be 'tensor', 'simplex', or 'mixed'")
    path = Path(path).resolve()
    if path.suffix.lower() not in {".geo", ".step", ".stp", ".msh"}:
        raise GmshImportError("expected a .geo, .step, .stp or .msh file")
    if not path.is_file():
        raise FileNotFoundError(path)
    auto = isinstance(mesh_size, str) and mesh_size == "auto"
    if mesh_size is not None and not auto and (
            not isinstance(mesh_size, (float, int, np.number)) or not np.isfinite(mesh_size) or mesh_size <= 0):
        raise ValueError("mesh_size must be finite and positive")
    if geometry_order is not None and (
            not isinstance(geometry_order, (int, np.integer)) or not 1 <= geometry_order <= 8):
        raise ValueError("geometry_order must be an integer between 1 and 8")
    if cell_type == "simplex" and geometry_order is not None and geometry_order > 4:
        raise ValueError("simplex geometry_order supports degrees 1 through 4")
    if geometry_tolerance is not None and (not np.isfinite(geometry_tolerance) or geometry_tolerance <= 0):
        raise ValueError("geometry_tolerance must be finite and positive")
    if not isinstance(max_sizing_steps, (int, np.integer)) or max_sizing_steps < 1:
        raise ValueError("max_sizing_steps must be a positive integer")
    if not np.isfinite(min_scaled_jacobian) or not 0 <= min_scaled_jacobian <= 1:
        raise ValueError("min_scaled_jacobian must lie in [0, 1]")
    methods = [name for name, _, _ in _QUAD_METHODS]
    if quad_algorithm not in methods:
        raise ValueError(f"quad_algorithm must be one of {methods}")
    quad_methods = _QUAD_METHODS[methods.index(quad_algorithm):]
    if cell_type in {"simplex", "mixed"}:
        quad_methods = ((cell_type, 6, 0),)
    cad = path.suffix.lower() != ".msh"
    if not cad and rational_geometry:
        raise ValueError("rational_geometry requires CAD input (.geo or STEP), not .msh")
    if not cad and mesh_size is not None:
        raise ValueError("mesh_size applies to CAD input, not an existing .msh mesh")
    if not cad and (geometry_order is not None or geometry_tolerance is not None):
        raise ValueError("geometry_order and geometry_tolerance apply to CAD input; .msh geometry is preserved")
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise ImportError('Gmsh input requires the optional dependency: pip install "jaxiga[gmsh]"') from exc
    if gmsh.isInitialized():
        raise RuntimeError("read_gmsh requires its own Gmsh session; finalize the active session first")
    if auto:
        return _auto_mesh(gmsh, path, geometry_order, geometry_tolerance, max_sizing_steps,
                          subdivide=subdivide, min_scaled_jacobian=min_scaled_jacobian,
                          quad_algorithm=quad_algorithm, verbose=verbose, cell_type=cell_type, rational_geometry=rational_geometry)
    failures = []
    for attempt, (method, algorithm, recombination) in enumerate(quad_methods):
        # Reopen in a fresh session so a failed algorithm cannot leak modified
        # options, partial meshes or entity attributes into the next attempt.
        gmsh.initialize([], readConfigFiles=False)
        try:
            gmsh.option.setNumber("General.Terminal", int(verbose))
            gmsh.open(str(path))
            dim = gmsh.model.getDimension()
            if dim not in (2, 3):
                raise GmshImportError(f"expected a 2D domain or 3D solid, found dimension {dim}")
            try:
                if cad:
                    if cell_type in {"simplex", "mixed"}:
                        _generate(gmsh, dim, mesh_size, algorithm, recombination, cell_type=cell_type)
                    else:
                        _generate(gmsh, dim, mesh_size, algorithm, recombination)
                mesh = _finish_mesh(gmsh, dim, path, subdivide, min_scaled_jacobian, geometry_order, cell_type)
                if cad and rational_geometry:
                    from jaxiga.geometry.gmsh_rational import reconstruct
                    mesh = reconstruct(gmsh, mesh)
            except _MeshingFailure as exc:
                if cad and dim == 2 and cell_type == "tensor":
                    failures.append(f"{method}: {exc}")
                    if attempt < len(quad_methods) - 1:
                        continue
                    raise _MeshingFailure(
                        "All 2D meshing methods failed: " + "; ".join(failures)
                    ) from exc
                raise
            if cad and dim == 2:
                mesh.meshing_method = method
            elif cad:
                mesh.meshing_method = cell_type if cell_type != "tensor" else "gmsh_3d"
            mesh.fallback_reason = "; ".join(failures) or None
            if cad:
                if cell_type == "simplex":
                    from jaxiga.geometry.gmsh_simplex import cad_error
                else:
                    from jaxiga.geometry.gmsh_cad import cad_error
                mesh._source_digest = sha256(path.read_bytes()).hexdigest()
                mesh.mesh_size = mesh_size
                mesh.geometry_error = cad_error(gmsh, mesh)
                if geometry_tolerance is not None and mesh.geometry_error > geometry_tolerance:
                    raise _MeshingFailure(
                        f"sampled CAD error {mesh.geometry_error:g} exceeds geometry_tolerance "
                        f"{geometry_tolerance:g}; use a smaller mesh_size or mesh_size='auto'"
                    )
            return mesh
        finally:
            gmsh.finalize()


def _auto_mesh(gmsh, path, geometry_order, tolerance, max_steps, **kwargs):
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", int(kwargs["verbose"]))
        gmsh.open(str(path))
        bounds = np.asarray(gmsh.model.getBoundingBox(-1, -1))
        extent = float(np.max(bounds[3:] - bounds[:3]))
    finally:
        gmsh.finalize()
    if not np.isfinite(extent) or extent <= 0:
        raise GmshImportError("CAD bounding box has no finite positive extent")
    tolerance = tolerance if tolerance is not None else 0.001 * extent
    history = []
    for step in range(max_steps):
        size = extent / (4 * 2 ** step)
        try:
            mesh = read_gmsh(path, mesh_size=size, geometry_order=geometry_order, **kwargs)
        except _MeshingFailure as exc:
            history.append({"mesh_size": size, "failure": str(exc)})
            continue
        history.append({"mesh_size": size, "patches": len(mesh),
                        "geometry_error": mesh.geometry_error,
                        "min_scaled_jacobian": mesh.min_scaled_jacobian})
        if mesh.geometry_error <= tolerance:
            mesh.sizing_history = tuple(history)
            return mesh
    raise GmshImportError(
        f"automatic sizing did not meet quality/CAD tolerance {tolerance:g} "
        f"in {max_steps} attempts: {history}"
    )
