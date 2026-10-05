"""Retain Gmsh quad/triangle or hex/tetra blocks without tensor subdivision."""

from dataclasses import replace
from types import SimpleNamespace
from itertools import combinations

import numpy as np

from jaxiga.geometry.mixed import conform_mixed


class _MeshView:
    def __init__(self, gmsh, dim, simplex):
        self.mesh = gmsh.model.mesh
        self.dim = dim
        self.simplex = simplex
        self.supports = set()
        kinds, tags, connectivity = self.mesh.getElements(dim)
        for kind, ids, conn in zip(kinds, tags, connectivity):
            name, _, _, count, _, primary = self.mesh.getElementProperties(kind)
            selected = name.startswith("Triangle" if dim == 2 else "Tetrahedron")
            if selected != simplex:
                continue
            for cell in np.asarray(conn).reshape(len(ids), count)[:, :primary]:
                for n in range(1, primary + 1):
                    self.supports.update(
                        tuple(sorted(c)) for c in combinations(cell, n)
                    )

    def __getattr__(self, name):
        return getattr(self.mesh, name)

    def getElements(self, dim=-1, tag=-1):
        kinds, tags, connectivity = self.mesh.getElements(dim, tag)
        result = [[], [], []]
        for kind, ids, conn in zip(kinds, tags, connectivity):
            name, edim, _, count, _, primary = self.mesh.getElementProperties(kind)
            cells = np.asarray(conn).reshape(len(ids), count)
            if edim == self.dim:
                selected = name.startswith(
                    "Triangle" if self.dim == 2 else "Tetrahedron"
                )
                keep = np.full(len(ids), selected == self.simplex)
            else:
                keep = np.array(
                    [tuple(sorted(c[:primary])) in self.supports for c in cells]
                )
            if keep.any():
                result[0].append(kind)
                result[1].append(np.asarray(ids)[keep])
                result[2].append(cells[keep].ravel())
        return tuple(result)


class _ModelView:
    def __init__(self, gmsh, dim, simplex):
        self.model = gmsh.model
        self.mesh = _MeshView(gmsh, dim, simplex)

    def __getattr__(self, name):
        return getattr(self.model, name)


def convert_mixed(gmsh, dim, path, min_quality):
    from jaxiga.geometry.gmsh import (
        _convert,
        _elements,
        GmshImportError,
        _MeshingFailure,
    )
    from jaxiga.geometry.gmsh_simplex import convert_simplex, simplex_quality
    from jaxiga.geometry.gmsh_geometry import spline_quality
    from jaxiga.geometry.simplex import SimplexPatch

    families = [
        gmsh.model.mesh.getElementProperties(k)[0] for k, _, _ in _elements(gmsh, dim)
    ]
    allowed = (
        ("Triangle", "Quadrilateral") if dim == 2 else ("Tetrahedron", "Hexahedron")
    )
    if any(not name.startswith(allowed) for name in families):
        raise GmshImportError(
            "mixed mode supports triangles/quads or tetrahedra/hexes; pyramids and prisms are not yet supported"
        )
    pieces = []
    for simplex, convert in [(False, _convert), (True, convert_simplex)]:
        family = allowed[0] if simplex else allowed[1]
        if not any(name.startswith(family) for name in families):
            continue
        view = SimpleNamespace(
            model=_ModelView(gmsh, dim, simplex), __version__=gmsh.__version__
        )
        pieces.append(convert(view, dim, path, min_quality))
    if len(pieces) == 1:
        return pieces[0]
    patches = []
    tags = []
    corners = []
    constraints = []
    cell_groups = {}
    boundary_groups = {}
    for mesh in pieces:
        offset = len(patches)
        patches.extend(mesh.patches)
        tags.extend(mesh.element_tags)
        corners.extend(mesh._corner_tags)
        constraints.extend(mesh._cad_constraints)
        for name, ids in mesh.cell_groups.items():
            cell_groups.setdefault(name, []).extend(offset + i for i in ids)
        for name, ids in mesh.boundary_groups.items():
            boundary_groups.setdefault(name, []).extend(
                (offset + i, side) for i, side in ids
            )
    try:
        patches = conform_mixed(patches)
        quality = [
            simplex_quality(p) if isinstance(p, SimplexPatch) else spline_quality(p)
            for p in patches
        ]
    except ValueError as exc:
        raise _MeshingFailure(str(exc)) from exc
    if min(q for q, _ in quality) < min_quality:
        raise _MeshingFailure(
            "mixed trace reconstruction fails the scaled-Jacobian threshold"
        )
    return replace(
        pieces[0],
        patches=patches,
        element_tags=np.asarray(tags),
        cell_groups={k: tuple(v) for k, v in cell_groups.items()},
        boundary_groups={k: tuple(v) for k, v in boundary_groups.items()},
        scaled_jacobians=np.array([q for q, _ in quality]),
        jacobian_lower_bounds=np.array([b for _, b in quality]),
        _corner_tags=tuple(corners),
        _cad_constraints=tuple(constraints),
    )
