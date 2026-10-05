"""Geometry import and export: geomdl JSON and G+Smo XML.

Patches are compared through their *geometry maps* rather than their
coefficients. Two files can describe the same surface with different control
data -- and the two formats disagree about control-point ordering, which is
exactly the mistake worth catching -- so agreement of the maps is the property
that matters.
"""

import numpy as np
import pytest

import jaxiga as jx
from jaxiga.geometry import io
from jaxiga.geometry.nurbs import evaluate_patch

QUARTER_ANNULUS_XML = """<?xml version="1.0"?>
<xml>
 <Geometry type="TensorNurbs2" id="0">
  <Basis type="TensorNurbsBasis2">
   <Basis type="TensorBSplineBasis2">
    <Basis type="BSplineBasis" index="0">
     <KnotVector degree="1">0 0 1 1</KnotVector>
    </Basis>
    <Basis type="BSplineBasis" index="1">
     <KnotVector degree="2">0 0 0 1 1 1</KnotVector>
    </Basis>
   </Basis>
   <weights>1 1 0.70710678118654757 0.70710678118654757 1 1</weights>
  </Basis>
  <coefs geoDim="2">1 0
2 0
1 1
2 2
0 1
0 2</coefs>
 </Geometry>
</xml>
"""


def map_difference(a, b, n=7):
    """Largest gap between two geometry maps over a parametric grid."""
    assert a.dim == b.dim
    grids = np.meshgrid(*[np.linspace(0.0, 1.0, n)] * a.dim, indexing="ij")
    params = np.stack([g.ravel() for g in grids], axis=1)
    return float(np.abs(evaluate_patch(a, params) - evaluate_patch(b, params)).max())


PATCHES = {
    "rectangle_nonsquare_net": jx.primitives.rectangle(0, 0, 3, 1)
    .elevate((2, 3))
    .insert_knots([0.3, 0.6], [0.5]),
    "quarter_annulus": jx.primitives.quarter_annulus(1.0, 2.0).elevate(2).refine(1),
    "plate_quadrant_C0": jx.primitives.plate_with_hole_quadrant(1.0, 4.0, 2)
    .elevate(3)
    .refine(1),
    "cuboid": jx.primitives.cuboid([0, 0, 0], [2, 1, 3]).elevate(2).refine(1),
    "interval": jx.primitives.interval(0.0, 2.0).elevate(3).refine(2),
}


@pytest.mark.parametrize("suffix", [".json", ".xml"])
@pytest.mark.parametrize("name", sorted(PATCHES))
def test_round_trip_preserves_the_geometry(tmp_path, name, suffix):
    patch = PATCHES[name]
    path = tmp_path / f"patch{suffix}"
    io.write(path, patch)
    back = io.read(path)[0]

    assert back.degree == patch.degree
    assert back.n_cp == patch.n_cp
    assert back.dim_phys == patch.dim_phys
    assert map_difference(patch, back) < 1e-13


@pytest.mark.parametrize("suffix", [".json", ".xml"])
def test_round_trip_of_several_patches(tmp_path, suffix):
    patches = jx.primitives.plate_with_hole(1.0, 4.0)
    path = tmp_path / f"multi{suffix}"
    io.write(path, patches)
    back = io.read(path)
    assert len(back) == len(patches)
    for a, b in zip(patches, back):
        assert map_difference(a, b) < 1e-13
    # and the result is still a usable space
    V = jx.FunctionSpace([p.elevate(2).refine(1) for p in back])
    assert V.n_elems > 0


def test_reading_a_geomdl_file_written_by_geomdl():
    """Cross-check against geomdl's own evaluator, not just our own writer."""
    geomdl = pytest.importorskip("geomdl")
    from geomdl import BSpline, exchange
    import tempfile, os

    surf = BSpline.Surface()
    surf.degree_u, surf.degree_v = 2, 1
    # deliberately a 3x2 net: a transposed reader would still pass on a square one
    surf.set_ctrlpts(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1.2, 1, 0], [0, 2, 0], [2, 2, 0]], 3, 2
    )
    surf.knotvector_u = [0, 0, 0, 1, 1, 1]
    surf.knotvector_v = [0, 0, 1, 1]

    path = os.path.join(tempfile.mkdtemp(), "surface.json")
    exchange.export_json(surf, path)
    patch = io.read(path)[0]

    assert patch.degree == (2, 1)
    assert patch.n_cp_per_dir == (3, 2)
    params = np.array([[u, v] for u in np.linspace(0, 1, 5) for v in np.linspace(0, 1, 5)])
    theirs = np.array(surf.evaluate_list([tuple(p) for p in params]))[:, :2]
    np.testing.assert_allclose(evaluate_patch(patch, params), theirs, atol=1e-13)


def test_reading_a_hand_written_gismo_file(tmp_path):
    """Parse XML this package did not write, with the documented ordering."""
    path = tmp_path / "annulus.xml"
    path.write_text(QUARTER_ANNULUS_XML)
    patch = io.read(path)[0]

    assert patch.degree == (1, 2)
    assert patch.n_cp_per_dir == (2, 3)
    assert map_difference(patch, jx.primitives.quarter_annulus(1.0, 2.0)) < 1e-12


def test_planar_surface_written_in_3d_comes_back_two_dimensional(tmp_path):
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(2)
    path = tmp_path / "flat.json"
    io.write(path, patch)  # padded to 3 components on the way out
    assert len(io.read(path)[0].ctrl_pts[0]) == 2


def test_unit_weights_are_written_as_a_bspline(tmp_path):
    path = tmp_path / "bspline.xml"
    io.write(path, jx.primitives.rectangle(0, 0, 1, 1).elevate(2))
    text = path.read_text()
    assert "TensorBSpline2" in text and "weights" not in text

    path = tmp_path / "nurbs.xml"
    io.write(path, jx.primitives.quarter_annulus(1.0, 2.0))
    text = path.read_text()
    assert "TensorNurbs2" in text and "<weights>" in text


def test_unknown_extension_raises(tmp_path):
    with pytest.raises(io.GeometryFormatError, match="unknown geometry format"):
        io.read(tmp_path / "geometry.stl")


def test_inconsistent_knot_vector_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"degree_u": 2, "knotvector_u": [0,0,0,1,1,1], "size_u": 3,'
        ' "degree_v": 1, "knotvector_v": [0,0,1,1,1], "size_v": 2,'
        ' "control_points": {"points": [[0,0],[1,0],[0,1],[1,1],[0,2],[1,2]]}}'
    )
    with pytest.raises(io.GeometryFormatError, match="inconsistent"):
        io.read(path)


def test_missing_geometry_element_raises(tmp_path):
    path = tmp_path / "empty.xml"
    path.write_text("<?xml version='1.0'?><xml></xml>")
    with pytest.raises(io.GeometryFormatError, match="no <Geometry>"):
        io.read(path)
