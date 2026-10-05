"""Exact MATLAB NURBS input, collapsed traces, and complex 3D examples."""

from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.io import loadmat, savemat

import jaxiga as jx
from jaxiga.examples.blade_3d import geometry as blade_geometry
from jaxiga.examples.connecting_rod_3d import geometry as rod_geometry
from jaxiga.geometry.gmsh_geometry import tensor_grid
from jaxiga.geometry.io import GeometryFormatError
from jaxiga.geometry.nurbs import evaluate_patch


DATA = Path(jx.__file__).parent / "examples/geometries"


@pytest.mark.parametrize(
    "filename", ["igapack_blade.mat", "igapack_connecting_rod.mat"]
)
def test_matlab_import_matches_independent_bspline_evaluation(filename):
    records = loadmat(DATA / filename, simplify_cells=True)
    names = [k for k in records if not k.startswith("__")]
    patches = jx.read_matlab_nurbs(DATA / filename, names)
    points = np.random.default_rng(7).uniform(0, 1, (19, 3))
    for name, p in zip(names, patches):
        d = records[name]
        B = [
            BSpline.design_matrix(
                points[:, k], d["knots"][k], int(d["order"][k]) - 1
            ).toarray()
            for k in range(3)
        ]
        H = np.einsum("qi,qj,qk,aijk->qa", *B, d["coefs"])
        np.testing.assert_allclose(
            evaluate_patch(p, points), H[:, :3] / H[:, -1:], atol=4e-12
        )
    assert len(patches) == (1 if "blade" in filename else 25)


def test_matlab_selection_and_validation(tmp_path):
    p = jx.read_matlab_nurbs(DATA / "igapack_connecting_rod.mat", ["solid4", "solid1A"])
    assert [a.degree for a in p] == [(2, 2, 1), (2, 1, 1)]
    with pytest.raises(GeometryFormatError, match="missing"):
        jx.read_matlab_nurbs(DATA / "igapack_blade.mat", "missing")
    d = loadmat(DATA / "igapack_blade.mat", simplify_cells=True)["blade"]
    d["coefs"][3, 0, 0, 0] = 0
    savemat(tmp_path / "bad.mat", {"bad": d})
    with pytest.raises(GeometryFormatError, match="weights positive"):
        jx.read_matlab_nurbs(tmp_path / "bad.mat")
    d["knots"][0] = np.array([])
    savemat(tmp_path / "bad_knots.mat", {"bad": d})
    with pytest.raises(GeometryFormatError, match="knots must"):
        jx.read_matlab_nurbs(tmp_path / "bad_knots.mat")
    savemat(tmp_path / "empty.mat", {"metadata": [1, 2]})
    with pytest.raises(GeometryFormatError, match="no NURBS"):
        jx.read_matlab_nurbs(tmp_path / "empty.mat")


def test_bezier_extraction_and_reversal_preserve_map_and_labels():
    p = jx.primitives.quarter_annulus(1, 2).elevate((2, 3)).refine(1)
    p = p.with_labels(u0="inner", u1="outer", v0="start", v1="end")
    q = np.random.default_rng(19).uniform(0, 1, (21, 2))
    for axis in (0, 1):
        mirrored = q.copy()
        mirrored[:, axis] = 1 - mirrored[:, axis]
        np.testing.assert_allclose(
            evaluate_patch(p.reverse(axis), q), evaluate_patch(p, mirrored), atol=1e-13
        )
        np.testing.assert_array_equal(
            p.reverse(axis).reverse(axis).ctrl_pts, p.ctrl_pts
        )
        assert p.reverse(axis).reverse(axis).labels == p.labels
    space = jx.FunctionSpace(p)
    cells = jx.bezier_cells(p)
    assert len(cells) == space.n_elems
    for cell, box in zip(cells, space.elem_vertex):
        np.testing.assert_allclose(
            evaluate_patch(cell, q),
            evaluate_patch(p, box[:2] + q * (box[2:] - box[:2])),
            atol=1e-13,
        )
        for side in cell.label_map:
            axis, end = "uv".index(side[0]), int(side[1])
            assert box[axis + 2 * end] == end


def wedge():
    r = tensor_grid([[0, 1]] * 3)
    xyz = np.column_stack([r[:, 0], r[:, 0] * r[:, 1], r[:, 2]])
    return jx.Patch.create([[0, 0, 1, 1]] * 3, (1,) * 3, xyz).elevate(2)


def test_collapsed_trace_has_unique_values_and_preserves_affine_fields():
    patch = wedge()
    coarse = jx.FunctionSpace(patch)
    space = jx.FunctionSpace(patch, collapse_degenerate=True)
    assert coarse.n_scalar_basis == 27
    assert space.n_scalar_basis == 21
    basis = jx.evaluate(space, jx.gauss(space))
    assert float(basis.w.sum()) == pytest.approx(0.5, abs=2e-14)
    solution = jx.Solution(space, space.cpts @ jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(
        solution.grad(basis=basis),
        np.broadcast_to([1.0, 2.0, 3.0], (1, basis.n_q, 1, 3)),
        atol=1e-13,
    )
    # Different v values name the same physical point on u=0.
    r = np.column_stack([np.zeros(9), np.linspace(0, 1, 9), np.full(9, 0.37)])
    boundary = jx.evaluate(space, jx.PointSet(np.array([0]), 2 * r - 1))
    random_field = jx.Solution(space, jnp.sin(jnp.arange(space.n_scalar_basis)))
    values = np.asarray(random_field.at(basis=boundary))
    np.testing.assert_allclose(
        values, np.full_like(values, values[0, 0, 0]), atol=1e-14
    )
    with pytest.raises(NotImplementedError, match="collapsed"):
        jx.refine_elements(space, [0])


def test_rod_has_connected_interfaces_and_exact_bore():
    patches = rod_geometry()
    space = jx.FunctionSpace(patches, vec=3)
    assert len(patches) == 25 and len(space.topology.interfaces) == 39
    connected = {0}
    for _ in patches:
        for face in space.topology.interfaces:
            if face.patch_a in connected or face.patch_b in connected:
                connected.update([face.patch_a, face.patch_b])
    assert len(connected) == 25
    b = jx.evaluate(space, jx.boundary_gauss(space, "loaded", 8))
    xy = np.asarray(b.x)[..., [0, 2]] - [146, 0]
    np.testing.assert_allclose(np.linalg.norm(xy, axis=-1), 11.5, atol=3e-12)
    assert float(b.w.sum()) == pytest.approx(2 * np.pi * 11.5 * 22, rel=1e-9)
    mixed = jx.FunctionSpace(rod_geometry(mixed=True), vec=3)
    assert mixed.n_elems == 30 and mixed.cell_type == "mixed"
    assert float(jx.integrate(lambda x: 1.0, mixed, quad=8)) == pytest.approx(
        float(jx.integrate(lambda x: 1.0, space, quad=8)), rel=2e-8
    )
    problem = jx.LinearElasticity(
        mixed,
        dirichlet=[jx.DirichletBC([0.0, 0.0, 0.0], where="fixed")],
        neumann=[
            jx.Neumann(lambda x, n, p: jnp.array([0.0, 0.0, -1.0]), where="loaded")
        ],
    )
    params = {"E": 4e5, "nu": 0.3}
    cg = jx.solve(
        problem,
        params=params,
        chunk=16,
        linear=jx.LinearOptions(method="cg", tol=1e-6, maxiter=10000),
    )
    direct = jx.solve(
        problem,
        params=params,
        chunk=16,
        linear=jx.LinearOptions(method="scipy", tol=1e-6),
    )
    assert cg.stats["converged"] and direct.stats["converged"]
    b = jx.evaluate(mixed, jx.gauss(mixed))
    u, ref = np.asarray(cg.at(basis=b)), np.asarray(direct.at(basis=b))
    assert np.linalg.norm(u - ref) / np.linalg.norm(ref) < 1e-4


def test_blade_cg_matches_direct_and_exports_finite_cell_stress(tmp_path):
    p = blade_geometry()
    space = jx.FunctionSpace(p, vec=3, collapse_degenerate=True)
    assert space.n_elems == 200 and space.n_dofs == 2112
    problem = jx.LinearElasticity(
        space,
        dirichlet=[jx.DirichletBC([0.0, 0.0, 0.0], where="fixed")],
        neumann=[
            jx.Neumann(lambda x, n, p: jnp.array([0.0, -0.001, 0.0]), where="loaded")
        ],
    )
    params = {"E": 1e5, "nu": 0.3}
    iterative = jx.solve(
        problem,
        params=params,
        chunk=32,
        linear=jx.LinearOptions(method="cg", tol=1e-6, maxiter=10000),
    )
    direct = jx.solve(
        problem,
        params=params,
        chunk=32,
        linear=jx.LinearOptions(method="scipy", tol=1e-6),
    )
    assert iterative.stats["converged"] and direct.stats["converged"]
    b = jx.evaluate(space, jx.gauss(space))
    u, ref = np.asarray(iterative.at(basis=b)), np.asarray(direct.at(basis=b))
    assert np.linalg.norm(u - ref) / np.linalg.norm(ref) < 1e-4
    assert np.isfinite(iterative.cell_average("von_mises")).all()
    with pytest.raises(ValueError, match="cell_fields"):
        iterative.to_vtk(tmp_path / "invalid", fields=("von_mises",))
    pytest.importorskip("pyevtk")
    path = iterative.to_vtk(tmp_path / "blade", n=1, cell_fields=("von_mises",))
    data = Path(path).read_bytes()
    split = data.index(b"<AppendedData")
    root = ET.fromstring(data[:split] + b"</VTKFile>")
    start = data.index(b"_", split) + 1
    assert root.find(".//CellData/DataArray[@Name='von_mises']") is not None
    assert root.find(".//PointData/DataArray[@Name='displacement']") is not None
    for array in root.iter("DataArray"):
        if array.attrib["type"] not in {"Float32", "Float64"}:
            continue
        pos = start + int(array.attrib["offset"])
        size = struct.unpack_from("<Q", data, pos)[0]
        dtype = "<f4" if array.attrib["type"] == "Float32" else "<f8"
        assert np.isfinite(
            np.frombuffer(data[pos + 8 : pos + 8 + size], dtype=dtype)
        ).all()
