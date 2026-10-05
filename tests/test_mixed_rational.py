"""Exact conic reconstruction, mixed trace coupling, and ParaView export."""

from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxiga as jx
from jaxiga.geometry.mixed import ordered_interfaces
from jaxiga.geometry.nurbs import evaluate_patch
from jaxiga.space.pointset import PointSet


GEOMETRIES = Path(jx.__file__).parent / "examples/geometries"


def cube(x0=0.0, x1=1.0):
    from jaxiga.geometry.gmsh_geometry import tensor_grid

    c = tensor_grid([[x0, x1], [0, 1], [0, 1]])
    return jx.Patch.create([[0, 0, 1, 1]] * 3, (1,) * 3, c)


def mixed_cells(dim, p=1):
    if dim == 2:
        a = jx.primitives.rectangle(0, 0, 1, 1).elevate(p)
        b = jx.primitives.rectangle(1, 0, 2, 1).elevate(p)
    else:
        a = cube().elevate(p)
        b = cube(1, 2).elevate(p)
    return [a, *jx.split_to_simplices(b)]


@pytest.mark.parametrize("dim", [2, 3])
def test_split_preserves_rational_map(dim):
    p = jx.primitives.quarter_annulus(1, 2) if dim == 2 else cube().elevate(2)
    if dim == 3:
        import dataclasses

        p = dataclasses.replace(p, weights=jnp.linspace(0.8, 1.2, p.n_cp))
    children = jx.split_to_simplices(p)
    assert len(children) == (2 if dim == 2 else 6)
    for child in children:
        bary = np.random.default_rng(12).dirichlet(np.ones(dim + 1), 11)
        vertices = np.asarray(child.ctrl_pts)[child.vertex_indices]
        if dim == 2:
            reference_vertices = np.column_stack(
                [
                    np.linalg.norm(vertices, axis=1) - 1,
                    np.arctan2(vertices[:, 1], vertices[:, 0]) / (np.pi / 2),
                ]
            )
        else:
            reference_vertices = vertices
        expected = evaluate_patch(p, bary @ reference_vertices)
        np.testing.assert_allclose(
            evaluate_patch(child, bary[:, 1:]), expected, atol=2e-12
        )
        assert np.all(np.asarray(child.weights) > 0)
    if dim == 2:
        a = jx.FunctionSpace(p)
        b = jx.FunctionSpace(children)
        assert float(jx.integrate(lambda x: 1.0, a, quad=10)) == pytest.approx(
            float(jx.integrate(lambda x: 1.0, b, quad=10)), rel=2e-8
        )


@pytest.mark.parametrize("dim", [2, 3])
def test_mixed_linear_patch_neumann_normals_and_geometry(dim):
    cells = mixed_cells(dim)
    space = jx.FunctionSpace(cells)
    assert space.cell_type == "mixed"
    assert float(jx.integrate(lambda x: 1.0, space)) == pytest.approx(2.0, abs=3e-12)
    exact = lambda x: 1 + 2 * x[0] - x[1]
    problem = jx.Poisson(
        space,
        dirichlet=[
            jx.DirichletBC(lambda x, p: exact(x), where=lambda x: np.ones(len(x), bool))
        ],
    )
    sol = jx.solve(problem, params={"a0": 1.0})
    assert jx.errornorm(sol, exact, "L2") < 2e-12
    normal_sum = np.zeros(dim)
    flux = 0.0
    for label in space.boundaries:
        b = jx.evaluate(space, jx.boundary_gauss(space, label))
        normal_sum += np.sum(b.w[..., None] * b.normal, axis=(0, 1))
        flux += float(jnp.sum(b.w * jnp.sum(b.x * b.normal, axis=-1)))
    np.testing.assert_allclose(normal_sum, 0, atol=2e-12)
    assert flux == pytest.approx(2 * dim, abs=2e-12)
    # General coefficient vectors must have equal values on the interface,
    # not just the affine manufactured solution.
    pairs, _ = ordered_interfaces(cells)
    u = jnp.sin(jnp.arange(space.n_scalar_basis, dtype=float))
    sol = jx.Solution(space, u)
    for a, b in pairs:
        bary = np.random.default_rng(11).dirichlet(np.ones(len(b.vertices)), 7)
        rb = bary @ b.reference
        ra = bary @ np.array([a.reference[a.vertices.index(v)] for v in b.vertices])
        ba = jx.evaluate(space, PointSet(np.array([a.cell]), 2 * ra - 1))
        bb = jx.evaluate(space, PointSet(np.array([b.cell]), 2 * rb - 1))
        np.testing.assert_allclose(ba.x, bb.x, atol=2e-12)
        np.testing.assert_allclose(sol.at(basis=ba), sol.at(basis=bb), atol=5e-12)


def test_mixed_quadratic_solve_and_shape_gradient():
    space = jx.FunctionSpace(mixed_cells(2, 2))
    exact = lambda x: x[0] * (2 - x[0]) * x[1] * (1 - x[1])
    # Zero boundary data; compare a mixed solution to the exact polynomial.
    problem = jx.Poisson(
        space,
        source=lambda x, p: 2 * (x[1] * (1 - x[1]) + x[0] * (2 - x[0])),
        dirichlet=[jx.DirichletBC(0.0, where=lambda x: np.ones(len(x), bool))],
    )
    solution = jx.solve(problem, params={"a0": 1.0})
    assert jx.errornorm(solution, exact, "L2") < 2e-12
    import dataclasses

    rule = jx.gauss(space)
    volume = lambda s: jx.evaluate(
        dataclasses.replace(space, cpts=space.cpts * s), rule
    ).w.sum()
    assert jax.grad(volume)(1.0) == pytest.approx(4.0, abs=2e-11)


def test_gmsh_mixed_import(tmp_path):
    pytest.importorskip("gmsh")
    path = tmp_path / "mixed.geo"
    path.write_text("""SetFactory("OpenCASCADE");
Rectangle(1)={0,0,0,1,1}; Rectangle(2)={1,0,0,1,1};
BooleanFragments{Surface{1};Delete;}{Surface{2};Delete;}
Transfinite Curve{:}=3; Transfinite Surface{1}; Recombine Surface{1};
Physical Surface("quad_region")={1}; Physical Surface("tri_region")={2};
""")
    mesh = jx.read_gmsh(path, cell_type="mixed", geometry_order=2)
    assert mesh.cell_type == "mixed" and not mesh.subdivided
    assert set(mesh.cell_groups) == {"quad_region", "tri_region"}
    assert float(jx.integrate(lambda x: 1.0, jx.FunctionSpace(mesh))) == pytest.approx(
        2
    )


@pytest.mark.parametrize("kind", ["tensor", "simplex"])
def test_exact_ellipse_and_shape_preserving_elevation(tmp_path, kind):
    pytest.importorskip("gmsh")
    path = tmp_path / "ellipse.geo"
    path.write_text(
        'SetFactory("OpenCASCADE"); Disk(1)={0,0,0,2,1}; Physical Curve("wall")={1}; Physical Surface("disk")={1};\n'
    )
    mesh = jx.read_gmsh(
        path, cell_type=kind, mesh_size=0.6, geometry_order=2, rational_geometry=True
    )
    assert mesh.geometry_representation == "rational_cad"
    fine = mesh.elevate(3)
    space = jx.FunctionSpace(fine)
    b = jx.evaluate(space, jx.boundary_gauss(space, "wall", n=11))
    xy = np.asarray(b.x)
    np.testing.assert_allclose(xy[..., 0] ** 2 / 4 + xy[..., 1] ** 2, 1, atol=2e-12)
    assert any(np.any(np.abs(np.asarray(p.weights) - 1) > 0.001) for p in fine)


def test_exact_cylinder_mixed_and_vtk(tmp_path):
    pytest.importorskip("gmsh")
    mesh = jx.read_gmsh(
        GEOMETRIES / "quarter_cylinder.geo", geometry_order=2, rational_geometry=True
    )
    assert len(mesh) == 2
    patches = [mesh[0], *jx.split_to_simplices(mesh[1])]
    space = jx.FunctionSpace(patches, vec=3)
    assert len(space.topology.interfaces) == 8
    assert jx.integrate(lambda x: 1.0, space, quad=12) == pytest.approx(
        1.5 * np.pi, abs=2e-9
    )
    for label, radius in [("inner", 1.0), ("outer", 2.0)]:
        b = jx.evaluate(space, jx.boundary_gauss(space, label, n=9))
        np.testing.assert_allclose(
            np.sum(np.asarray(b.x)[..., :2] ** 2, axis=-1), radius**2, atol=5e-12
        )
    solution = jx.Solution(space, space.cpts.ravel())
    output = solution.to_vtk(tmp_path / "cylinder.vtu", n=2)
    header = Path(output).read_bytes().split(b"<AppendedData")[0].decode()
    assert (
        'Name="displacement" NumberOfComponents="3"' in header
        or 'Name="displacement"' in header
    )
    assert 'Name="element_id"' in header
    from jaxiga.post.vtk import _grid, MixedCells

    _, coords, cells, _ = _grid(space, 2)
    assert isinstance(cells, MixedCells)
    assert set(cells.types) == {10, 12}
    assert cells.offsets[-1] == len(cells.connectivity)
    assert cells.connectivity.max() < len(coords[0])
    assert np.all(np.diff(np.r_[0, cells.offsets]) == np.where(cells.types == 10, 4, 8))
    assert re.search('NumberOfCells="' + str(len(cells.types)) + '"', header)


def test_unsupported_rational_surface_is_explicit(tmp_path):
    pytest.importorskip("gmsh")
    path = tmp_path / "sphere.geo"
    path.write_text('SetFactory("OpenCASCADE"); Sphere(1)={0,0,0,1};\n')
    with pytest.raises(ValueError, match="does not support Sphere"):
        jx.read_gmsh(path, cell_type="simplex", mesh_size=0.9, rational_geometry=True)


def test_mixed_probe_does_not_extrapolate_outside_triangle():
    quad = jx.primitives.rectangle(0, 0, 1, 1)
    tri = jx.primitives.triangle([[1, 0], [2, 0], [1, 1]])
    space = jx.FunctionSpace([quad, tri])
    solution = jx.Solution(space, space.cpts[:, 0])
    np.testing.assert_allclose(
        solution.probe([[0.8, 0.7], [1.1, 0.5]])[:, 0], [0.8, 1.1], atol=1e-12
    )
    assert np.isnan(solution.probe([[1.8, 0.8]])).all()


def test_vtu_binary_connectivity_vectors_and_owners(tmp_path):
    import struct
    import xml.etree.ElementTree as ET

    space = jx.FunctionSpace(mixed_cells(3), vec=3)
    path = jx.Solution(space, space.cpts.ravel()).to_vtk(tmp_path / "mixed", n=2)
    data = Path(path).read_bytes()
    split = data.index(b"<AppendedData")
    root = ET.fromstring(data[:split] + b"</VTKFile>")
    start = data.index(b"_", split) + 1
    arrays = {}
    types = {"Float32": "<f4", "Float64": "<f8", "Int64": "<i8", "UInt8": "u1"}
    for item in root.iter("DataArray"):
        position = start + int(item.attrib["offset"])
        count = struct.unpack_from("<Q", data, position)[0]
        arrays[item.attrib["Name"]] = np.frombuffer(
            data[position + 8 : position + 8 + count], dtype=types[item.attrib["type"]]
        )
    points = arrays["points"].reshape(-1, 3)
    np.testing.assert_allclose(arrays["displacement"].reshape(-1, 3), points, atol=1e-6)
    assert set(arrays["types"]) == {10, 12}
    assert arrays["offsets"][-1] == len(arrays["connectivity"])
    assert arrays["connectivity"].max() < len(points)
    assert len(arrays["element_id"]) == len(arrays["types"])
    assert set(arrays["element_id"]) == set(range(space.n_elems))


def test_cylinder_pressure_matches_lame_solution():
    pytest.importorskip("gmsh")
    mesh = jx.read_gmsh(
        GEOMETRIES / "quarter_cylinder.geo", geometry_order=2, rational_geometry=True
    )
    space = jx.FunctionSpace([mesh[0], *jx.split_to_simplices(mesh[1])], vec=3)
    problem = jx.LinearElasticity(
        space,
        dirichlet=[
            jx.DirichletBC(0.0, where="sym_x", component=0),
            jx.DirichletBC(0.0, where="sym_y", component=1),
            jx.DirichletBC(0.0, where="ends", component=2),
        ],
        neumann=[jx.Neumann(lambda x, n, p: -10 * n, where="inner")],
    )
    solution = jx.solve(problem, params={"E": 1e5, "nu": 0.3})
    iterative = jx.solve(
        problem, params={"E": 1e5, "nu": 0.3}, chunk=16,
        linear=jx.LinearOptions(method="cg", tol=1e-10, maxiter=5000),
    )
    assert iterative.stats["converged"]
    np.testing.assert_allclose(iterative.u, solution.u, rtol=1e-5, atol=1e-9)
    basis = jx.evaluate(space, jx.gauss(space))
    xy = np.asarray(basis.x)[..., :2]
    r = np.linalg.norm(xy, axis=-1)
    ur = 1.3 / 1e5 * ((1 - 0.6) * 10 / 3 * r + (40 / 3) / r)
    expected = ur[..., None] * xy / r[..., None]
    displacement = np.asarray(solution.at(basis=basis))
    relative = np.sqrt(
        np.sum(basis.w[..., None] * (displacement[..., :2] - expected) ** 2)
        / np.sum(basis.w[..., None] * expected**2)
    )
    assert relative < 0.04
    assert np.all(np.sum(displacement[..., :2] * xy, axis=-1) > 0)
    np.testing.assert_allclose(displacement[..., 2], 0, atol=5e-6)
    assert np.isfinite(solution.field("von_mises", basis=basis)).all()


def test_existing_mixed_hex_tet_mesh_is_elevated_and_coupled(tmp_path):
    pytest.importorskip("gmsh")
    from itertools import permutations
    from jaxiga.geometry.gmsh_geometry import tensor_grid

    vertices = tensor_grid([[0, 1, 2], [0, 1], [0, 1]])

    def ids(points):
        return [
            int(np.flatnonzero(np.all(vertices == p, axis=1))[0]) + 1 for p in points
        ]

    hex_corners = tensor_grid([[0, 1], [0, 1], [0, 1]])
    elements = [(5, ids(hex_corners[[0, 1, 3, 2, 4, 5, 7, 6]]))]
    for permutation in permutations(range(3)):
        ref = [np.array([1, 0, 0])]
        for axis in permutation:
            p = ref[-1].copy()
            p[axis] += 1
            ref.append(p)
        ref = np.asarray(ref)
        if np.linalg.det(ref[1:] - ref[0]) < 0:
            ref[[0, 1]] = ref[[1, 0]]
        elements.append((4, ids(ref)))
    text = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$Nodes", str(len(vertices))]
    text += [f"{i + 1} " + " ".join(map(str, p)) for i, p in enumerate(vertices)]
    text += ["$EndNodes", "$Elements", str(len(elements))]
    text += [
        f"{i + 1} {kind} 0 " + " ".join(map(str, conn))
        for i, (kind, conn) in enumerate(elements)
    ]
    text += ["$EndElements"]
    path = tmp_path / "mixed.msh"
    path.write_text("\n".join(text) + "\n")
    mesh = jx.read_gmsh(path, cell_type="mixed")
    assert len(mesh) == 7 and mesh.cell_type == "mixed" and not mesh.subdivided
    assert any(isinstance(p, jx.SimplexPatch) and p.degree[0] == 2 for p in mesh)
    space = jx.FunctionSpace(mesh)
    assert float(jx.integrate(lambda x: 1.0, space)) == pytest.approx(2.0, abs=1e-12)
    exact = lambda x: x.sum()
    problem = jx.Poisson(
        space,
        dirichlet=[
            jx.DirichletBC(lambda x, p: exact(x), where=lambda x: np.ones(len(x), bool))
        ],
    )
    assert jx.errornorm(jx.solve(problem, params={"a0": 1.0}), exact, "L2") < 1e-12


def test_incomplete_hex_tet_face_coverage_is_rejected():
    vertices = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [2, 0, 0]], dtype=float)
    if np.linalg.det(vertices[1:] - vertices[0]) < 0:
        vertices[[0, 1]] = vertices[[1, 0]]
    tet = jx.SimplexPatch.from_vertices(vertices).elevate(2)
    with pytest.raises(ValueError, match="exactly two triangles"):
        jx.FunctionSpace([cube(), tet])
