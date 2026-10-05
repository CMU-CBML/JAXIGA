"""CAD-to-patch integration, mesh validity and boundary-name preservation."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jaxiga as jx
from jaxiga.geometry.gmsh import GmshImportError
from jaxiga.geometry.nurbs import evaluate_patch


gmsh = pytest.importorskip("gmsh")


def write_mesh(path, points, elements, groups=()):
    """Write MSH2 with sparse node/element tags, independent of the importer."""
    node_tags = [10 + 7 * i for i in range(len(points))]
    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", str(len(groups))]
    lines += [f"{dim} {tag} {json.dumps(name)}" for dim, tag, name in groups]
    lines += ["$EndPhysicalNames", "$Nodes", str(len(points))]
    for tag, xyz in zip(node_tags, points):
        lines.append(f"{tag} " + " ".join(map(str, xyz)))
    lines += ["$EndNodes", "$Elements", str(len(elements))]
    for i, (kind, physical, entity, indices) in enumerate(elements):
        lines.append(f"{100 + 3*i} {kind} 2 {physical} {entity} "
                     + " ".join(str(node_tags[j]) for j in indices))
    lines += ["$EndElements"]
    path.write_text("\n".join(lines) + "\n")
    return path


def square(tmp_path, reverse=False):
    return write_mesh(
        tmp_path / "square.msh", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        [(1, 2, 1, [0, 3]), (1, 3, 2, [1, 2]),
         (3, 4, 1, [0, 3, 2, 1] if reverse else [0, 1, 2, 3])],
        [(1, 2, "fixed"), (1, 3, "loaded"), (2, 4, "solid")],
    )


def test_quad_geometry_labels_and_sparse_tags(tmp_path):
    mesh = jx.io.read(square(tmp_path))
    assert len(mesh) == 1
    assert mesh.element_tags.tolist() == [106]
    assert mesh.cell_groups == {"solid": (0,)}
    assert mesh.physical_names[1, 2] == "fixed"
    assert mesh.boundary_groups == {"fixed": ((0, "u0"),), "loaded": ((0, "u1"),)}
    np.testing.assert_allclose(evaluate_patch(mesh[0], [[0.2, 0.7]]), [[0.2, 0.7]])
    assert mesh.min_scaled_jacobian == pytest.approx(1)
    assert mesh.jacobian_lower_bounds[0] > 0
    assert mesh.meshing_method == "existing_mesh"
    assert not mesh.subdivided and mesh.fallback_reason is None
    V = jx.FunctionSpace([p.elevate(3).refine(1) for p in mesh])
    assert len(V.boundary("fixed").dofs) == 5
    assert not gmsh.isInitialized()


def test_reversed_quad_is_reoriented_with_labels(tmp_path):
    mesh = jx.read_gmsh(square(tmp_path, reverse=True))
    assert mesh.min_scaled_jacobian > 0
    V = jx.FunctionSpace(mesh)
    for name, coordinate in [("fixed", 0), ("loaded", 1)]:
        np.testing.assert_allclose(np.asarray(V.cpts)[V.boundary(name).dofs, 0], coordinate)


def test_triangle_subdivision_preserves_boundary_and_domain_groups(tmp_path):
    path = write_mesh(tmp_path / "tri.msh", [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                      [(1, 2, 1, [0, 1]), (2, 3, 1, [0, 1, 2])],
                      [(1, 2, "bottom"), (2, 3, "domain")])
    with pytest.raises(GmshImportError, match="quadrilateral"):
        jx.read_gmsh(path, subdivide=False)
    mesh = jx.read_gmsh(path)
    assert len(mesh) == 3
    assert len(mesh.boundary_groups["bottom"]) == 2
    assert len(mesh.cell_groups["domain"]) == 3
    V = jx.FunctionSpace([p.elevate(2) for p in mesh])
    assert len(V.topology.interfaces) == 3
    assert jx.integrate(lambda x: 1.0, V) == pytest.approx(0.5)


def test_tet_subdivision_and_affine_patch_test(tmp_path):
    path = write_mesh(tmp_path / "tet.msh",
                      [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                      [(2, 2, 1, [0, 2, 1]), (2, 2, 2, [0, 1, 3]),
                       (2, 2, 3, [1, 2, 3]), (2, 2, 4, [2, 0, 3]),
                       (4, 3, 1, [0, 1, 2, 3])],
                      [(2, 2, "outside"), (3, 3, "solid")])
    mesh = jx.read_gmsh(path)
    assert len(mesh) == 4
    assert len(mesh.boundary_groups["outside"]) == 12
    assert len(mesh.cell_groups["solid"]) == 4
    V = jx.FunctionSpace([p.elevate(2) for p in mesh])
    assert len(V.topology.interfaces) == 6
    problem = jx.Poisson(V, dirichlet=[
        jx.DirichletBC(lambda x, p: jnp.sum(x), where="outside")])
    solution = jx.solve(problem, params={"a0": 1.0})
    assert jx.errornorm(solution, lambda x: jnp.sum(x), "L2") < 1e-11
    assert jx.integrate(lambda x: 1.0, V) == pytest.approx(1 / 6)


def test_geo_meshing_preserves_named_boundaries(tmp_path):
    path = tmp_path / "plate.geo"
    path.write_text('''SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 2, 1};
Physical Curve("fixed") = {4};
Physical Surface("plate") = {1};
''')
    mesh = jx.read_gmsh(path, mesh_size=0.5)
    assert mesh.meshing_method == "quasi_structured"
    assert mesh.fallback_reason is None
    assert len(mesh) > 1
    assert len(mesh.cell_groups["plate"]) == len(mesh)
    V = jx.FunctionSpace(mesh)
    np.testing.assert_allclose(np.asarray(V.cpts)[V.boundary("fixed").dofs, 0], 0, atol=1e-12)


def test_step_solid_to_hexes(tmp_path):
    path = tmp_path / "solid.step"
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(0, 0, 0, 1, 0.7, 0.5)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    mesh = jx.read_gmsh(path, mesh_size=0.8)
    assert mesh.meshing_method == "gmsh_3d"
    assert mesh.subdivided and mesh.fallback_reason is None
    assert len(mesh) > 1
    assert all(p.dim == 3 and p.n_cp == 8 for p in mesh)
    assert np.all(mesh.jacobian_lower_bounds > 0)
    volume = sum(jx.integrate(lambda x: 1.0, jx.FunctionSpace(p)) for p in mesh)
    assert volume == pytest.approx(0.35)


@pytest.mark.parametrize("indices", [[0, 1, 3, 2], [0, 1, 1, 3]])
def test_folded_or_repeated_vertex_quad_rejected(tmp_path, indices):
    path = write_mesh(tmp_path / "bad.msh",
                      [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                      [(3, 0, 1, indices)])
    with pytest.raises(GmshImportError, match="inverted|repeated"):
        jx.read_gmsh(path)
    assert not gmsh.isInitialized()


def test_distortion_threshold(tmp_path):
    path = write_mesh(tmp_path / "skew.msh",
                      [(0, 0, 0), (1, 0, 0), (2, 0.01, 0), (1, 0.01, 0)],
                      [(3, 0, 1, [0, 1, 2, 3])])
    with pytest.raises(GmshImportError, match="distorted"):
        jx.read_gmsh(path)
    mesh = jx.read_gmsh(path, min_scaled_jacobian=0)
    assert 0 < mesh.min_scaled_jacobian < 0.02


def test_high_order_mesh_is_not_silently_linearized(tmp_path):
    path = write_mesh(tmp_path / "quadratic.msh",
                      [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                       (0.5, 0, 0), (1, 0.5, 0), (0.5, 1, 0), (0, 0.5, 0), (0.5, 0.5, 0)],
                      [(10, 0, 1, list(range(9)))])
    mesh = jx.read_gmsh(path)
    assert mesh[0].degree == (2, 2)
    assert mesh[0].n_cp == 9


def test_duplicate_cells_rejected(tmp_path):
    path = write_mesh(tmp_path / "duplicate.msh",
                      [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                      [(3, 0, 1, [0, 1, 2, 3]), (3, 0, 1, [0, 1, 2, 3])])
    with pytest.raises(GmshImportError, match="duplicate"):
        jx.read_gmsh(path)


def test_shell_mesh_rejected(tmp_path):
    path = write_mesh(tmp_path / "shell.msh",
                      [(0, 0, 0), (1, 0, 0), (1, 1, 0.5), (0, 1, 0)],
                      [(3, 0, 1, [0, 1, 2, 3])])
    with pytest.raises(GmshImportError, match="parallel to XY"):
        jx.read_gmsh(path)


def test_active_gmsh_session_is_untouched(tmp_path):
    path = square(tmp_path)
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.model.add("existing work")
        with pytest.raises(RuntimeError, match="active session"):
            jx.read_gmsh(path)
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == "existing work"
    finally:
        gmsh.finalize()


def test_overlapping_boundary_names_rejected(tmp_path):
    path = tmp_path / "overlap.geo"
    path.write_text('''SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1};
Physical Curve("one") = {1};
Physical Curve("two") = {1};
''')
    with pytest.raises(GmshImportError, match="overlapping physical groups"):
        jx.read_gmsh(path, mesh_size=0.5)
    assert not gmsh.isInitialized()


def test_showcase_geometry_has_three_holes():
    path = Path(__file__).resolve().parents[1] / "src/jaxiga/examples/geometries/perforated_plate.geo"
    mesh = jx.read_gmsh(path)
    assert set(mesh.boundary_groups) >= {"fixed", "loaded", "holes"}
    assert mesh.min_scaled_jacobian > 0.05
    # Euler characteristic of this connected planar quad mesh is 1 - 3 holes.
    corners = [tuple(map(tuple, np.asarray(p.ctrl_pts))) for p in mesh]
    vertices = {point for cell in corners for point in cell}
    edges = {tuple(sorted((cell[a], cell[b]))) for cell in corners
             for a, b in [(0, 1), (1, 3), (3, 2), (2, 0)]}
    assert len(vertices) - len(edges) + len(mesh) == -2


def test_subdivided_adjacent_tets_share_all_three_interface_quads(tmp_path):
    path = write_mesh(tmp_path / "two_tets.msh",
                      [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)],
                      [(4, 1, 1, [0, 1, 2, 3]), (4, 1, 1, [0, 2, 1, 4])],
                      [(3, 1, "solid")])
    mesh = jx.read_gmsh(path)
    assert len(mesh) == 8
    space = jx.FunctionSpace([p.elevate(2) for p in mesh])
    assert len(space.topology.interfaces) == 15
    assert jx.integrate(lambda x: 1.0, space) == pytest.approx(1 / 3)


def test_reversed_hex_geometry_and_surface_label(tmp_path):
    vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    path = write_mesh(tmp_path / "hex.msh", vertices,
                      [(3, 2, 1, [4, 5, 6, 7]), (5, 3, 1, [1, 0, 3, 2, 5, 4, 7, 6])],
                      [(2, 2, "top"), (3, 3, "solid")])
    mesh = jx.read_gmsh(path)
    assert mesh.min_scaled_jacobian == pytest.approx(1)
    space = jx.FunctionSpace(mesh)
    np.testing.assert_allclose(np.asarray(space.cpts)[space.boundary('top').dofs, 2], 1)
    assert jx.integrate(lambda x: 1.0, space) == pytest.approx(1)


def test_missing_optional_dependency_message(tmp_path, monkeypatch):
    import builtins

    path = square(tmp_path)
    original = builtins.__import__

    def without_gmsh(name, *args, **kwargs):
        if name == "gmsh":
            raise ImportError("optional dependency unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_gmsh)
    with pytest.raises(ImportError, match=r'jaxiga\[gmsh\]'):
        jx.read_gmsh(path)


@pytest.mark.parametrize("failure", ["generation", "quality"])
@pytest.mark.parametrize("failed_attempts", [1, 2])
def test_meshing_failure_retries_cleanly_and_preserves_labels(
        tmp_path, monkeypatch, failure, failed_attempts):
    path = tmp_path / "plate.geo"
    path.write_text('''SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 2, 1};
Physical Curve("fixed") = {4};
Physical Surface("plate") = {1};
''')
    original = gmsh.model.mesh.generate
    calls = []

    def generate(dim):
        calls.append((gmsh.option.getNumber("Mesh.Algorithm"),
                      gmsh.option.getNumber("Mesh.RecombinationAlgorithm")))
        assert gmsh.option.getNumber("Mesh.MeshSizeFactor") == 1
        if len(calls) <= failed_attempts:
            if failure == "generation":
                gmsh.option.setNumber("Mesh.MeshSizeFactor", 99)
                raise Exception("simulated meshing failure")
            original(dim)
            # Collapse an edge after meshing to exercise actual quality checks.
            nodes = gmsh.model.mesh.getElementsByType(3)[1][:4]
            point = gmsh.model.mesh.getNode(int(nodes[1]))[0]
            gmsh.model.mesh.setNode(int(nodes[0]), point, [])
        else:
            original(dim)

    monkeypatch.setattr(gmsh.model.mesh, "generate", generate)
    mesh = jx.read_gmsh(path, mesh_size=0.5)
    assert calls == [(11, 1), (8, 1), (6, 0)][:failed_attempts + 1]
    assert mesh.meshing_method == ("blossom" if failed_attempts == 1 else "frontal_delaunay")
    assert "quasi_structured:" in mesh.fallback_reason
    assert ("blossom:" in mesh.fallback_reason) == (failed_attempts == 2)
    assert ("simulated" if failure == "generation" else "distorted") in mesh.fallback_reason
    assert mesh.min_scaled_jacobian >= 0.05
    assert len(mesh.cell_groups["plate"]) == len(mesh)
    space = jx.FunctionSpace(mesh)
    np.testing.assert_allclose(np.asarray(space.cpts)[space.boundary("fixed").dofs, 0], 0)
    assert jx.integrate(lambda x: 1.0, space) == pytest.approx(2)
    assert not gmsh.isInitialized()


def test_all_meshing_attempts_fail_with_reasons(tmp_path, monkeypatch):
    path = tmp_path / "plate.geo"
    path.write_text('SetFactory("OpenCASCADE"); Rectangle(1) = {0, 0, 0, 1, 1};')
    calls = []

    def fail(dim):
        calls.append(dim)
        raise Exception(f"failure {len(calls)}")

    monkeypatch.setattr(gmsh.model.mesh, "generate", fail)
    with pytest.raises(GmshImportError,
                       match="quasi_structured.*failure 1.*blossom.*failure 2.*frontal_delaunay.*failure 3"):
        jx.read_gmsh(path)
    assert calls == [2, 2, 2]
    assert not gmsh.isInitialized()


@pytest.mark.parametrize("subdivide", [True, False])
def test_fallback_respects_subdivision_flag(tmp_path, monkeypatch, subdivide):
    path = tmp_path / "plate.geo"
    path.write_text('SetFactory("OpenCASCADE"); Rectangle(1) = {0, 0, 0, 1, 1};')
    original = gmsh.model.mesh.generate

    def generate(dim):
        if gmsh.option.getNumber("Mesh.Algorithm") in (11, 8):
            raise Exception("preferred mesher unavailable")
        # Leave triangles to test the fallback's final conversion step.
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        original(dim)

    monkeypatch.setattr(gmsh.model.mesh, "generate", generate)
    if subdivide:
        mesh = jx.read_gmsh(path, mesh_size=0.5)
        assert mesh.subdivided
        assert all(p.n_cp == 4 for p in mesh)
        assert mesh.min_scaled_jacobian >= 0.05
    else:
        with pytest.raises(GmshImportError, match="quadrilateral"):
            jx.read_gmsh(path, mesh_size=0.5, subdivide=False)
    assert not gmsh.isInitialized()


@pytest.mark.parametrize("quad_algorithm", ["blossom", "frontal_delaunay"])
def test_transfinite_constraints_are_preserved(tmp_path, quad_algorithm):
    path = tmp_path / "structured.geo"
    path.write_text('''SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 1, 1};
Transfinite Curve {:} = 3;
Transfinite Surface {1};
Recombine Surface {1};
''')
    mesh = jx.read_gmsh(path, subdivide=False, quad_algorithm=quad_algorithm)
    assert mesh.meshing_method == quad_algorithm
    assert len(mesh) == 4
    assert not mesh.subdivided
    assert mesh.min_scaled_jacobian == pytest.approx(1)


def test_unknown_quad_algorithm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="quad_algorithm must be one of"):
        jx.read_gmsh(square(tmp_path), quad_algorithm="unknown")
    assert not gmsh.isInitialized()


def polynomial_mesh(tmp_path, dim, degree, serendip=False, reverse=False):
    """Independent Gmsh nodal fixture and reference evaluations, not Bernstein data."""
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        kind = gmsh.model.mesh.getElementType("Quadrangle" if dim == 2 else "Hexahedron",
                                              degree, serendip)
        _, _, _, count, ref, _ = gmsh.model.mesh.getElementProperties(kind)
        ref = np.asarray(ref).reshape(count, dim)
        xyz = np.zeros((count, 3))
        xyz[:, :dim] = (ref + 1) / 2
        xyz[:, 1] += 0.08 * (1 - ref[:, 0] ** 2)
        if dim == 3:
            xyz[:, 2] += 0.04 * (1 - ref[:, 0] ** 2) * ref[:, 1]
        if reverse:
            xyz[:, 0] = 1 - xyz[:, 0]
        rng = np.random.default_rng(42)
        samples = rng.uniform(-0.95, 0.95, (17, dim))
        query = np.zeros((len(samples), 3))
        query[:, :dim] = samples
        if reverse:
            query[:, 0] *= -1
        _, basis, _ = gmsh.model.mesh.getBasisFunctions(kind, query.ravel(), "Lagrange")
        expected = np.asarray(basis).reshape(len(samples), count) @ xyz[:, :dim]
        face_kind = gmsh.model.mesh.getElementType("Line" if dim == 2 else "Quadrangle",
                                                   degree, serendip if dim == 3 else False)
        _, _, _, face_count, face_ref, _ = gmsh.model.mesh.getElementProperties(face_kind)
        face_ref = np.asarray(face_ref).reshape(face_count, dim - 1)
        face = []
        for point in face_ref:
            face.append(int(np.flatnonzero(np.all(np.isclose(ref, np.r_[-1, point]), axis=1))[0]))
    finally:
        gmsh.finalize()
    path = write_mesh(tmp_path / "curved.msh", xyz,
                      [(face_kind, 2, 1, face), (kind, 3, 1, list(range(count)))],
                      [(dim - 1, 2, "boundary"), (dim, 3, "solid")])
    return path, (samples + 1) / 2, expected


@pytest.mark.parametrize("dim,degree,serendip", [
    (2, 2, False), (2, 3, False), (2, 4, False), (2, 2, True),
    (3, 2, False), (3, 3, False), (3, 4, False), (3, 2, True),
])
def test_high_order_polynomial_geometry_matches_gmsh(tmp_path, dim, degree, serendip):
    path, parameters, expected = polynomial_mesh(tmp_path, dim, degree, serendip)
    mesh = jx.read_gmsh(path)
    assert mesh[0].degree == (degree,) * dim
    np.testing.assert_allclose(evaluate_patch(mesh[0], parameters), expected, atol=2e-12)
    # Both operations must preserve the imported polynomial, with no CAD needed.
    refined = mesh.elevate(degree + 1, to_cad=False).refine(to_cad=False)
    np.testing.assert_allclose(evaluate_patch(refined[0], parameters), expected, atol=3e-12)
    assert mesh.boundary_groups == {"boundary": ((0, "u0"),)}
    assert mesh.min_scaled_jacobian > 0


@pytest.mark.parametrize("dim", [2, 3])
def test_reversed_high_order_geometry_and_labels(tmp_path, dim):
    path, parameters, expected = polynomial_mesh(tmp_path, dim, 3, reverse=True)
    mesh = jx.read_gmsh(path)
    np.testing.assert_allclose(evaluate_patch(mesh[0], parameters), expected, atol=2e-12)
    assert mesh.boundary_groups == {"boundary": ((0, "u1"),)}


@pytest.fixture
def curved_cad(tmp_path):
    path = tmp_path / "ring.geo"
    path.write_text('''SetFactory("OpenCASCADE");
Rectangle(1) = {-2, -2, 0, 4, 4};
Disk(2) = {0, 0, 0, 1, 1};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }
hole() = Curve In BoundingBox{-1.01, -1.01, -0.01, 1.01, 1.01, 0.01};
Physical Curve("hole") = {hole()};
Physical Surface("domain") = {1};
Mesh.MinimumCirclePoints = 8;
''')
    return path


def test_cad_refinement_improves_geometry_and_preserves_interfaces(curved_cad):
    coarse = jx.read_gmsh(curved_cad, mesh_size=1.5, geometry_order=2)
    assert coarse.geometry_error > 1e-8
    ordinary = coarse.refine(to_cad=False)
    assert ordinary.geometry_error == coarse.geometry_error
    # CAD interpolation improves with resolution, but the maximum error need
    # not decrease after every single small change to an interpolation space.
    for fine in (coarse.refine(2), coarse.elevate(4),
                 coarse.insert_knots([0.25, 0.5, 0.75], [0.25, 0.5, 0.75])):
        assert len(fine) == len(coarse)
        assert fine.geometry_error < coarse.geometry_error / 2
        assert fine.min_scaled_jacobian >= 0.05
        assert np.all(fine.jacobian_lower_bounds > 0)
        assert fine.boundary_groups == coarse.boundary_groups
        assert len(jx.FunctionSpace(fine).topology.interfaces) == len(jx.FunctionSpace(coarse).topology.interfaces)
    assert not gmsh.isInitialized()


def test_auto_sizing_reduces_target_until_cad_tolerance(curved_cad):
    mesh = jx.read_gmsh(curved_cad, mesh_size="auto", geometry_order=2, geometry_tolerance=1e-5)
    assert mesh.geometry_error <= 1e-5
    assert mesh.min_scaled_jacobian >= 0.05
    assert mesh.mesh_size == mesh.sizing_history[-1]["mesh_size"]
    assert len(mesh.sizing_history) >= 2
    assert not gmsh.isInitialized()


def test_auto_sizing_reports_exhausted_budget(curved_cad):
    with pytest.raises(GmshImportError, match="automatic sizing did not meet"):
        jx.read_gmsh(curved_cad, mesh_size="auto", geometry_order=1,
                     geometry_tolerance=1e-12, max_sizing_steps=1)
    assert not gmsh.isInitialized()


def test_cad_refinement_requires_unchanged_source(curved_cad, tmp_path):
    imported = jx.read_gmsh(square(tmp_path))
    with pytest.raises(GmshImportError, match="CAD-aware refinement requires"):
        imported.refine()
    mesh = jx.read_gmsh(curved_cad, mesh_size=1.5, geometry_order=2)
    curved_cad.write_text(curved_cad.read_text() + "\n// changed\n")
    with pytest.raises(GmshImportError, match="source CAD file is missing or changed"):
        mesh.refine()
    assert not gmsh.isInitialized()


def test_curved_hex_cad_refinement(tmp_path):
    path = tmp_path / "quarter_tube.geo"
    path.write_text('''SetFactory("OpenCASCADE");
Point(1) = {1, 0, 0}; Point(2) = {2, 0, 0};
Point(3) = {0, 2, 0}; Point(4) = {0, 1, 0}; Point(5) = {0, 0, 0};
Line(1) = {1, 2}; Circle(2) = {2, 5, 3};
Line(3) = {3, 4}; Circle(4) = {4, 5, 1};
Curve Loop(1) = {1, 2, 3, 4}; Plane Surface(1) = {1};
Transfinite Curve {:} = 2; Transfinite Surface {1}; Recombine Surface {1};
out() = Extrude {0, 0, 1} { Surface{1}; Layers{2}; Recombine; };
Physical Surface("top") = {out(0)};
Physical Volume("solid") = {out(1)};
''')
    mesh = jx.read_gmsh(path, geometry_order=2)
    assert len(mesh) == 2
    assert mesh[0].degree == (2, 2, 2)
    assert mesh.geometry_error > 1e-5
    fine = mesh.refine(2)
    assert fine.geometry_error < mesh.geometry_error / 2
    assert fine.min_scaled_jacobian > 0.05
    assert fine.cell_groups == mesh.cell_groups
    assert fine.boundary_groups == mesh.boundary_groups
    assert len(jx.FunctionSpace(fine).topology.interfaces) == 1
    # The exact volume is a quarter annulus, extruded by one unit.
    volume = jx.integrate(lambda x: 1.0, jx.FunctionSpace(fine))
    assert volume == pytest.approx(3 * np.pi / 4, rel=1e-3)


def test_high_order_simplex_is_not_silently_linearized(tmp_path):
    path = write_mesh(tmp_path / "tri6.msh",
                      [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                       (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0)],
                      [(9, 0, 1, list(range(6)))])
    with pytest.raises(GmshImportError, match="high-order mixed/simplex"):
        jx.read_gmsh(path)


def test_cad_refinement_samples_lie_on_circle(curved_cad):
    from jaxiga.geometry.gmsh_geometry import spline_basis, tensor_grid
    mesh = jx.read_gmsh(curved_cad, mesh_size=1.5, geometry_order=2).refine()
    for index, side in mesh.boundary_groups["hole"]:
        patch = mesh[index]
        axes = [np.array([np.mean(k[i + 1:i + p + 1]) for i in range(n)])
                for k, p, n in zip(patch.knots, patch.degree, patch.n_cp_per_dir)]
        axes["uvw".index(side[0])] = [int(side[1])]
        points = spline_basis(patch, tensor_grid(axes)) @ np.asarray(patch.ctrl_pts)
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1, atol=1e-10)


def test_cad_refinement_preserves_active_session(curved_cad):
    mesh = jx.read_gmsh(curved_cad, mesh_size=1.5, geometry_order=2)
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.model.add("user model")
        with pytest.raises(RuntimeError, match="active session"):
            mesh.refine()
        assert gmsh.model.getCurrent() == "user model"
    finally:
        gmsh.finalize()


def test_high_order_fold_is_rejected_even_with_valid_corner_geometry(tmp_path):
    gmsh.initialize([], readConfigFiles=False)
    try:
        kind = gmsh.model.mesh.getElementType("Quadrangle", 3)
        _, _, _, count, ref, _ = gmsh.model.mesh.getElementProperties(kind)
        uv = (np.asarray(ref).reshape(count, 2) + 1) / 2
    finally:
        gmsh.finalize()
    xyz = np.column_stack((uv, np.zeros(count)))
    v = uv[:, 1]
    xyz[:, 1] -= 10 * v * (1 - v) * (v - 0.5)
    path = write_mesh(tmp_path / "fold.msh", xyz, [(kind, 0, 1, list(range(count)))])
    with pytest.raises(GmshImportError, match="inverted|positive geometry Jacobian"):
        jx.read_gmsh(path)
    assert not gmsh.isInitialized()


def test_curved_import_preserves_nonzero_xy_plane(curved_cad):
    content = curved_cad.read_text().replace("{-2, -2, 0, 4, 4}", "{-2, -2, 3, 4, 4}")
    content = content.replace("{0, 0, 0, 1, 1}", "{0, 0, 3, 1, 1}")
    content = content.replace("-1.01, -1.01, -0.01, 1.01, 1.01, 0.01", "-1.01, -1.01, 2.99, 1.01, 1.01, 3.01")
    curved_cad.write_text(content)
    mesh = jx.read_gmsh(curved_cad, mesh_size=1.5, geometry_order=2)
    assert mesh._plane_z == pytest.approx(3)
    fine = mesh.refine(2)
    assert fine.geometry_error < mesh.geometry_error / 2
    assert "hole" in fine.boundary_groups
