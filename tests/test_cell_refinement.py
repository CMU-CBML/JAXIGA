"""Uniform subdivision must preserve rational maps and conforming interfaces."""

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jaxiga as jx
from jaxiga.geometry.gmsh_geometry import tensor_grid
from jaxiga.geometry.mixed import ordered_interfaces
from jaxiga.geometry.nurbs import evaluate_patch
from jaxiga.geometry.refinement import _simplex_children, _simplex_diagonal
from jaxiga.space.pointset import PointSet


def box(dim, start=0):
    xyz = tensor_grid([[start, start + 1]] + [[0, 1]] * (dim - 1))
    return jx.Patch.create([[0, 0, 1, 1]] * dim, (1,) * dim, xyz)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [False, True])
@pytest.mark.parametrize("degree", [1, 3, 6])
def test_rational_subdivision_is_exact(dim, simplex, degree):
    patch = (
        jx.SimplexPatch.from_vertices(np.vstack([np.zeros(dim), np.eye(dim)]))
        if simplex
        else box(dim)
    ).elevate(degree)
    rng = np.random.default_rng(153)
    patch = replace(
        patch,
        weights=jnp.asarray(rng.uniform(0.5, 2, patch.n_cp)),
        ctrl_pts=patch.ctrl_pts
        + jnp.asarray(rng.normal(0, 0.015, patch.ctrl_pts.shape)),
    )
    fine = jx.refine_cells(patch)
    assert len(fine) == 2**dim
    points = (
        rng.dirichlet(np.ones(dim + 1), 17)[:, 1:]
        if simplex
        else rng.uniform(0, 1, (17, dim))
    )
    from itertools import product

    for child, mapping in zip(
        fine,
        _simplex_children(dim, _simplex_diagonal(patch))
        if simplex
        else product((0, 1), repeat=dim),
    ):
        parent_ref = (
            np.column_stack([1 - points.sum(axis=1), points]) @ mapping
            if simplex
            else (points + np.asarray(mapping)) / 2
        )
        np.testing.assert_allclose(
            evaluate_patch(child, points),
            evaluate_patch(patch, parent_ref),
            atol=2e-13,
            rtol=2e-13,
        )
        assert child.degree == patch.degree
        assert np.min(child.weights) >= np.min(patch.weights) - 1e-14
        assert np.max(child.weights) <= np.max(patch.weights) + 1e-14
    # For affine reference geometry, the children partition rather than overlap.
    if simplex:
        determinants = [np.linalg.det(v[1:] - v[0]) for v in _simplex_children(dim)]
        np.testing.assert_allclose(determinants, 1 / 2**dim)


@pytest.mark.parametrize("dim", [2, 3])
def test_refined_mixed_mesh_continuity_volume_and_linear_solve(dim):
    patches = [box(dim), *jx.split_to_simplices(box(dim, 1))]
    cells = jx.refine_cells(patches, n=1)
    space = jx.FunctionSpace(cells)
    assert float(jx.integrate(lambda x: 1.0, space)) == pytest.approx(2, abs=3e-12)
    exact = lambda x: 1 + x.sum()
    problem = jx.Poisson(
        space,
        dirichlet=[
            jx.DirichletBC(lambda x, p: exact(x), where=lambda x: np.ones(len(x), bool))
        ],
    )
    assert jx.errornorm(jx.solve(problem, params={"a0": 1.0}), exact, "L2") < 3e-12
    pairs, exterior = ordered_interfaces(cells)
    # No internal face may silently become an exterior boundary.
    for face in exterior:
        physical = evaluate_patch(cells[face.cell], face.reference)
        assert any(
            np.allclose(physical[:, d], end)
            for d in range(dim)
            for end in (0, 2 if d == 0 else 1)
        )
    solution = jx.Solution(space, jnp.sin(jnp.arange(space.n_scalar_basis)))
    ea, eb, ra, rb = [], [], [], []
    for a, b in pairs:
        bary = np.random.default_rng(42).dirichlet(np.ones(len(b.vertices)), 5)
        ea.append(a.cell)
        eb.append(b.cell)
        rb.append(bary @ b.reference)
        ra.append(
            bary @ np.array([a.reference[a.vertices.index(v)] for v in b.vertices])
        )
    ba = jx.evaluate(space, PointSet(np.array(ea), 2 * np.array(ra) - 1))
    bb = jx.evaluate(space, PointSet(np.array(eb), 2 * np.array(rb) - 1))
    np.testing.assert_allclose(ba.x, bb.x, atol=2e-12)
    np.testing.assert_allclose(solution.at(basis=ba), solution.at(basis=bb), atol=3e-12)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("simplex", [False, True])
def test_labels_survive_repeated_refinement(dim, simplex):
    p = box(dim)
    p = p.with_labels(**{side: f"boundary_{side}" for side in p.side_names()})
    coarse = jx.split_to_simplices(p) if simplex else [p]
    fine = jx.refine_cells(coarse, n=2)
    assert len(fine) == len(coarse) * 2 ** (2 * dim)
    assert {v for p in fine for v in p.label_map.values()} == set(p.label_map.values())
    _, exterior = ordered_interfaces(fine)
    labeled = {(e, side) for e, patch in enumerate(fine) for side in patch.label_map}
    assert labeled == {(face.cell, face.side) for face in exterior}
    repeated = jx.refine_cells(jx.refine_cells(coarse))
    for a, b in zip(fine, repeated):
        np.testing.assert_array_equal(a.ctrl_pts, b.ctrl_pts)
        np.testing.assert_array_equal(a.weights, b.weights)
        assert a.labels == b.labels


def test_refinement_validation_and_noop():
    p = box(2)
    assert jx.refine_cells([p], 0) == [p]
    for n in [-1, 1.5, True]:
        with pytest.raises(ValueError, match="non-negative integer"):
            jx.refine_cells(p, n)
    with pytest.raises(ValueError, match="at least one cell"):
        jx.refine_cells([])
    with pytest.raises(ValueError, match="single-span"):
        jx.refine_cells(p.refine(1))
    with pytest.raises(ValueError, match="one dimension"):
        jx.refine_cells([p, box(3)])


def test_repeated_tet_subdivision_avoids_elongating_reference_cells():
    patch = jx.SimplexPatch.from_vertices(np.vstack([np.zeros(3), np.eye(3)]))
    cells = jx.refine_cells(patch, n=3)
    vertices = np.array([np.asarray(p.ctrl_pts)[p.vertex_indices] for p in cells])
    jacobians = vertices[:, 1:] - vertices[:, :1]
    np.testing.assert_allclose(np.linalg.det(jacobians), 1 / 512, atol=1e-16)
    # A fixed octahedron diagonal instead produces condition numbers > 8 here.
    assert np.linalg.cond(jacobians).max() < 6


def test_refined_cylinder_geometry_and_pressure_convergence(tmp_path):
    pytest.importorskip("gmsh")
    geometry = Path(jx.__file__).parent / "examples/geometries/quarter_cylinder.geo"
    mesh = jx.read_gmsh(geometry, geometry_order=2, rational_geometry=True)
    patches = [mesh[0], *jx.split_to_simplices(mesh[1])]
    errors = []
    for level in [0, 1]:
        cells = jx.refine_cells(patches, level)
        space = jx.FunctionSpace(cells, vec=3)
        assert space.n_elems == 7 * 8**level
        assert float(jx.integrate(lambda x: 1.0, space)) == pytest.approx(
            1.5 * np.pi, rel=1e-8
        )
        for label, radius in [("inner", 1), ("outer", 2)]:
            b = jx.evaluate(space, jx.boundary_gauss(space, label))
            np.testing.assert_allclose(
                np.linalg.norm(b.x[..., :2], axis=-1), radius, atol=2e-12
            )
            assert float(b.w.sum()) == pytest.approx(radius * np.pi, rel=1e-8)
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
        b = jx.evaluate(space, jx.gauss(space))
        xy = np.asarray(b.x)[..., :2]
        r = np.linalg.norm(xy, axis=-1)
        ur = 1.3 / 1e5 * (0.4 * 10 / 3 * r + (40 / 3) / r)
        expected = np.concatenate(
            [ur[..., None] * xy / r[..., None], np.zeros((*r.shape, 1))], axis=-1
        )
        u = np.asarray(solution.at(basis=b))
        errors.append(
            np.sqrt(
                np.sum(b.w[..., None] * (u - expected) ** 2)
                / np.sum(b.w[..., None] * expected**2)
            )
        )
    assert errors[0] < 0.02
    assert errors[1] < errors[0] / 3, errors
    output = solution.to_vtk(
        tmp_path / "refined_cylinder.vtu", n=2, fields=("von_mises",)
    )
    assert Path(output).stat().st_size > 1000
