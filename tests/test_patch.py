"""Patch refinement, exact geometry and refinement-operator invariance."""

import numpy as np
import pytest

import jaxiga.config  # noqa: F401  (enables x64)
from jaxiga.geometry import primitives
from jaxiga.geometry.nurbs import Patch, evaluate_patch






PRIMITIVE_CASES = [
    ("quadrilateral", lambda: primitives.rectangle(0, 0, 2, 1)),
    ("quarter_annulus", lambda: primitives.quarter_annulus(1.0, 4.0)),
    ("disk", lambda: primitives.disk([0.0, 0.0], 2.0)),
] + [
    (f"plate_with_hole_q{q}",
     lambda q=q: primitives.plate_with_hole_quadrant(1.0, 4.0, q))
    for q in (1, 2, 3, 4)
]


# -- geometry invariance ---------------------------------------------------


@pytest.mark.parametrize("name,make_new", PRIMITIVE_CASES, ids=[c[0] for c in PRIMITIVE_CASES])
def test_refinement_preserves_geometry(name, make_new):
    """Elevation and knot insertion are exact: the surface must not move."""
    patch = make_new()
    rng = np.random.default_rng(1234)
    xi = rng.uniform(0.0, 1.0, size=(40, patch.dim))

    base = evaluate_patch(patch, xi)
    for refined in (patch.elevate(4), patch.refine(2), patch.elevate(4).refine(2)):
        np.testing.assert_allclose(evaluate_patch(refined, xi), base, rtol=1e-11, atol=1e-12)


def test_disk_is_exactly_circular():
    """A NURBS disk must reproduce the circle exactly, not approximately."""
    patch = primitives.disk([0.5, -0.25], 2.0).elevate(3).refine(2)
    # the v1 edge of the unit-disk control net traces the boundary circle
    t = np.linspace(0, 1, 50)
    edge = evaluate_patch(patch, np.stack([t, np.ones_like(t)], axis=1))
    r = np.hypot(edge[:, 0] - 0.5, edge[:, 1] + 0.25)
    np.testing.assert_allclose(r, 2.0, rtol=1e-12)


def test_plate_with_hole_boundary_radius():
    """The 'hole' side must lie exactly on the circle of the given radius."""
    for q in (1, 2, 3, 4):
        patch = primitives.plate_with_hole_quadrant(1.5, 4.0, q).elevate(4).refine(2)
        t = np.linspace(0, 1, 37)
        hole = evaluate_patch(patch, np.stack([t, np.zeros_like(t)], axis=1))
        np.testing.assert_allclose(np.hypot(*hole.T), 1.5, rtol=1e-11)


def test_plate_with_hole_sector_geometry_is_exact():
    """Hole edge on the circle, outer edge on one straight side of the square."""
    rad, side = 1.5, 4.0
    for k in range(4):
        patch = primitives.plate_with_hole_sector(rad, side, k).elevate(3).refine(2)
        t = np.linspace(0, 1, 41)
        hole = evaluate_patch(patch, np.stack([t, np.zeros_like(t)], axis=1))
        np.testing.assert_allclose(np.hypot(*hole.T), rad, rtol=1e-11)

        outer = evaluate_patch(patch, np.stack([t, np.ones_like(t)], axis=1))
        # the side faced by sector k is the one whose outward normal is at
        # k * 90 degrees, so exactly one coordinate is pinned to +-side
        normal = np.array([np.cos(k * np.pi / 2), np.sin(k * np.pi / 2)])
        np.testing.assert_allclose(outer @ normal, side, atol=1e-11)

        # the cut edges lie on the diagonals
        for u in (0.0, 1.0):
            cut = evaluate_patch(patch, np.stack([np.full(7, u), np.linspace(0, 1, 7)], 1))
            assert np.abs(np.abs(cut[:, 0]) - np.abs(cut[:, 1])).max() < 1e-11


def test_plate_with_hole_sector_has_no_interior_knot():
    """The point of the layout: neither direction needs a repeated knot."""
    patch = primitives.plate_with_hole_sector(1.0, 4.0, 0)
    u, v = patch.knot_arrays()
    np.testing.assert_allclose(u, [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(v, [0, 0, 1, 1])
    assert patch.degree == (2, 1)


def test_plate_with_hole_labels_lie_on_their_axes():
    for q in (1, 2, 3, 4):
        patch = primitives.plate_with_hole_quadrant(1.0, 4.0, q)
        labels = patch.label_map
        assert labels["v0"] == "hole" and labels["v1"] == "outer"
        assert sorted([labels["u0"], labels["u1"]]) == ["sym_x", "sym_y"]

        for side, label in ((s, labels[s]) for s in ("u0", "u1")):
            u = 0.0 if side == "u0" else 1.0
            pts = evaluate_patch(patch, np.stack([np.full(7, u), np.linspace(0, 1, 7)], 1))
            axis = 0 if label == "sym_x" else 1
            np.testing.assert_allclose(pts[:, axis], 0.0, atol=1e-12)


# -- refinement operator ---------------------------------------------------


@pytest.mark.parametrize("name,make_new", PRIMITIVE_CASES, ids=[c[0] for c in PRIMITIVE_CASES])
def test_refinement_operator(name, make_new):
    """P maps coarse homogeneous control points onto the fine ones."""
    coarse = make_new()
    fine = coarse.elevate(4).refine(2)

    P = coarse.refinement_operator(fine)
    assert P.shape == (fine.n_cp, coarse.n_cp)
    np.testing.assert_allclose(
        P @ np.asarray(coarse.homogeneous()), np.asarray(fine.homogeneous()), rtol=1e-11, atol=1e-12
    )


def test_refinement_operator_rejects_coarsening():
    patch = primitives.quadrilateral([[0, 0], [1, 0], [1, 1], [0, 1]]).elevate(3)
    coarser = primitives.quadrilateral([[0, 0], [1, 0], [1, 1], [0, 1]])
    with pytest.raises(ValueError, match="lower degree"):
        patch.refinement_operator(coarser)


# -- dataclass / pytree hygiene -------------------------------------------


def test_patch_is_a_pytree_roundtrip():
    import jax

    patch = primitives.quarter_annulus(1.0, 2.0).elevate(3)
    leaves, treedef = jax.tree_util.tree_flatten(patch)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.degree == patch.degree
    assert rebuilt.knots == patch.knots
    assert rebuilt.labels == patch.labels
    np.testing.assert_array_equal(np.asarray(rebuilt.ctrl_pts), np.asarray(patch.ctrl_pts))


def test_patch_metadata_is_hashable():
    """Metadata must be hashable or jit will reject the pytree."""
    patch = primitives.disk([0, 0], 1.0)
    hash((patch.knots, patch.degree, patch.labels))


def test_knots_are_normalized():
    patch = Patch.create(knots=[[2.0, 2.0, 5.0, 8.0, 8.0]], degree=(1,),
                         ctrl_pts=[[0.0], [1.0], [3.0]])
    np.testing.assert_allclose(np.asarray(patch.knots[0]), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_construction_validates_sizes():
    with pytest.raises(ValueError, match="control points"):
        Patch.create(knots=[[0, 0, 1, 1]], degree=(1,), ctrl_pts=[[0.0], [1.0], [2.0]])
    with pytest.raises(ValueError, match="weights must be"):
        Patch.create(knots=[[0, 0, 1, 1]], degree=(1,), ctrl_pts=[[0.0], [1.0]], weights=[1.0])
    with pytest.raises(ValueError, match="knot vectors but"):
        Patch.create(knots=[[0, 0, 1, 1]], degree=(1, 1), ctrl_pts=[[0.0], [1.0]])


def test_elevate_rejects_degree_reduction():
    patch = primitives.quadrilateral([[0, 0], [1, 0], [1, 1], [0, 1]]).elevate(3)
    with pytest.raises(ValueError, match="cannot lower degree"):
        patch.elevate(2)


def test_with_labels_rejects_unknown_side():
    patch = primitives.quadrilateral([[0, 0], [1, 0], [1, 1], [0, 1]])
    with pytest.raises(ValueError, match="unknown side"):
        patch.with_labels(w0="nope")


def test_legacy_side_names_accepted():
    patch = primitives.quadrilateral([[0, 0], [1, 0], [1, 1], [0, 1]])
    assert patch.with_labels(down="fixed").label_map["v0"] == "fixed"


def test_interval_and_cuboid_shapes():
    assert primitives.interval(0.0, 2.0).n_cp == 2
    box = primitives.cuboid([0, 0, 0], [1, 2, 3])
    assert box.dim == 3 and box.n_cp == 8
    np.testing.assert_allclose(np.asarray(box.ctrl_pts)[0], [0, 0, 0])
    np.testing.assert_allclose(np.asarray(box.ctrl_pts)[-1], [1, 2, 3])
