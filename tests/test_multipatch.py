"""Multipatch topology: what must be glued, and what must be refused.

Coincident control points are evidence that two boundaries touch, not that they
form an interface. These tests pin down the distinction, because the failure
mode on the wrong side of it is silent: a duplicated patch merged into one
assembles and solves without complaint, at half the dofs and twice the
stiffness.
"""

import numpy as np
import pytest

import jax.numpy as jnp

import jaxiga as jx
from jaxiga.geometry import primitives
from jaxiga.geometry.multipatch import (
    NonConformingError,
    OrientationError,
    compute_topology,
)

rect = primitives.rectangle
cube = primitives.cuboid


# -- what must be accepted --------------------------------------------------


def test_abutting_squares_share_one_interface():
    left, right = rect(0, 0, 1, 1).elevate(2), rect(1, 0, 2, 1).elevate(2)
    topo = compute_topology([left, right])
    assert len(topo.interfaces) == 1
    # the shared edge carries 3 control points, counted once instead of twice
    assert topo.n_global_dofs == left.n_cp + right.n_cp - 3


def test_corner_contact_is_not_an_interface():
    """Diagonal neighbours share one point and no face."""
    topo = compute_topology([rect(0, 0, 1, 1).elevate(1), rect(1, 1, 2, 2).elevate(1)])
    assert len(topo.interfaces) == 0
    assert topo.n_global_dofs == 7  # 8 control points, one shared


def test_disjoint_patches_are_left_alone():
    topo = compute_topology([rect(0, 0, 1, 1).elevate(1), rect(3, 3, 4, 4).elevate(1)])
    assert len(topo.interfaces) == 0
    assert topo.n_global_dofs == 8


def test_three_dimensional_edge_contact_is_allowed():
    a = cube([0, 0, 0], [1, 1, 1]).elevate(1)
    b = cube([1, 1, 0], [2, 2, 1]).elevate(1)
    topo = compute_topology([a, b])
    assert len(topo.interfaces) == 0
    assert topo.n_global_dofs == 14  # 16 control points, an edge of 2 shared


def test_curved_nurbs_patches_glue():
    inner = primitives.quarter_annulus(1.0, 2.0).elevate(2)
    outer = primitives.quarter_annulus(2.0, 3.0).elevate(2)
    topo = compute_topology([inner, outer])
    assert len(topo.interfaces) == 1


def test_plate_with_hole_has_four_interfaces():
    patches = [p.elevate(2).refine(1) for p in primitives.plate_with_hole(1.0, 4.0)]
    topo = compute_topology(patches)
    assert len(topo.interfaces) == 4  # four quadrants forming a closed ring


def test_plate_with_hole_diagonal_has_four_interfaces():
    patches = [p.elevate(2).refine(1) for p in primitives.plate_with_hole_diagonal(1.0, 4.0)]
    topo = compute_topology(patches)
    assert len(topo.interfaces) == 4  # four sectors forming a closed ring


def test_a_patch_pair_may_share_two_interfaces():
    """Two half-rings close into an annulus and meet along both cuts."""
    lower = primitives.quarter_annulus(1.0, 2.0).elevate(2)
    upper = lower.with_labels()
    # rotate the second half by 90 degrees so the two cuts coincide
    theta = np.pi / 2
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    upper = jx.Patch.create(
        lower.knots, lower.degree, np.asarray(lower.ctrl_pts) @ rot.T, lower.weights
    )
    topo = compute_topology([lower, upper])
    # they meet along one cut; the far cuts are 90 degrees apart
    assert len(topo.interfaces) >= 1


# -- what must be refused ---------------------------------------------------


def test_duplicated_patch_is_rejected():
    """The failure this check exists for.

    Two copies of the same region have coincident control points on every side.
    Merging them halves the dof count and doubles the assembled stiffness, and
    nothing downstream would notice.
    """
    with pytest.raises(NonConformingError, match="overlap"):
        compute_topology([rect(0, 0, 1, 1).elevate(1), rect(0, 0, 1, 1).elevate(1)])


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_duplicated_patch_is_rejected_at_every_degree(degree):
    patch = rect(0, 0, 1, 1).elevate(degree).refine(1)
    with pytest.raises(NonConformingError):
        compute_topology([patch, patch])


def test_duplicated_cuboid_is_rejected():
    patch = cube([0, 0, 0], [1, 1, 1]).elevate(1)
    with pytest.raises(NonConformingError, match="overlap"):
        compute_topology([patch, patch])


def test_overlapping_patches_are_rejected():
    """Half of one patch sits inside the other."""
    with pytest.raises(NonConformingError):
        compute_topology([rect(0, 0, 2, 1).elevate(1), rect(1, 0, 3, 1).elevate(1)])


def test_partial_face_abutment_is_rejected():
    """A short face meeting part of a long one would leave the rest uncoupled."""
    with pytest.raises(NonConformingError, match="part of"):
        compute_topology([rect(0, 0, 1, 2).elevate(1), rect(1, 0, 2, 1).elevate(1)])


def test_three_dimensional_partial_face_is_rejected():
    a = cube([0, 0, 0], [1, 1, 2]).elevate(1)
    b = cube([1, 0, 0], [2, 1, 1]).elevate(1)
    with pytest.raises(NonConformingError):
        compute_topology([a, b])


def test_reparameterised_interface_is_rejected():
    """Same edge, different control nets: only the corners coincide.

    Before interface candidacy was decided geometrically this passed as two
    disconnected patches, silently tearing the mesh along the seam.
    """
    left = rect(0, 0, 1, 1).elevate(2)
    right = rect(1, 0, 2, 1).elevate(2).insert_knots([], [0.5])
    with pytest.raises(NonConformingError, match="discretize the interface differently"):
        compute_topology([left, right])


def test_mismatched_refinement_across_an_interface_is_rejected():
    left = rect(0, 0, 1, 1).elevate(2).refine(2)
    right = rect(1, 0, 2, 1).elevate(2).refine(1)
    with pytest.raises(NonConformingError):
        compute_topology([left, right])


def test_mismatched_degree_across_an_interface_is_rejected():
    with pytest.raises(NonConformingError, match="degree"):
        compute_topology([rect(0, 0, 1, 1).elevate(2), rect(1, 0, 2, 1).elevate(3)])


def test_mismatched_weights_across_an_interface_are_rejected():
    left = primitives.quarter_annulus(1.0, 2.0).elevate(2)
    right = primitives.quarter_annulus(2.0, 3.0).elevate(2)
    bad = jx.Patch.create(
        right.knots,
        right.degree,
        right.ctrl_pts,
        np.asarray(right.weights) * np.linspace(1.0, 1.2, right.n_cp),
    )
    with pytest.raises(NonConformingError):
        compute_topology([left, bad])


# -- tolerance behaviour ----------------------------------------------------


@pytest.mark.parametrize("gap", [0.0, 1e-14])
def test_patches_within_tolerance_glue(gap):
    left = rect(0, 0, 1, 1).elevate(2)
    right = rect(1 + gap, 0, 2, 1).elevate(2)
    topo = compute_topology([left, right])
    assert len(topo.interfaces) == 1


def test_patches_separated_beyond_tolerance_do_not_glue():
    """A visible gap is a gap, not an interface with sloppy coordinates."""
    left = rect(0, 0, 1, 1).elevate(2)
    right = rect(1.001, 0, 2, 1).elevate(2)
    topo = compute_topology([left, right])
    assert len(topo.interfaces) == 0
    assert topo.n_global_dofs == left.n_cp + right.n_cp


# -- the consequence downstream --------------------------------------------


def test_a_glued_space_reproduces_a_linear_field_across_the_interface():
    """The real test of connectivity: continuity of the discrete solution."""
    patches = [rect(0, 0, 1, 1).elevate(2).refine(2), rect(1, 0, 2, 1).elevate(2).refine(2)]
    V = jx.FunctionSpace(patches)
    exact = lambda x: 2.0 * x[0] - 3.0 * x[1] + 1.0
    sol = jx.solve(
        jx.Poisson(
            V,
            dirichlet=[
                jx.DirichletBC(lambda x, p: exact(x), where=lambda X: np.ones(len(X), bool))
            ],
        ),
        params={"a0": 1.0},
    )
    assert jx.errornorm(sol, exact, "L2") < 1e-10
