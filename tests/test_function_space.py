"""FunctionSpace connectivity, boundaries, multipatch gluing and dof maps."""

import numpy as np
import pytest

import jaxiga.config  # noqa: F401
from jaxiga.geometry import primitives
from jaxiga.geometry.multipatch import NonConformingError, compute_topology
from jaxiga.space.function_space import FunctionSpace, expand_dofs, make_dofmap










def test_boundary_elements_touch_their_side():
    V = FunctionSpace(primitives.quadrilateral([[0, 0], [1, 0], [1, 1], [0, 1]]).elevate(2).refine(2))
    for label, direction, end in [("left", 0, 0), ("right", 0, 1), ("bottom", 1, 0), ("top", 1, 1)]:
        bset = V.boundaries[label]
        col = direction if end == 0 else V.dim + direction
        expected = 0.0 if end == 0 else 1.0
        np.testing.assert_allclose(V.elem_vertex[bset.elems, col], expected, atol=1e-14)
        assert set(bset.sides.tolist()) == {2 * direction + end}


# -- multipatch ------------------------------------------------------------






def test_multipatch_finds_all_interfaces():
    patches = [primitives.plate_with_hole_quadrant(1.0, 4.0, q).elevate(3).refine(1)
               for q in (2, 3, 4, 1)]
    V = FunctionSpace(patches)
    # four quadrants in a ring -> four interfaces, and the cut edges are interior
    assert len(V.topology.interfaces) == 4
    assert sorted(V.boundaries) == ["hole", "outer"]


def test_two_patch_gluing_shares_interface_dofs():
    left = primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1)
    right = primitives.rectangle(1, 0, 2, 1).elevate(2).refine(1)
    V = FunctionSpace([left, right])

    # shared edge has (n_cp in v) dofs counted once
    assert V.n_scalar_basis == 2 * left.n_cp - left.n_cp_per_dir[1]
    assert len(V.topology.interfaces) == 1
    iface = V.topology.interfaces[0]
    assert {iface.side_a, iface.side_b} == {"u0", "u1"}


def test_disjoint_patches_are_not_glued():
    a = primitives.rectangle(0, 0, 1, 1).elevate(2)
    b = primitives.rectangle(5, 5, 6, 6).elevate(2)
    V = FunctionSpace([a, b])
    assert V.n_scalar_basis == a.n_cp + b.n_cp
    assert V.topology.interfaces == ()


def test_nonconforming_interface_raises():
    """Nonconforming interfaces must raise a clear error."""
    left = primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2)
    right = primitives.rectangle(1, 0, 2, 1).elevate(2).refine(1)
    with pytest.raises(NonConformingError, match="discretize the interface differently"):
        compute_topology([left, right])


def test_mixed_degrees_rejected():
    a = primitives.rectangle(0, 0, 1, 1).elevate(2)
    b = primitives.rectangle(1, 0, 2, 1).elevate(3)
    with pytest.raises(ValueError, match="same degree|share a degree"):
        FunctionSpace([a, b])


# -- dof bookkeeping -------------------------------------------------------


def test_expand_dofs_interleaves():
    np.testing.assert_array_equal(expand_dofs(np.array([0, 3]), 1), [0, 3])
    np.testing.assert_array_equal(expand_dofs(np.array([0, 3]), 2), [0, 1, 6, 7])
    np.testing.assert_array_equal(expand_dofs(np.array([2]), 3), [6, 7, 8])
    np.testing.assert_array_equal(
        expand_dofs(np.array([[0, 1], [2, 3]]), 2), [[0, 1, 2, 3], [4, 5, 6, 7]]
    )




def test_vec_space_dof_count():
    V = FunctionSpace(primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1), vec=2)
    assert V.n_dofs == 2 * V.n_scalar_basis
    assert V.elem_dofs_vec().shape == (V.n_elems, V.n_local * 2)


def test_dofmap_lift_restrict_roundtrip():
    import jax.numpy as jnp

    dm = make_dofmap(6, prescribed=[1, 4], values=jnp.array([10.0, 20.0]))
    assert dm.n_free == 4
    u = dm.lift(jnp.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(u), [1.0, 10.0, 2.0, 3.0, 20.0, 4.0])
    np.testing.assert_allclose(np.asarray(dm.restrict(u)), [1.0, 2.0, 3.0, 4.0])


def test_dofmap_sorts_prescribed_with_values():
    import jax.numpy as jnp

    dm = make_dofmap(4, prescribed=[3, 0], values=jnp.array([30.0, 0.0]))
    np.testing.assert_array_equal(dm.prescribed, [0, 3])
    np.testing.assert_allclose(np.asarray(dm.values), [0.0, 30.0])
    np.testing.assert_allclose(np.asarray(dm.lift(jnp.array([1.0, 2.0]))), [0.0, 1.0, 2.0, 30.0])


def test_dofmap_values_are_differentiable():
    """Dirichlet data must be a traced leaf, not baked-in constants."""
    import jax
    import jax.numpy as jnp

    def total(vals):
        dm = make_dofmap(4, prescribed=[0, 3], values=vals)
        return jnp.sum(dm.lift(jnp.array([1.0, 2.0])) ** 2)

    g = jax.grad(total)(jnp.array([3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(g), [6.0, 8.0])


# -- pytree hygiene --------------------------------------------------------


def test_function_space_pytree_roundtrip():
    import jax

    V = FunctionSpace(primitives.quarter_annulus(1.0, 2.0).elevate(2).refine(1), vec=2)
    leaves, treedef = jax.tree_util.tree_flatten(V)
    W = jax.tree_util.tree_unflatten(treedef, leaves)

    assert (W.dim, W.vec, W.degree, W.n_elems) == (V.dim, V.vec, V.degree, V.n_elems)
    np.testing.assert_allclose(np.asarray(W.cpts), np.asarray(V.cpts))


def test_function_space_survives_jit():
    """Static metadata must be hashable for the space to cross a jit boundary."""
    import jax
    import jax.numpy as jnp

    V = FunctionSpace(primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))

    @jax.jit
    def total_weight(space):
        return jnp.sum(space.wgts)

    np.testing.assert_allclose(float(total_weight(V)), float(jnp.sum(V.wgts)))
    # a structurally identical but distinct object must not blow up
    V2 = FunctionSpace(primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))
    np.testing.assert_allclose(float(total_weight(V2)), float(jnp.sum(V2.wgts)))


def test_shape_derivative_flows_to_control_points():
    """cpts is a leaf, so geometry gradients need no special machinery."""
    import dataclasses

    import jax
    import jax.numpy as jnp

    V = FunctionSpace(primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))

    def f(cpts):
        return jnp.sum(dataclasses.replace(V, cpts=cpts).cpts ** 2)

    g = jax.grad(f)(V.cpts)
    np.testing.assert_allclose(np.asarray(g), 2 * np.asarray(V.cpts), rtol=1e-12)


@pytest.mark.parametrize("degree", [2, 3, 4])
@pytest.mark.parametrize("n_refine", [0, 1, 2])
def test_tensor_product_connectivity_and_boundary_dofs(degree, n_refine):
    patch = primitives.rectangle(0, 0, 2, 1).elevate(degree).refine(n_refine)
    space = FunctionSpace(patch)
    spans = 2 ** n_refine
    n = spans + degree
    assert space.n_elems == spans ** 2
    assert space.n_scalar_basis == n ** 2
    indices = np.arange(n ** 2).reshape(n, n)
    for label, expected in (("left", indices[:, 0]), ("right", indices[:, -1]),
                            ("bottom", indices[0]), ("top", indices[-1])):
        np.testing.assert_array_equal(np.sort(space.boundaries[label].dofs),
                                      np.sort(expected))
    for v in range(spans):
        for u in range(spans):
            expected = indices[v:v + degree + 1, u:u + degree + 1].ravel()
            np.testing.assert_array_equal(space.elem_dofs[v * spans + u], expected)
