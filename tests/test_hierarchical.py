"""Local refinement with THB-splines.

The invariants worth testing are the ones that make a hierarchical space a
drop-in replacement for a tensor-product one: it reduces to the old behaviour
exactly when nothing is refined, it reproduces the geometry, its basis is a
partition of unity, and everything downstream keeps working on it.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxiga as jx
from jaxiga.geometry.nurbs import evaluate_patch
from jaxiga.space import pointset as P
from jaxiga.space.bernstein import bernstein_tensor
from jaxiga.space.evaluation import evaluate
from jaxiga.space.hierarchical import HierarchyError

ALL = lambda x: jnp.full(x.shape[0], True)


def square(deg=2, refine=2):
    return jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine)


def annulus(deg=3, refine=2):
    return jx.primitives.quarter_annulus(1.0, 2.0).elevate(deg).refine(refine)


def spline_partition_error(space, n=5):
    """``max |sum_i N_i - 1|`` over the *B-spline* functions.

    Deliberately not ``sum_i R_i``: the rational basis is normalized by its own
    denominator, so ``sum R_i = 1`` holds for any extraction whatsoever and
    cannot distinguish THB from HB. Truncation is a statement about the
    pre-rational functions, and that is what this measures.
    """
    ps = P.grid(space, n)
    B, _, _ = bernstein_tensor(np.asarray(ps.ref), space.degree, order=0)
    N = np.einsum("eij,qj->eqi", np.asarray(space.extraction), B)
    return float(np.abs(N.sum(-1) - 1.0).max())


def geometry_error(space, patch, n=4):
    """How far the hierarchical geometry map is from the level-0 patch."""
    ps = P.grid(space, n)
    x = np.asarray(evaluate(space, ps).x).reshape(-1, patch.dim_phys)
    ref = np.asarray(ps.ref)
    boxes = space.elem_vertex
    dim = space.dim
    params = np.concatenate(
        [
            boxes[e, :dim] + (ref + 1.0) / 2.0 * (boxes[e, dim:] - boxes[e, :dim])
            for e in range(space.n_elems)
        ],
        axis=0,
    )
    return float(np.abs(x - evaluate_patch(patch, params)).max())


# -- 1. no-op regression ---------------------------------------------------


@pytest.mark.parametrize("deg,ref", [(2, 2), (3, 1)])
def test_empty_refinement_reproduces_the_tensor_product_space(deg, ref):
    patch = square(deg, ref)
    V = jx.FunctionSpace(patch)
    W = jx.refine_elements(V, [])

    assert W.n_elems == V.n_elems
    assert W.n_scalar_basis == V.n_scalar_basis
    assert W.n_local == W.n_bernstein  # no padding at all
    np.testing.assert_array_equal(np.asarray(W.elem_dofs), np.asarray(V.elem_dofs))
    np.testing.assert_allclose(
        np.asarray(W.extraction), np.asarray(V.extraction), atol=1e-14
    )
    np.testing.assert_allclose(np.asarray(W.cpts), np.asarray(V.cpts), atol=1e-14)
    np.testing.assert_allclose(np.asarray(W.wgts), np.asarray(V.wgts), atol=1e-14)
    np.testing.assert_allclose(W.elem_vertex, V.elem_vertex, atol=1e-14)
    for label, bset in V.boundaries.items():
        np.testing.assert_array_equal(np.sort(W.boundaries[label].dofs), np.sort(bset.dofs))


def test_tensor_product_space_is_not_hierarchical():
    V = jx.FunctionSpace(square())
    assert not V.is_hierarchical
    assert jx.refine_elements(V, []).is_hierarchical


# -- 2. partition of unity, and the contrast with untruncated HB ------------


@pytest.mark.parametrize("patch", [square(2, 2), square(3, 2), annulus(3, 2)])
def test_thb_is_a_partition_of_unity_and_hb_is_not(patch):
    V = jx.FunctionSpace(patch)
    marked = [0, 1, 2, 7]
    thb = jx.refine_elements(V, marked, truncate=True)
    hb = jx.refine_elements(V, marked, truncate=False)

    assert spline_partition_error(thb) < 1e-12
    # The contrast is the point of truncation, so assert HB really fails.
    assert spline_partition_error(hb) > 1e-3


# -- 3. geometry reproduction ----------------------------------------------


@pytest.mark.parametrize("patch", [square(2, 2), annulus(3, 2)])
def test_hierarchical_geometry_matches_the_level_zero_patch(patch):
    V = jx.FunctionSpace(patch)
    W = jx.refine_elements(V, [0, 1, 2, 7])
    assert geometry_error(W, patch) < 1e-12
    # ... and stays exact after several cycles
    for _ in range(3):
        W = jx.refine_elements(W, np.arange(min(4, W.n_elems)))
    assert geometry_error(W, patch) < 1e-12


def test_untruncated_hb_does_not_reproduce_the_geometry():
    patch = square(2, 2)
    V = jx.FunctionSpace(patch)
    hb = jx.refine_elements(V, [0, 1, 2, 7], truncate=False)
    assert geometry_error(hb, patch) > 1e-3


# -- 4. hand-counted dimensions --------------------------------------------


def test_one_dimensional_two_level_dimension_count():
    # p=2, four level-0 elements -> six functions. Refining the first element
    # activates level-1 functions and deactivates the level-0 ones whose
    # support lies entirely in the refined region.
    patch = jx.primitives.interval(0.0, 1.0).elevate(2).refine(2)
    V = jx.FunctionSpace(patch)
    assert V.n_scalar_basis == 6 and V.n_elems == 4

    W = jx.refine_elements(V, [0], admissible=False)
    assert W.n_elems == 5  # one element replaced by two
    # every function is still supported on some leaf, so no dof is lost
    assert W.n_scalar_basis > V.n_scalar_basis
    assert spline_partition_error(W) < 1e-12


def test_two_dimensional_corner_refinement_counts():
    V = jx.FunctionSpace(square(2, 2))  # 4x4 elements, 6x6 = 36 functions
    assert (V.n_elems, V.n_scalar_basis) == (16, 36)

    W = jx.refine_elements(V, [0, 1, 4, 5])  # the 2x2 corner block
    # four elements become sixteen
    assert W.n_elems == 16 - 4 + 16
    assert spline_partition_error(W) < 1e-12
    assert W.n_local > W.n_bernstein  # padding is genuinely in play


# -- admissibility ---------------------------------------------------------


def test_admissibility_bounds_the_local_function_count():
    """Class-2 admissibility is what keeps ``n_local_max`` near ``(p+1)^dim``."""
    V = jx.FunctionSpace(square(2, 1))
    W = V
    for _ in range(6):
        centre = (W.elem_vertex[:, :2] + W.elem_vertex[:, 2:]) / 2
        W = jx.refine_elements(W, np.argsort(np.linalg.norm(centre, axis=1))[:3])
    assert W.hierarchy.n_levels >= 6
    # two successive levels of functions at most, so at most twice the count
    assert W.n_local <= 2 * W.n_bernstein
    assert spline_partition_error(W) < 1e-11


def test_refusing_a_mesh_with_too_many_functions_per_element():
    V = jx.FunctionSpace(square(2, 1))
    W = V
    with pytest.raises(HierarchyError, match="gathers"):
        for _ in range(8):
            centre = (W.elem_vertex[:, :2] + W.elem_vertex[:, 2:]) / 2
            W = jx.refine_elements(
                W, np.argsort(np.linalg.norm(centre, axis=1))[:2], admissible=False
            )


def test_marking_out_of_range_raises():
    V = jx.FunctionSpace(square())
    with pytest.raises(IndexError):
        jx.refine_elements(V, [V.n_elems])


# -- solving on an adapted space -------------------------------------------


EXACT = lambda x: jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
SOURCE = lambda x, p: 2 * jnp.pi**2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])


def poisson_on(space):
    return jx.Poisson(space, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=SOURCE)


def test_galerkin_and_energy_agree_on_a_hierarchical_space():
    """The padded elements must be inert for *every* method, not just assembly."""
    V = jx.refine_elements(jx.FunctionSpace(square(2, 2)), [0, 1, 4, 5])
    prob = poisson_on(V)
    a = jx.solve(prob, params={"a0": 1.0})
    b = jx.solve(prob, params={"a0": 1.0}, method="energy", tol=1e-12, max_steps=4000)
    rel = float(jnp.abs(a.u - b.u).max() / jnp.abs(a.u).max())
    assert rel < 1e-6


def test_refinement_reduces_the_error():
    V = jx.FunctionSpace(square(2, 2))
    err = []
    for _ in range(3):
        sol = jx.solve(poisson_on(V), params={"a0": 1.0})
        err.append(jx.errornorm(sol, EXACT, "L2"))
        V = jx.refine_elements(V, jx.dorfler_mark(jx.residual_indicator(sol), 0.5))
    assert err[-1] < err[0]
    assert all(b <= a for a, b in zip(err, err[1:]))


def test_gradients_flow_on_an_adapted_space():
    """Section 2.10's contract holds on adapted spaces too."""
    V = jx.refine_elements(jx.FunctionSpace(square(2, 1)), [0, 1])

    def loss(a0):
        return jnp.sum(jx.solve(poisson_on(V), params={"a0": a0}).u ** 2)

    g = float(jax.grad(loss)(1.3))
    h = 1e-6
    fd = float((loss(1.3 + h) - loss(1.3 - h)) / (2 * h))
    assert abs(g - fd) / abs(fd) < 1e-6


def test_shape_derivative_on_an_adapted_space():
    V = jx.refine_elements(jx.FunctionSpace(square(2, 1)), [0, 1])
    prob = poisson_on(V)

    def loss(cpts):
        import dataclasses

        space = dataclasses.replace(V, cpts=cpts)
        return jnp.sum(jx.solve(prob.with_space(space), params={"a0": 1.0}).u ** 2)

    g = jax.grad(loss)(V.cpts)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.abs(g).max()) > 0


# -- multipatch ------------------------------------------------------------


def lshape(deg=2, refine=1):
    return [
        p.elevate(deg).refine(refine)
        for p in (
            jx.primitives.rectangle(0, 0, 1, 1),
            jx.primitives.rectangle(-1, 0, 0, 1),
            jx.primitives.rectangle(-1, -1, 0, 0),
        )
    ]


def test_multipatch_refinement_stays_conforming():
    """The closure must mirror marking across interfaces, or the zip breaks."""
    V = jx.FunctionSpace(lshape())
    W = V
    for _ in range(4):
        centre = (W.elem_vertex[:, :2] + W.elem_vertex[:, 2:]) / 2
        # mark near the re-entrant corner, which three patches share
        W = jx.refine_elements(W, np.argsort(np.linalg.norm(centre, axis=1))[:4])
    assert spline_partition_error(W) < 1e-11
    assert W.n_local <= 2 * W.n_bernstein
    # a conforming space solves; a torn one would not reproduce a linear field
    sol = jx.solve(
        jx.Poisson(W, dirichlet=[jx.DirichletBC(lambda x, p: x[0] + 2 * x[1], where=ALL)]),
        params={"a0": 1.0},
    )
    assert jx.errornorm(sol, lambda x: x[0] + 2 * x[1], "L2") < 1e-10


# -- marking ---------------------------------------------------------------


def test_dorfler_marks_the_bulk():
    eta = np.array([3.0, 1.0, 1.0, 1.0])
    # 9 of the total 12 sits in the first element
    assert list(jx.dorfler_mark(eta, 0.5)) == [0]
    assert list(jx.dorfler_mark(eta, 1.0)) == [0, 1, 2, 3]


def test_dorfler_rejects_a_bad_fraction():
    with pytest.raises(ValueError):
        jx.dorfler_mark(np.ones(3), 0.0)


def test_indicator_is_large_where_the_solution_is_rough():
    """A source concentrated in one corner should be marked there."""
    V = jx.FunctionSpace(square(2, 3))
    bump = lambda x, p: jnp.exp(-200.0 * ((x[0] - 0.2) ** 2 + (x[1] - 0.2) ** 2))
    sol = jx.solve(
        jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=bump),
        params={"a0": 1.0},
    )
    eta = jx.residual_indicator(sol)
    centre = (V.elem_vertex[:, :2] + V.elem_vertex[:, 2:]) / 2
    worst = centre[int(np.argmax(eta))]
    assert np.linalg.norm(worst - np.array([0.2, 0.2])) < 0.2


def test_collocation_is_refused_on_a_hierarchical_space():
    """Section 23.7: Greville systems for THB are open research."""
    V = jx.refine_elements(jx.FunctionSpace(square(3, 2)), [0, 1])
    problem = jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=SOURCE)
    with pytest.raises(NotImplementedError, match="hierarchical"):
        jx.solve(problem, method="collocation", params={"a0": 1.0})
    with pytest.raises(NotImplementedError, match="tensor-product"):
        P.greville(V)


# -- field transfer between nested meshes ----------------------------------


def _nested_pair(deg=2, base=6, seed=0):
    """A coarse hierarchical space and a ragged two-step refinement of it."""
    kn = np.linspace(0.0, 1.0, base + 1)[1:-1]
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).insert_knots(kn, kn)
    coarse = jx.refine_elements(jx.FunctionSpace(patch, vec=1), [])
    rng = np.random.default_rng(seed)
    mid = jx.refine_elements(coarse, rng.choice(coarse.n_elems, 12, replace=False))
    fine = jx.refine_elements(mid, rng.choice(mid.n_elems, 15, replace=False))
    return coarse, fine


def test_parent_map_finds_the_containing_element():
    """Every fine element must sit inside the coarse element it names."""
    from jaxiga.space.transfer import parent_map

    coarse, fine = _nested_pair()
    pmap = parent_map(coarse, fine)
    assert len(pmap.parent) == fine.n_elems

    dim = coarse.dim
    box_f, box_c = np.asarray(fine.elem_vertex), np.asarray(coarse.elem_vertex)
    parent = box_c[pmap.parent]
    assert np.all(box_f[:, :dim] >= parent[:, :dim] - 1e-12)
    assert np.all(box_f[:, dim:] <= parent[:, dim:] + 1e-12)
    # the affine reference map must send the fine cube into the coarse one
    corners = np.array([[-1.0, -1.0], [1.0, 1.0]])
    mapped = pmap.to_coarse(corners)
    assert np.all(np.abs(mapped) <= 1.0 + 1e-12)


def test_parent_map_rejects_unrelated_meshes():
    from jaxiga.space.transfer import parent_map

    coarse, _ = _nested_pair(base=6)
    other, _ = _nested_pair(base=5)
    with pytest.raises(ValueError, match="not a refinement"):
        parent_map(coarse, other)
    # nor is the coarse mesh a refinement of its own refinement
    _, fine = _nested_pair(base=6)
    with pytest.raises(ValueError):
        parent_map(fine, coarse)


def test_projection_onto_a_nested_mesh_is_exact():
    """Nested spaces: transferring a coarse field to the fine mesh loses nothing."""
    from jaxiga.post.solution import Solution
    from jaxiga.space.transfer import coarse_basis_at, parent_map, project

    deg = 2
    coarse, fine = _nested_pair(deg=deg)
    b_c = evaluate(coarse, P.gauss(coarse))
    f = lambda x: 0.3 + x[..., 0] ** deg - 0.7 * x[..., 1] ** deg + 0.4 * x[..., 0] * x[..., 1]

    u_c = project(f(b_c.x), coarse, b_c)
    sol_c = Solution(coarse, u_c)
    assert np.abs(np.asarray(sol_c.at(basis=b_c))[..., 0] - np.asarray(f(b_c.x))).max() < 1e-12

    ps_f = P.gauss(fine)
    b_f = evaluate(fine, ps_f)
    pmap = parent_map(coarse, fine)
    at_fine = sol_c.at(basis=coarse_basis_at(coarse, pmap, ps_f))[..., 0]
    sol_f = Solution(fine, project(at_fine, fine, b_f))
    err = np.abs(np.asarray(sol_f.at(basis=b_f))[..., 0] - np.asarray(f(b_f.x))).max()
    assert err < 1e-12


def test_projection_handles_vector_fields():
    from jaxiga.post.solution import Solution
    from jaxiga.space.transfer import project

    kn = np.linspace(0.0, 1.0, 5)[1:-1]
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(2).insert_knots(kn, kn)
    V = jx.refine_elements(jx.FunctionSpace(patch, vec=2), [])
    b = evaluate(V, P.gauss(V))
    exact = jnp.stack([b.x[..., 0] ** 2, 1.0 - b.x[..., 1] ** 2], axis=-1)
    sol = Solution(V, project(exact, V, b))
    assert np.abs(np.asarray(sol.at(basis=b)) - np.asarray(exact)).max() < 1e-12


def test_quadrature_carry_is_exact_on_untouched_elements():
    """Point data must be copied verbatim wherever the mesh did not change."""
    from jaxiga.space.transfer import carry_points, parent_map

    coarse, fine = _nested_pair()
    ps_c, ps_f = P.gauss(coarse), P.gauss(fine)
    rng = np.random.default_rng(3)
    values = rng.normal(size=(coarse.n_elems, ps_c.n_q))

    pmap = parent_map(coarse, fine)
    carried = np.asarray(carry_points(values, pmap, ps_f.ref, ps_c.ref))
    same = pmap.scale[:, 0] == 1.0
    assert same.sum() > 0
    assert np.array_equal(carried[same], values[pmap.parent[same]])
    # and everywhere it must reuse a value that exists on the parent
    assert np.isin(carried, values).all()


def test_transfer_works_in_three_dimensions():
    """Nothing in the transfer machinery may assume two dimensions.

    The phase-field solver built on it is 2D only because the spectral split
    is; this keeps the parts underneath honest about that.
    """
    from jaxiga.post.solution import Solution
    from jaxiga.space.transfer import (
        carry_points,
        coarse_basis_at,
        parent_map,
        project,
    )

    patch = jx.primitives.cuboid((0, 0, 0), (1, 1, 1)).elevate(2).refine(1)
    coarse = jx.refine_elements(jx.FunctionSpace(patch, vec=1), [])
    fine = jx.refine_elements(coarse, [0, 3, 5])

    b_c = evaluate(coarse, P.gauss(coarse))
    f = lambda x: 1.0 + x[..., 0] ** 2 - 0.5 * x[..., 1] ** 2 + 0.25 * x[..., 2] ** 2
    sol_c = Solution(coarse, project(f(b_c.x), coarse, b_c))

    ps_f = P.gauss(fine)
    b_f = evaluate(fine, ps_f)
    pmap = parent_map(coarse, fine)
    assert pmap.scale.shape[1] == 3

    at_fine = sol_c.at(basis=coarse_basis_at(coarse, pmap, ps_f))[..., 0]
    sol_f = Solution(fine, project(at_fine, fine, b_f))
    assert np.abs(np.asarray(sol_f.at(basis=b_f))[..., 0] - np.asarray(f(b_f.x))).max() < 1e-11

    rng = np.random.default_rng(1)
    values = rng.normal(size=(coarse.n_elems, P.gauss(coarse).n_q))
    carried = np.asarray(carry_points(values, pmap, ps_f.ref, P.gauss(coarse).ref))
    same = pmap.scale[:, 0] == 1.0
    assert np.array_equal(carried[same], values[pmap.parent[same]])


# -- construction fast paths ----------------------------------------------


def test_gather_rows_matches_sparse_slicing():
    """The hand-rolled CSR row gather must equal what slicing the matrix gives."""
    import scipy.sparse as sp

    from jaxiga.space.hierarchical import _gather_rows

    rng = np.random.default_rng(5)
    for _ in range(50):
        n_rows, n_cols = int(rng.integers(4, 40)), int(rng.integers(4, 40))
        dense = rng.normal(size=(n_rows, n_cols))
        dense[rng.random((n_rows, n_cols)) < 0.7] = 0.0
        A = sp.csr_matrix(dense)
        rows = rng.choice(n_rows, size=int(rng.integers(1, n_rows + 1)), replace=False)

        block = A[rows, :].tocsc()
        block.eliminate_zeros()
        want_cols = np.unique(block.nonzero()[1])
        want = np.asarray(block[:, want_cols].todense())

        cols, V = _gather_rows(A.indptr, A.indices, A.data, np.asarray(rows))
        assert np.array_equal(cols, want_cols)
        assert V.shape == want.shape
        np.testing.assert_array_equal(V, want)


def test_gather_rows_handles_empty_rows():
    import scipy.sparse as sp

    from jaxiga.space.hierarchical import _gather_rows

    A = sp.csr_matrix(np.zeros((5, 6)))
    cols, V = _gather_rows(A.indptr, A.indices, A.data, np.array([0, 2, 4]))
    assert cols.size == 0 and V.shape == (3, 0)


def test_extraction_ids_label_distinct_operators():
    """Identical 1D operators must share a label, different ones must not."""
    from jaxiga.space.hierarchical import _extraction_ids

    a, b = np.eye(3), np.tril(np.ones((3, 3)))
    ids = _extraction_ids([a, b, a.copy(), b.copy(), a])
    assert ids[0] == ids[2] == ids[4]
    assert ids[1] == ids[3]
    assert ids[0] != ids[1]

    # a real open knot vector reuses one interior operator for most elements
    from jaxiga.space.bernstein import bezier_extraction

    knots = np.concatenate([[0, 0, 0], np.linspace(0, 1, 12), [1, 1, 1]])
    ops = bezier_extraction(knots, 2)[0]
    ids = _extraction_ids(ops)
    assert len(np.unique(ids)) < len(ops), "no operator was reused"


def test_tensor_indices_matches_a_meshgrid_reference():
    from jaxiga.space.hierarchical import _ravel, _tensor_indices

    rng = np.random.default_rng(6)
    for _ in range(200):
        dim = int(rng.integers(1, 4))
        dims = tuple(int(rng.integers(3, 12)) for _ in range(dim))
        ranges = [
            np.sort(rng.choice(d, size=int(rng.integers(1, d + 1)), replace=False))
            for d in dims
        ]
        mesh = np.meshgrid(*ranges, indexing="ij")
        want = _ravel(tuple(m.ravel(order="F") for m in mesh), dims)
        np.testing.assert_array_equal(_tensor_indices(ranges, dims), want)


def test_support_extension_union_matches_the_per_element_form():
    """The batched support extension must be exactly the union of the single ones."""
    from jaxiga.space.hierarchical import (
        _Levels,
        _support_extension,
        _support_extension_union,
    )

    rng = np.random.default_rng(9)
    for patch, deg in [
        (jx.primitives.rectangle(0, 0, 1, 1), 2),
        (jx.primitives.rectangle(0, 0, 1, 1), 3),
        (jx.primitives.cuboid((0, 0, 0), (1, 0.2, 1)), 2),
    ]:
        levels = _Levels(patch.elevate(deg).refine(2), 3)
        for l in (1, 2):
            n = int(np.prod(levels.n_elem[l]))
            for size in (1, 5, min(n, 40)):
                elems = rng.choice(n, size=size, replace=False)
                want = np.unique(
                    np.concatenate([_support_extension(levels, l, e) for e in elems])
                )
                got = _support_extension_union(levels, l, elems)
                np.testing.assert_array_equal(got, want)

    # and the empty case
    levels = _Levels(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2), 2)
    assert _support_extension_union(levels, 1, []).size == 0


def test_closure_is_unchanged_by_the_batched_sweep():
    """Admissibility must still hold, and hold for the same reason.

    Every live element's whole level-(l-1) support extension has to be inside
    the level-(l-1) domain -- that is the class-2 condition the batched sweep
    replaced a per-element loop to enforce.
    """
    from jaxiga.space.hierarchical import _State, _support_extension_union

    kn = np.linspace(0.0, 1.0, 7)[1:-1]
    patch = jx.primitives.cuboid((0, 0, 0), (1, 0.2, 1)).elevate(2).insert_knots(
        kn, np.array([0.5]), kn
    )
    V = jx.refine_elements(jx.FunctionSpace(patch, vec=1), [])
    rng = np.random.default_rng(4)
    for n in (20, 40, 80):
        V = jx.refine_elements(V, rng.choice(V.n_elems, min(n, V.n_elems), replace=False))

    state = _State(list(V.patches), V.hierarchy.refined)
    dom, _ = state.domains()
    for i, levels in enumerate(state.levels):
        for l in range(1, len(dom[i])):
            if not dom[i][l]:
                continue
            required = _support_extension_union(
                levels, l, np.fromiter(dom[i][l], dtype=np.int64, count=len(dom[i][l]))
            )
            missing = [int(c) for c in required if int(c) not in dom[i][l - 1]]
            assert not missing, f"level {l} of patch {i} is not admissible: {missing[:5]}"
