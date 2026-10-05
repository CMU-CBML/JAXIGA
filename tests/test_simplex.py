"""Simplex basis, physical calculus, conformity, and manufactured PDE solves."""
import dataclasses
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxiga as jx
from jaxiga.geometry.nurbs import evaluate_patch
from jaxiga.space.bernstein import voigt_pairs
from jaxiga.space.simplex import bernstein_jnp, bernstein_simplex, gauss_rule, lattice


def unit(dim, degree=1, labels=None):
    return jx.SimplexPatch.from_vertices(np.vstack([np.zeros(dim), np.eye(dim)]), labels).elevate(degree)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('degree', [1, 2, 3, 4])
def test_basis_derivatives_and_exact_quadrature(dim, degree):
    pts, weights = gauss_rule(dim, degree + 2)
    B, dB, d2B = bernstein_simplex(pts, degree, order=2)
    np.testing.assert_allclose(B.sum(axis=1), 1, atol=2e-15)
    np.testing.assert_allclose(dB.sum(axis=-1), 0, atol=5e-15)
    np.testing.assert_allclose(d2B.sum(axis=-1), 0, atol=3e-14)
    # Integral of each total-degree Bernstein polynomial is p!/(p+d)!.
    np.testing.assert_allclose(weights @ B, factorial(degree) / factorial(degree + dim), atol=1e-15)
    point = pts[len(pts) // 2]
    derivative = jax.jacfwd(lambda x: bernstein_jnp(x, degree))
    second = jax.jacfwd(derivative)(point)
    np.testing.assert_allclose(derivative(point).T, dB[len(pts)//2], atol=3e-14)
    for k, (a, b) in enumerate(voigt_pairs(dim)):
        np.testing.assert_allclose(second[:, a, b], d2B[len(pts)//2, k], atol=3e-14)


@pytest.mark.parametrize('dim', [2, 3])
def test_geometry_elevation_rational_derivatives_and_shape_gradient(dim):
    p = unit(dim, 2)
    p = dataclasses.replace(p, weights=jnp.linspace(.85, 1.15, p.n_cp))
    space = jx.FunctionSpace(p)
    ps = jx.gauss(space)
    basis = jx.evaluate(space, ps, order=2)
    np.testing.assert_allclose(evaluate_patch(p.elevate(4), ps.ref), basis.x[0], atol=1e-14)
    np.testing.assert_allclose(np.einsum('qdj,ja->qda', basis.dR[0], p.ctrl_pts),
                               np.broadcast_to(np.eye(dim), (basis.n_q, dim, dim)), atol=1e-13)
    np.testing.assert_allclose(np.einsum('qkj,ja->qka', basis.d2R[0], p.ctrl_pts), 0, atol=1e-12)
    # Dilation scales the full geometry volume by scale**dim, even with rational weights.
    volume = lambda s: jx.evaluate(dataclasses.replace(space, cpts=space.cpts*s), ps).w.sum()
    assert jax.jit(jax.grad(volume))(1.) == pytest.approx(dim * float(volume(1.)), rel=1e-12)


@pytest.mark.parametrize('dim', [2, 3])
def test_boundary_normals_divergence_and_neumann_solve(dim):
    patch = unit(dim, 2, {f'f{i}': f'face{i}' for i in range(dim + 1)})
    space = jx.FunctionSpace(patch)
    surface_flux = np.zeros(dim)
    divergence = 0.
    for label in space.boundaries:
        b = jx.evaluate(space, jx.boundary_gauss(space, label))
        surface_flux += np.sum(np.asarray(b.w)[..., None] * b.normal, axis=(0, 1))
        divergence += float(jnp.sum(b.w * jnp.sum(b.x * b.normal, axis=-1)))
    np.testing.assert_allclose(surface_flux, 0., atol=1e-14)
    assert divergence == pytest.approx(dim / factorial(dim), abs=1e-14)
    problem = jx.Poisson(space, dirichlet=[jx.DirichletBC(0., where='face1')],
                         neumann=[jx.Neumann(lambda x, n, p: n[0:1], where=f'face{i}')
                                  for i in range(dim + 1) if i != 1])
    sol = jx.solve(problem, params={'a0': 1.})
    assert jx.errornorm(sol, lambda x: x[0], 'L2') < 2e-13
    inside = np.full((1, dim), .15)
    np.testing.assert_allclose(sol.probe(inside), .15, atol=1e-12)
    # Inside the bounding cube but outside the simplex: never extrapolate.
    assert np.isnan(sol.probe(np.full((1, dim), .9))).all()


@pytest.mark.parametrize('dim', [2, 3])
def test_poisson_bubble_exact_with_interior_unknown(dim):
    space = jx.FunctionSpace(unit(dim, dim + 1))
    exact = lambda x: (1 - x.sum()) * jnp.prod(x)
    source = lambda x, p: -jnp.trace(jax.hessian(exact)(x))
    problem = jx.Poisson(space, source=source,
                         dirichlet=[jx.DirichletBC(0., where=lambda x: np.ones(len(x), bool))])
    sol = jx.solve(problem, params={'a0': 1.})
    assert jx.errornorm(sol, exact, 'L2') < 1e-13
    b = jx.evaluate(space, jx.gauss(space), order=2)
    local = np.asarray(sol.u)[np.asarray(space.elem_dofs)]
    hess = np.einsum('eqki,ei->eqk', b.d2R, local)[0]
    expected = jax.vmap(jax.hessian(exact))(b.x[0])
    for k, (a, c) in enumerate(voigt_pairs(dim)):
        np.testing.assert_allclose(hess[:, k], expected[:, a, c], atol=2e-12)


@pytest.mark.parametrize('dim', [2, 3])
def test_conforming_connectivity_and_linear_patch_test(dim):
    a = unit(dim, 4)
    vertices = np.vstack([np.zeros(dim), np.eye(dim)])
    vertices[-1, -1] = -1
    vertices[[0, 1]] = vertices[[1, 0]]  # positive orientation, reversed common trace
    b = jx.SimplexPatch.from_vertices(vertices).elevate(4)
    space = jx.FunctionSpace([a, b])
    assert len(space.topology.interfaces) == 1
    shared = len(lattice(dim - 1, 4))
    assert space.n_scalar_basis == 2*a.n_cp - shared
    exact = lambda x: 1 + jnp.sum(x * jnp.arange(1, dim + 1))
    problem = jx.Poisson(space, dirichlet=[jx.DirichletBC(lambda x,p: exact(x),
                         where=lambda x: np.ones(len(x), bool))])
    sol = jx.solve(problem, params={'a0': 1.})
    assert jx.errornorm(sol, exact, 'L2') < 5e-13
    # A matching corner set does not excuse a torn high-order trace.
    i = np.flatnonzero(np.all(b.multi_indices[:, :2] > 0, axis=1) & (b.multi_indices[:, -1] == 0))[0]
    bad = dataclasses.replace(b, ctrl_pts=b.ctrl_pts.at[i, 0].add(.01))
    with pytest.raises(jx.NonConformingError, match='trace'):
        jx.FunctionSpace([a, bad])


def test_mixed_families_and_unsupported_operations_fail_explicitly():
    p = unit(2)
    with pytest.raises(ValueError, match='overlap'):
        jx.FunctionSpace([p, jx.primitives.quadrilateral([[0,0],[1,0],[1,1],[0,1]])])
    with pytest.raises(NotImplementedError, match='knot'):
        p.insert_knots([.5])
    with pytest.raises(NotImplementedError, match='Galerkin'):
        jx.greville(jx.FunctionSpace(p))


@pytest.mark.parametrize('dim', [2, 3])
def test_vtk_sampling_preserves_domain_volume(dim):
    from jaxiga.post.vtk import _grid
    space = jx.FunctionSpace(unit(dim, 2))
    for n in (1, 2, 4):
        basis, coords, cells, kind = _grid(space, n)
        xyz = np.column_stack(coords)[:, :dim]
        volumes = np.linalg.det(xyz[cells[:, 1:]] - xyz[cells[:, :1]]) / factorial(dim)
        assert np.all(volumes > 0)
        assert volumes.sum() == pytest.approx(1 / factorial(dim), abs=1e-13)
        assert kind == (5 if dim == 2 else 10)
        assert cells.max() < basis.n_q


def test_vector_elasticity_and_differentiable_solve():
    space = jx.FunctionSpace(unit(2, 3), vec=2)
    affine = lambda x: jnp.array([.1*x[0]+.03*x[1], -.02*x[0]+.04*x[1]])
    problem = jx.LinearElasticity(space, plane='stress', dirichlet=[
        jx.DirichletBC(lambda x,p: affine(x), where=lambda x: np.ones(len(x), bool))])
    sol = jx.solve(problem, params={'E': 100., 'nu': .3})
    basis = jx.evaluate(space, jx.gauss(space))
    np.testing.assert_allclose(sol.at(basis=basis), jax.vmap(jax.vmap(affine))(basis.x), atol=2e-13)
    assert np.isfinite(sol.field('von_mises', basis=basis)).all()
    scalar = jx.FunctionSpace(unit(2, 3))
    poisson = jx.Poisson(scalar, source=lambda x,p: 1., dirichlet=[
        jx.DirichletBC(0., where=lambda x: np.ones(len(x), bool))])
    def objective(a):
        return jnp.sum(jx.solve(poisson, params={'a0': a}).u)
    value, gradient = jax.value_and_grad(objective)(2.)
    assert gradient == pytest.approx(-float(value)/2, rel=1e-11)


@pytest.mark.parametrize('dim', [2, 3])
def test_curved_rational_probe_and_jit(dim):
    patch = unit(dim, 2)
    control = patch.ctrl_pts.at[1, 1].add(.05)
    patch = dataclasses.replace(patch, ctrl_pts=control, weights=jnp.linspace(.9, 1.1, patch.n_cp))
    space = jx.FunctionSpace(patch)
    solution = jx.Solution(space, space.cpts[:, 0])
    params = np.random.default_rng(5).dirichlet(np.ones(dim+1), 8)[:, 1:]
    points = evaluate_patch(patch, params)
    targets = jx.locate_points(space, points)
    result = jax.jit(lambda x: solution.probe(x, targets=targets))(points)
    np.testing.assert_allclose(result[:, 0], points[:, 0], atol=2e-12)


@pytest.mark.parametrize('dim', [2, 3])
def test_one_point_rule_integrates_constant_and_centroid(dim):
    points, weights = gauss_rule(dim, 1)
    np.testing.assert_allclose(points, 1 / (dim + 1), atol=1e-15)
    assert weights.sum() == pytest.approx(1 / factorial(dim), abs=1e-15)


def test_quadratic_triangles_converge_on_poisson_square():
    exact = lambda x: x[0]*(1-x[0])*x[1]*(1-x[1])
    errors = []
    for n in (2, 4, 8):
        patches = []
        for i in range(n):
            for j in range(n):
                a,b,c,d = np.array([[i,j],[i+1,j],[i+1,j+1],[i,j+1]], dtype=float)/n
                patches.extend([jx.primitives.triangle([a,b,c]).elevate(2),
                                jx.primitives.triangle([a,c,d]).elevate(2)])
        space = jx.FunctionSpace(patches)
        problem = jx.Poisson(space,
            source=lambda x,p: 2*(x[0]*(1-x[0])+x[1]*(1-x[1])),
            dirichlet=[jx.DirichletBC(0., where=lambda x: np.ones(len(x), bool))])
        sol = jx.solve(problem, params={'a0': 1.})
        errors.append(float(jx.errornorm(sol, exact, 'L2')))
    assert errors[0] > errors[1] > errors[2]
    assert np.log2(errors[-2]/errors[-1]) > 2.8
