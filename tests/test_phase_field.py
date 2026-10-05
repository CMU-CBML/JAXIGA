"""Phase-field fracture and the per-quadrature-point field mechanism."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxiga as jx
from jaxiga.forms import materials
from jaxiga.forms.library import (
    edge_crack_distance,
    initial_history,
    principal_strains,
    spectral_split,
    tensile_energy,
)
from jaxiga.methods._common import PointField, reactions
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

E, NU = 210e3, 0.3


def spaces(deg=2, refine=2):
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine)
    return jx.FunctionSpace(patch, vec=2), jx.FunctionSpace(patch, vec=1)


# -- PointField ------------------------------------------------------------


def test_point_field_matches_an_equivalent_closure():
    """A field given per quadrature point must equal the same field as a closure."""
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2))
    basis = evaluate(V, P.gauss(V))
    allb = lambda x: jnp.full(x.shape[0], True)  # noqa: E731

    class ByParam(jx.Poisson):
        def energy(self, g, u, x, p):
            return 0.5 * p["a0"] * jnp.sum(g**2)

    class ByClosure(jx.Poisson):
        def energy(self, g, u, x, p):
            return 0.5 * (1.0 + x[0] ** 2) * jnp.sum(g**2)

    kw = dict(dirichlet=[jx.DirichletBC(0.0, where=allb)], source=lambda x, p: 1.0)
    a0 = 1.0 + jnp.asarray(basis.x)[..., 0] ** 2

    s1 = jx.solve(ByParam(V, **kw), params={"a0": PointField(a0)})
    s2 = jx.solve(ByClosure(V, **kw), params={})
    np.testing.assert_allclose(np.asarray(s1.u), np.asarray(s2.u), atol=1e-14)


def test_point_field_is_differentiable():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))
    basis = evaluate(V, P.gauss(V))
    allb = lambda x: jnp.full(x.shape[0], True)  # noqa: E731

    class ByParam(jx.Poisson):
        def energy(self, g, u, x, p):
            return 0.5 * p["a0"] * jnp.sum(g**2)

    problem = ByParam(V, dirichlet=[jx.DirichletBC(0.0, where=allb)],
                      source=lambda x, p: 1.0)
    ones = jnp.ones((basis.n_elems, basis.n_q))

    def loss(scale):
        sol = jx.solve(problem, params={"a0": PointField(scale * ones)})
        return jnp.sum(sol.u**2)

    g = float(jax.grad(loss)(1.0))
    h = 1e-6
    fd = (loss(1.0 + h) - loss(1.0 - h)) / (2 * h)
    np.testing.assert_allclose(g, float(fd), rtol=1e-5)


# -- reactions -------------------------------------------------------------


def test_reaction_matches_the_analytical_bar():
    """A clamped bar in tension: F = E A u / L exactly."""
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2), vec=2)
    u_top = 1e-3
    problem = jx.LinearElasticity(
        V, plane="stress",
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                   jx.DirichletBC([0.0, u_top], where="top")],
    )
    sol = jx.solve(problem, params={"E": E, "nu": 0.0})
    R = reactions(sol, params={"E": E, "nu": 0.0})
    top = np.asarray(V.boundaries["top"].dofs)
    np.testing.assert_allclose(float(jnp.sum(R[2 * top + 1])), E * u_top, rtol=1e-10)


def test_reactions_vanish_on_free_dofs():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1), vec=2)
    problem = jx.LinearElasticity(
        V, dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                      jx.DirichletBC([0.0, 1e-3], where="top")],
    )
    sol = jx.solve(problem, params={"E": E, "nu": NU})
    R = np.asarray(reactions(sol, params={"E": E, "nu": NU}))
    from jaxiga.methods._common import build_context

    free = np.asarray(build_context(problem, {"E": E, "nu": NU}).dofmap.free)
    np.testing.assert_allclose(R[free], 0.0, atol=1e-30)


# -- spectral split --------------------------------------------------------


def test_split_sums_to_the_elastic_energy():
    lam, mu = materials.lame(E, NU)
    C = materials.plane_strain(E, NU)
    rng = np.random.default_rng(0)

    for _ in range(6):
        g = jnp.asarray(rng.normal(scale=1e-3, size=(2, 2)))
        eps = materials.strain_voigt(g)
        pp, pm = spectral_split(eps, lam, mu)
        np.testing.assert_allclose(float(pp + pm), float(0.5 * eps @ C @ eps), rtol=1e-6)


def test_split_sums_to_the_elastic_energy_in_3d():
    lam, mu = materials.lame(E, NU)
    C = materials.constitutive(E, NU, 3)
    rng = np.random.default_rng(11)

    for _ in range(8):
        g = rng.normal(scale=1e-3, size=(3, 3))
        eps = materials.strain_voigt(jnp.asarray(g))
        pp, pm = spectral_split(eps, lam, mu)
        np.testing.assert_allclose(
            float(pp + pm), float(0.5 * eps @ C @ eps), rtol=1e-10
        )


def test_principal_strains_match_a_dense_eigensolver():
    """The closed-form 3x3 eigenvalues must be the eigenvalues."""
    rng = np.random.default_rng(12)
    worst = 0.0
    for _ in range(200):
        A = rng.normal(scale=1e-3, size=(3, 3))
        A = 0.5 * (A + A.T)
        voigt = jnp.array(
            [A[0, 0], A[1, 1], A[2, 2], 2 * A[1, 2], 2 * A[0, 2], 2 * A[0, 1]]
        )
        mine = np.sort(np.asarray(principal_strains(voigt)))
        reference = np.sort(np.linalg.eigvalsh(A))
        worst = max(worst, np.abs(mine - reference).max() / np.abs(reference).max())
    assert worst < 1e-10, worst

    # and the 2D branch is the 2x2 closed form
    B = np.array([[2e-3, 4e-4], [4e-4, -1e-3]])
    mine = np.sort(np.asarray(principal_strains(jnp.array([B[0, 0], B[1, 1], 2 * B[0, 1]]))))
    np.testing.assert_allclose(mine, np.sort(np.linalg.eigvalsh(B)), atol=1e-12)


@pytest.mark.parametrize(
    "name,voigt",
    [
        ("zero strain", [0.0] * 6),
        ("hydrostatic tension", [1e-3, 1e-3, 1e-3, 0, 0, 0]),
        ("hydrostatic compression", [-1e-3, -1e-3, -1e-3, 0, 0, 0]),
        ("uniaxial, two equal", [1e-3, -3e-4, -3e-4, 0, 0, 0]),
        ("pure shear", [0, 0, 0, 0, 0, 2e-3]),
        ("two equal plus shear", [1e-3, 5e-4, 5e-4, 4e-4, 0, 0]),
    ],
)
def test_split_is_finite_at_degenerate_strain_states(name, voigt):
    """Repeated principal strains are ordinary states, not edge cases.

    Uniaxial strain has two equal principal values and zero strain has three,
    and a simulation starts from the latter. The closed-form eigenvalues go
    through an ``arccos`` that is singular exactly there, so this is the test
    that the regularisation actually holds.
    """
    lam, mu = materials.lame(E, NU)
    v = jnp.asarray(voigt, dtype=float)

    e = np.asarray(principal_strains(v))
    assert np.isfinite(e).all()

    pp, pm = spectral_split(v, lam, mu)
    assert np.isfinite([float(pp), float(pm)]).all()

    C = materials.constitutive(E, NU, 3)
    np.testing.assert_allclose(float(pp + pm), float(0.5 * v @ C @ v), atol=1e-12)

    for k in (0, 1):
        g = np.asarray(jax.grad(lambda w: spectral_split(w, lam, mu)[k])(v))
        assert np.isfinite(g).all(), f"non-finite gradient of psi_{k} at {name}"


def test_split_is_signed_correctly():
    """Pure tension has no compressive part, pure compression no tensile part."""
    lam, mu = materials.lame(E, NU)

    tension = materials.strain_voigt(jnp.array([[1e-3, 0.0], [0.0, 1e-3]]))
    pp, pm = spectral_split(tension, lam, mu)
    assert float(pp) > 0 and float(pm) < 1e-12

    compression = materials.strain_voigt(jnp.array([[-1e-3, 0.0], [0.0, -1e-3]]))
    pp, pm = spectral_split(compression, lam, mu)
    assert float(pm) > 0 and float(pp) < 1e-12


# -- the two subproblems ---------------------------------------------------


def test_undamaged_phase_field_reduces_to_elasticity():
    """At phi = 0 the displacement subproblem must be plain elasticity."""
    Vu, _ = spaces()
    basis = evaluate(Vu, P.gauss(Vu))
    zero = jnp.zeros((basis.n_elems, basis.n_q))
    bcs = [jx.DirichletBC([0.0, 0.0], where="bottom"),
           jx.DirichletBC([0.0, 1e-3], where="top")]

    pf = jx.solve(
        jx.PhaseFieldDisplacement(Vu, plane="strain", dirichlet=bcs),
        params={"E": E, "nu": NU, "phi": PointField(zero)},
    )
    el = jx.solve(
        jx.LinearElasticity(Vu, plane="strain", dirichlet=bcs), params={"E": E, "nu": NU}
    )
    # the only difference is the residual stiffness k
    np.testing.assert_allclose(np.asarray(pf.u), np.asarray(el.u), atol=1e-12)


def test_damage_softens_the_structure():
    Vu, _ = spaces()
    basis = evaluate(Vu, P.gauss(Vu))
    bcs = [jx.DirichletBC([0.0, 0.0], where="bottom"),
           jx.DirichletBC([0.0, 1e-3], where="top")]
    problem = jx.PhaseFieldDisplacement(Vu, plane="strain", dirichlet=bcs)
    top = np.asarray(Vu.boundaries["top"].dofs)

    forces = []
    for level in (0.0, 0.3, 0.6):
        phi = jnp.full((basis.n_elems, basis.n_q), level)
        params = {"E": E, "nu": NU, "phi": PointField(phi)}
        sol = jx.solve(problem, params=params)
        forces.append(float(jnp.sum(reactions(sol, params=params)[2 * top + 1])))

    assert forces[0] > forces[1] > forces[2] > 0, forces
    # isotropic degradation scales the stiffness by (1-phi)^2
    np.testing.assert_allclose(forces[1] / forces[0], 0.7**2, rtol=1e-6)


def test_damage_problem_localizes_around_the_history_field():
    Vu, Vd = spaces(deg=2, refine=3)
    basis = evaluate(Vu, P.gauss(Vu))
    ell, Gc = 0.06, 2.7

    history = initial_history(basis, edge_crack_distance((0.5, 0.5)), 1e3, ell, Gc)
    sol_d = jx.solve(
        jx.PhaseFieldDamage(Vd),
        params={"Gc": Gc, "l": ell, "H": PointField(history)},
    )
    phi = np.asarray(sol_d.at(basis=basis)[..., 0])
    x = np.asarray(basis.x)

    assert phi.max() > 0.9, "the seeded crack should be fully damaged"
    # damage concentrates on the notch: y ~ 0.5 and x < 0.5
    on_crack = (np.abs(x[..., 1] - 0.5) < ell) & (x[..., 0] < 0.5)
    assert phi[on_crack].mean() > 5 * phi[~on_crack].mean()


def test_history_field_is_monotone():
    """Irreversibility: the history field can only grow."""
    Vu, _ = spaces()
    basis = evaluate(Vu, P.gauss(Vu))
    zero = jnp.zeros((basis.n_elems, basis.n_q))
    params = {"E": E, "nu": NU}

    H = jnp.zeros((basis.n_elems, basis.n_q))
    prev = None
    for u_top in (1e-3, 2e-3, 1e-3):  # note the unload at the end
        problem = jx.PhaseFieldDisplacement(
            Vu, plane="strain",
            dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                       jx.DirichletBC([0.0, u_top], where="top")],
        )
        sol = jx.solve(problem, params={**params, "phi": PointField(zero)})
        H = jnp.maximum(H, tensile_energy(sol, params, basis))
        if prev is not None:
            assert float(jnp.min(H - prev)) >= -1e-14, "history decreased"
        prev = H

    assert float(jnp.max(H)) > 0


def test_staggered_loop_produces_a_softening_curve():
    """A short staggered run: the reaction must peak and then fall."""
    Vu, Vd = spaces(deg=2, refine=4)
    basis = evaluate(Vu, P.gauss(Vu))
    ell, Gc = 0.06, 2.7
    top = np.asarray(Vu.boundaries["top"].dofs)

    history = initial_history(basis, edge_crack_distance((0.5, 0.5)), 1e3, ell, Gc)
    phi = jnp.zeros((basis.n_elems, basis.n_q))
    damage = jx.PhaseFieldDamage(Vd)

    forces, areas = [], []
    for step in range(1, 9):
        u_top = 0.012 * step / 8
        disp = jx.PhaseFieldDisplacement(
            Vu, plane="strain",
            dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                       jx.DirichletBC([0.0, u_top], where="top")],
        )
        for _ in range(6):
            p = {"E": E, "nu": NU, "phi": PointField(phi)}
            sol_u = jx.solve(disp, params=p)
            history = jnp.maximum(history, tensile_energy(sol_u, {"E": E, "nu": NU}, basis))
            sol_d = jx.solve(damage, params={"Gc": Gc, "l": ell, "H": PointField(history)})
            new = jnp.clip(sol_d.at(basis=basis)[..., 0], 0.0, 1.0)
            if float(jnp.max(jnp.abs(new - phi))) < 1e-4:
                phi = new
                break
            phi = new

        forces.append(float(jnp.sum(reactions(sol_u, params=p)[2 * top + 1])))
        areas.append(float(jnp.sum(basis.w * phi)))

    peak = int(np.argmax(forces))
    assert 0 < peak < len(forces) - 1, f"no interior peak in {forces}"
    assert forces[-1] < 0.5 * forces[peak], "structure did not soften"
    assert all(b >= a - 1e-9 for a, b in zip(areas, areas[1:])), "damage decreased"


# -- graded meshes ---------------------------------------------------------


def test_graded_knots_cluster_where_asked():
    from jaxiga.geometry.primitives import graded_knots

    k = graded_knots(n_fine=20, n_coarse=3, centre=0.5, half_width=0.1)
    assert np.all((k > 0) & (k < 1)), "interior knots only"
    assert np.all(np.diff(k) > 0), "knots must be sorted and distinct"

    inside = k[(k >= 0.4 - 1e-12) & (k <= 0.6 + 1e-12)]
    h_fine = np.diff(inside).max()
    h_coarse = np.diff(k[k <= 0.4 + 1e-12]).max()
    assert h_fine < h_coarse / 3, f"band not refined: {h_fine} vs {h_coarse}"


def test_graded_knots_rejects_bad_bands():
    from jaxiga.geometry.primitives import graded_knots

    with pytest.raises(ValueError, match="half_width"):
        graded_knots(10, 3, half_width=0.7)
    with pytest.raises(ValueError, match="strictly inside"):
        graded_knots(10, 3, centre=0.05, half_width=0.2)


def test_graded_mesh_preserves_geometry_and_solves():
    """A graded patch is the same geometry, just discretised differently."""
    from jaxiga.geometry.nurbs import evaluate_patch
    from jaxiga.geometry.primitives import graded_knots

    base = jx.primitives.rectangle(0, 0, 1, 1).elevate(2)
    graded = base.insert_knots(
        np.linspace(0, 1, 9)[1:-1], graded_knots(16, 3, half_width=0.15)
    )

    rng = np.random.default_rng(0)
    xi = rng.uniform(0, 1, size=(20, 2))
    np.testing.assert_allclose(
        evaluate_patch(graded, xi), evaluate_patch(base, xi), rtol=1e-11, atol=1e-12
    )

    # and it still solves: a linear field must be reproduced exactly
    V = jx.FunctionSpace(graded)
    exact = lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1]  # noqa: E731
    problem = jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(lambda x, p: 1.0 + 2.0 * x[0] - 3.0 * x[1],
                                  where=lambda x: jnp.full(x.shape[0], True))],
        source=lambda x, p: 0.0,
    )
    assert jx.errornorm(jx.solve(problem, params={"a0": 1.0}), exact, "L2") < 1e-12


# -- adaptive staggered solver ---------------------------------------------


def notched_square(base=6, levels=2, deg=2, ell=0.05):
    """A coarse notched unit square, refined only around the initial crack."""
    from jaxiga.solvers.phase_field import seed_refine

    kn = np.linspace(0.0, 1.0, base + 1)[1:-1]
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).insert_knots(kn, kn)
    V = jx.refine_elements(jx.FunctionSpace(patch, vec=1), [])
    crack = edge_crack_distance((0.5, 0.5))
    return seed_refine(V, crack, ell, levels), crack


def tension_solver(V, crack, ell=0.05, **kwargs):
    from jaxiga.solvers.phase_field import (
        Adaptivity,
        Fracture,
        Material,
        StaggeredSolver,
    )

    opts = dict(max_level=3, dilate=1.0)
    opts.update(kwargs.pop("adaptivity", {}))
    return StaggeredSolver(
        V,
        Material(E, NU, plane="strain"),
        Fracture(Gc=2.7, ell=ell),
        dirichlet=[
            jx.DirichletBC([0.0, 0.0], where="bottom"),
            jx.DirichletBC([0.0, 1.0], where="top"),
        ],
        reaction=("top", 1),
        crack=crack,
        adaptivity=Adaptivity(**opts),
        **kwargs,
    )


def test_vector_space_is_the_scalar_space_with_more_components():
    """The solver builds its displacement space by changing `vec`, not rebuilding."""
    import dataclasses

    from jaxiga.space.hierarchical import build_hierarchical_space

    V, _ = notched_square(base=4, levels=1)
    cheap = dataclasses.replace(V, vec=2)
    built = build_hierarchical_space(V.patches, V.hierarchy, vec=2)
    assert cheap.n_dofs == built.n_dofs
    assert np.array_equal(np.asarray(cheap.elem_dofs), np.asarray(built.elem_dofs))
    assert sorted(cheap.boundaries) == sorted(built.boundaries)
    for label in built.boundaries:
        assert np.array_equal(
            np.asarray(cheap.boundaries[label].dofs),
            np.asarray(built.boundaries[label].dofs),
        )


def test_seed_refine_only_touches_the_crack():
    """The initial refinement must reach the crack and leave the rest coarse."""
    V, crack = notched_square(base=6, levels=2)
    box = np.asarray(V.elem_vertex)
    level = np.asarray(V.elem_key)[:, 1]

    # the far field is untouched; the crack line is at the deepest level
    assert (level == 0).any(), "nothing was left coarse"
    deep = level == 2
    assert deep.any(), "the crack was not resolved"
    assert np.abs(0.5 * (box[deep, 1] + box[deep, 3]) - 0.5).max() < 0.25
    # and the refinement reaches the notch mouth on the left edge
    assert box[deep, 0].min() < 1e-12
    # nothing was refined in the half of the plate the notch does not touch
    assert box[level > 0, 3].max() < 0.85 and box[level > 0, 1].min() > 0.15


def test_adaptive_run_refines_along_the_crack_and_softens():
    """Damage-driven refinement must follow the crack and produce a peak."""
    V, crack = notched_square(base=6, levels=2)
    solver = tension_solver(V, crack, adaptivity={"max_level": 3})
    start = solver.mesh.Vd.n_elems

    loads = np.cumsum(np.full(40, 5e-4))
    forces, elems = [], []
    for step in solver.run(loads, keep_solutions=False):
        forces.append(step.force)
        elems.append(step.n_elems)
    forces = np.array(forces)

    assert solver.remeshes > 0
    assert elems[-1] > start
    peak = int(np.argmax(forces))
    assert 0 < peak < len(forces) - 1, "no peak: the specimen never cracked"
    assert forces[-1] < 0.6 * forces[peak], "no softening after the peak"

    # refinement went to the crack, not everywhere
    box = np.asarray(solver.mesh.Vd.elem_vertex)
    deep = np.asarray(solver.mesh.Vd.elem_key)[:, 1] == 3
    assert deep.any()
    assert np.abs(0.5 * (box[deep, 1] + box[deep, 3]) - 0.5).max() < 0.25


def test_a_remesh_preserves_the_state_it_carries():
    """Damage, history and the reaction must survive a remesh unchanged.

    The sharp form of "adapting does not change the physics". The spaces are
    nested, so the damage field has an exact representation on the refined
    mesh and the projection must find it; the history field is carried point by
    point; and with both preserved, re-solving the displacement at the same
    load must return the same force.
    """
    from jaxiga.post.solution import Solution
    from jaxiga.space import pointset as P
    from jaxiga.space.evaluation import evaluate
    from jaxiga.space.transfer import coarse_basis_at, parent_map

    def force_at(mesh, load, phi):
        """Solve the displacement subproblem on `mesh` and report the reaction."""
        b, data = mesh.assemble_u(load, phi)
        u_free = mesh.lu_u.solve(np.asarray(data), np.asarray(b))
        return float(mesh.reaction_sum(mesh.lift_u(jnp.asarray(u_free), load), phi))

    V, crack = notched_square(base=6, levels=2)
    solver = tension_solver(V, crack, adaptivity={"max_level": 2})
    load = 6 * 5e-4
    for _ in solver.run(np.cumsum(np.full(6, 5e-4)), keep_solutions=False):
        pass
    assert float(jnp.max(solver.phi)) > 0.5, "nothing to carry"

    before = solver.mesh
    phi_before, hist_before = solver.phi_dofs, solver.hist
    force_before = force_at(before, load, solver.phi)

    # force a remesh of the coarse elements around the crack
    box = np.asarray(before.Vd.elem_vertex)
    level = np.asarray(before.Vd.elem_key)[:, 1]
    marked = np.flatnonzero((level < 2) & (np.abs(0.5 * (box[:, 1] + box[:, 3]) - 0.5) < 0.3))
    assert marked.size
    after = solver._remesh(marked)
    assert after.Vd.n_elems > before.Vd.n_elems

    # the damage field is the same function, sampled on the new quadrature set
    pmap = parent_map(before.Vd, after.Vd)
    ps = P.gauss(after.Vu)
    old_at_new = Solution(before.Vd, phi_before).at(
        basis=coarse_basis_at(before.Vd, pmap, ps)
    )[..., 0]
    new_at_new = Solution(after.Vd, solver.phi_dofs).at(
        basis=evaluate(after.Vd, ps)
    )[..., 0]
    assert np.abs(np.asarray(new_at_new - old_at_new)).max() < 1e-10

    # the history never decreases anywhere it was already set
    carried = np.asarray(solver.hist)
    assert carried.min() >= 0.0
    assert np.abs(carried.max() - float(jnp.max(hist_before))) < 1e-9 * float(
        jnp.max(hist_before)
    )

    # and the same load returns the same force
    assert abs(force_at(after, load, solver.phi) - force_before) < 0.02 * abs(force_before)


def test_history_field_survives_a_remesh():
    """Damage may not heal across a remesh: the transferred history must hold it."""
    V, crack = notched_square(base=6, levels=1)
    solver = tension_solver(V, crack, adaptivity={"max_level": 3})
    loads = np.cumsum(np.full(12, 1e-3))

    peak_damage = []
    for step in solver.run(loads, keep_solutions=False):
        peak_damage.append(float(jnp.max(solver.phi)))
    assert solver.remeshes > 0
    d = np.array(peak_damage)
    assert d.max() > 0.9
    assert np.all(np.diff(d) > -1e-3), "damage decreased across a load step"


def test_cg_and_direct_solvers_agree():
    """The 3D default solver must reproduce what the 2D default produces.

    Conjugate gradients replace the cached factorisation in 3D, where fill-in
    makes a direct solve cost more than the assembly feeding it. The two paths
    solve the same systems, so asked for the same accuracy they must return the
    same load-displacement curve.

    Both are held to 1e-10 here. At their defaults they part company at the 1e-3
    level after the peak, because the direct path's tolerance is not a solve
    accuracy but the point at which a *reused* factorisation is judged no longer
    good enough -- and on the softening branch small differences compound
    through the staggered iteration.
    """
    V, crack = notched_square(base=6, levels=2)
    loads = np.cumsum(np.full(15, 5e-4))

    def curve(kind):
        solver = tension_solver(
            V, crack, adaptivity={"max_level": 2}, linear=kind, linear_tol=1e-10
        )
        forces = np.array([s.force for s in solver.run(loads, keep_solutions=False)])
        return solver, forces

    direct, f_direct = curve("direct")
    iterative, f_cg = curve("cg")

    assert direct.linear_kind == "direct" and iterative.linear_kind == "cg"
    assert iterative.mesh.lu_u.failures == 0
    np.testing.assert_allclose(f_cg, f_direct, rtol=1e-6)


def test_solver_picks_a_solver_by_dimension():
    """3D gets conjugate gradients without being asked; 2D keeps the factorisation."""
    from jaxiga.solvers.phase_field import (
        CachedLU,
        Fracture,
        JacobiCG,
        Material,
        StaggeredSolver,
    )

    plate = jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1)
    flat = StaggeredSolver(
        jx.FunctionSpace(plate, vec=1), Material(E, NU), Fracture(2.7, 0.2),
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                   jx.DirichletBC([0.0, 1.0], where="top")],
        reaction=("top", 1),
    )
    assert isinstance(flat.mesh.lu_u, CachedLU)

    block = jx.primitives.cuboid((0, 0, 0), (1, 1, 1)).elevate(2).refine(1)
    solid = StaggeredSolver(
        jx.FunctionSpace(block, vec=1), Material(E, NU), Fracture(2.7, 0.2),
        dirichlet=[jx.DirichletBC([0.0, 0.0, 0.0], where="back"),
                   jx.DirichletBC([0.0, 0.0, 1.0], where="front")],
        reaction=("front", 2),
    )
    assert isinstance(solid.mesh.lu_u, JacobiCG)
    assert solid.mesh.Vu.vec == 3


def test_solver_rejects_an_unknown_linear_solver():
    from jaxiga.solvers.phase_field import Fracture, Material, StaggeredSolver

    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1)
    with pytest.raises(ValueError, match="linear must be"):
        StaggeredSolver(
            jx.FunctionSpace(patch, vec=1), Material(E, NU), Fracture(2.7, 0.2),
            dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom")],
            reaction=("top", 1), linear="lapack",
        )


def test_solver_rejects_a_vector_damage_space():
    from jaxiga.solvers.phase_field import Fracture, Material, StaggeredSolver

    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1)
    with pytest.raises(ValueError, match="scalar"):
        StaggeredSolver(
            jx.FunctionSpace(patch, vec=2),
            Material(E, NU),
            Fracture(2.7, 0.1),
            dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom")],
            reaction=("top", 1),
        )
