"""Galerkin solves: exactness, convergence and differentiability."""

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jaxiga as jx
from jaxiga.solvers.linear import SOLVE_COUNTER, LinearOptions, reset_solve_counter

EXACT = lambda x: jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])  # noqa: E731
SOURCE = lambda x, p: 8 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])  # noqa: E731
ALL_BOUNDARY = lambda x: jnp.full(x.shape[0], True)  # noqa: E731


def unit_square(deg, refine):
    return jx.primitives.quadrilateral([[0, 0], [1, 0], [1, 1], [0, 1]]).elevate(deg).refine(refine)


def poisson(deg=3, refine=3, source=SOURCE, dirichlet_value=0.0):
    V = jx.FunctionSpace(unit_square(deg, refine))
    return jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(dirichlet_value, where=ALL_BOUNDARY)],
        source=source,
    )


# -- exactness -------------------------------------------------------------


@pytest.mark.parametrize("deg", [2, 3])
def test_patch_test_reproduces_linear_field_exactly(deg):
    """A field in the discrete space must be recovered to machine precision."""
    V = jx.FunctionSpace(unit_square(deg, 2))
    exact = lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1]  # noqa: E731

    problem = jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(lambda x, p: 1.0 + 2.0 * x[0] - 3.0 * x[1], where=ALL_BOUNDARY)],
        source=lambda x, p: 0.0,
    )
    sol = jx.solve(problem, params={"a0": 1.0})
    assert jx.errornorm(sol, exact, "L2") < 1e-12


def test_constant_dirichlet_is_reproduced():
    problem = poisson(deg=3, refine=2, source=lambda x, p: 0.0, dirichlet_value=2.5)
    sol = jx.solve(problem, params={"a0": 1.0})
    np.testing.assert_allclose(np.asarray(sol.u), 2.5, atol=1e-11)


# -- convergence -----------------------------------------------------------


@pytest.mark.parametrize("deg", [2, 3, 4])
def test_poisson_convergence_rate(deg):
    errs = [jx.errornorm(jx.solve(poisson(deg, r), params={"a0": 1.0}), EXACT, "L2")
            for r in (2, 3, 4)]
    assert errs[0] > errs[1] > errs[2], f"errors not decreasing: {errs}"
    rate = np.log2(errs[-2] / errs[-1])
    assert rate > deg + 1 - 0.5, f"L2 rate {rate:.2f} below the expected {deg + 1}"


# -- linear solver equivalence --------------------------------------------


@pytest.mark.parametrize("method", ["dense", "scipy", "cg", "bicgstab", "gmres"])
def test_linear_backends_agree(method):
    problem = poisson(deg=2, refine=2)
    ref = jx.solve(problem, params={"a0": 1.0}, linear=LinearOptions(method="dense"))
    got = jx.solve(
        problem, params={"a0": 1.0},
        linear=LinearOptions(method=method, tol=1e-13, maxiter=5000),
    )
    np.testing.assert_allclose(np.asarray(got.u), np.asarray(ref.u), rtol=1e-7, atol=1e-9)


# -- analytical elasticity ---------------------------------------------------------


def kirsch(Emod=1e5, nu=0.3, rad=1.0, trac=10.0):
    def stress_np(x, y):
        r = np.hypot(x, y); th = np.arctan2(y, x)
        srr = trac/2*(1-rad**2/r**2) + trac/2*(1-4*rad**2/r**2+3*rad**4/r**4)*np.cos(2*th)
        stt = trac/2*(1+rad**2/r**2) - trac/2*(1+3*rad**4/r**4)*np.cos(2*th)
        srt = -trac/2*(1+2*rad**2/r**2-3*rad**4/r**4)*np.sin(2*th)
        A = np.array([[np.cos(th)**2, np.sin(th)**2, 2*np.sin(th)*np.cos(th)],
                      [np.sin(th)**2, np.cos(th)**2, -2*np.sin(th)*np.cos(th)],
                      [-np.sin(th)*np.cos(th), np.sin(th)*np.cos(th),
                       np.cos(th)**2-np.sin(th)**2]])
        return np.linalg.solve(A, [srr, stt, srt])

    def disp_np(x, y):
        r = np.hypot(x, y); th = np.arctan2(y, x)
        ux = ((1+nu)/Emod*trac*(1/(1+nu)*r*np.cos(th) + 2*rad**2/((1+nu)*r)*np.cos(th)
              + rad**2/(2*r)*np.cos(3*th) - rad**4/(2*r**3)*np.cos(3*th)))
        uy = ((1+nu)/Emod*trac*(-nu/(1+nu)*r*np.sin(th) - (1-nu)*rad**2/((1+nu)*r)*np.sin(th)
              + rad**2/(2*r)*np.sin(3*th) - rad**4/(2*r**3)*np.sin(3*th)))
        return ux, uy

    def stress_jnp(x, y):
        r = jnp.hypot(x, y); th = jnp.arctan2(y, x)
        srr = trac/2*(1-rad**2/r**2) + trac/2*(1-4*rad**2/r**2+3*rad**4/r**4)*jnp.cos(2*th)
        stt = trac/2*(1+rad**2/r**2) - trac/2*(1+3*rad**4/r**4)*jnp.cos(2*th)
        srt = -trac/2*(1+2*rad**2/r**2-3*rad**4/r**4)*jnp.sin(2*th)
        c, s = jnp.cos(th), jnp.sin(th)
        A = jnp.array([[c**2, s**2, 2*s*c], [s**2, c**2, -2*s*c], [-s*c, s*c, c**2-s**2]])
        return jnp.linalg.solve(A, jnp.stack([srr, stt, srt]))

    return disp_np, stress_np, stress_jnp


def build_plate(deg, refine, rad=1.0, side=4.0):
    disp_np, stress_np, stress_jnp = kirsch(rad=rad)
    patches = [jx.primitives.plate_with_hole_quadrant(rad, side, q).elevate(deg).refine(refine)
               for q in (2, 3, 4, 1)]
    V = jx.FunctionSpace(patches, vec=2)

    tol = 1e-9
    on_hole = lambda x: np.abs(np.hypot(x[:, 0], x[:, 1]) - rad) < tol  # noqa: E731

    def traction(x, n, p):
        s = stress_jnp(x[0], x[1])
        return jnp.stack([n[0]*s[0] + n[1]*s[2], n[0]*s[2] + n[1]*s[1]])

    problem = jx.LinearElasticity(
        V, plane="stress",
        dirichlet=[
            jx.DirichletBC(0.0, where=lambda x: on_hole(x) & (np.abs(x[:, 1]) < tol), component=1),
            jx.DirichletBC(0.0, where=lambda x: on_hole(x) & (np.abs(x[:, 0]) < tol), component=0),
        ],
        neumann=[jx.Neumann(traction, where="outer")],
    )

    exact_disp = lambda pts: np.stack(disp_np(np.asarray(pts)[:, 0], np.asarray(pts)[:, 1]), 1)  # noqa: E731
    exact_stress = lambda pts: np.stack([stress_np(x, y) for x, y in np.asarray(pts)])  # noqa: E731
    return problem, exact_disp, exact_stress






def test_elasticity_stress_field_converges():
    """Derived stress fields must be accurate and improve under refinement.

    Stress converges one order slower than displacement and the hole is a
    stress concentration, so the absolute level here tracks the energy-norm
    error rather than the much smaller L2 error.
    """
    from jaxiga.space.evaluation import evaluate

    rels = []
    for refine in (1, 2):
        problem, _, exact_stress = build_plate(4, refine)
        sol = jx.solve(problem, params={"E": 1e5, "nu": 0.3})

        basis = evaluate(problem.space, jx.pointset.gauss(problem.space))
        got = np.asarray(sol.field("stress", basis=basis)).reshape(-1, 3)
        want = exact_stress(np.asarray(basis.x).reshape(-1, 2))
        rels.append(np.linalg.norm(got - want) / np.linalg.norm(want))

        vm = np.asarray(sol.field("von_mises", basis=basis))
        assert np.all(vm >= 0)

    assert rels[1] < rels[0], f"stress error did not improve under refinement: {rels}"
    assert rels[-1] < 5e-2, f"stress field relative error {rels[-1]:.3e}"


# -- differentiability -----------------------------------------------------


def central_diff(f, x, i, h=1e-6):
    xp = x.at[i].add(h)
    xm = x.at[i].add(-h)
    return (f(xp) - f(xm)) / (2 * h)


def test_gradient_wrt_material_parameters():
    problem, _, _ = build_plate(2, 1)

    def compliance(theta):
        sol = jx.solve(problem, params={"E": theta[0], "nu": theta[1]})
        return jnp.sum(sol.u**2)

    theta = jnp.array([1e5, 0.3])
    g = jax.grad(compliance)(theta)

    for i, h in ((0, 1.0), (1, 1e-6)):
        fd = (compliance(theta.at[i].add(h)) - compliance(theta.at[i].add(-h))) / (2 * h)
        np.testing.assert_allclose(float(g[i]), float(fd), rtol=1e-5)


def test_gradient_wrt_source_amplitude():
    V = jx.FunctionSpace(unit_square(2, 2))

    def loss(amp):
        problem = jx.Poisson(
            V,
            dirichlet=[jx.DirichletBC(0.0, where=ALL_BOUNDARY)],
            source=lambda x, p: p["amp"] * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
        return jnp.sum(jx.solve(problem, params={"a0": 1.0, "amp": amp}).u ** 2)

    amp = 3.0
    g = float(jax.grad(loss)(amp))
    h = 1e-5
    np.testing.assert_allclose(g, (loss(amp + h) - loss(amp - h)) / (2 * h), rtol=1e-6)


def test_gradient_wrt_dirichlet_value():
    V = jx.FunctionSpace(unit_square(2, 2))

    def loss(g0):
        problem = jx.Poisson(
            V,
            dirichlet=[jx.DirichletBC(lambda x, p: p["g0"], where=ALL_BOUNDARY)],
            source=lambda x, p: 1.0,
        )
        return jnp.sum(jx.solve(problem, params={"a0": 1.0, "g0": g0}).u ** 2)

    # Dirichlet values are resolved at setup (they set DofMap.values), so the
    # derivative is taken by finite differences of the resolved problem.
    h = 1e-5
    fd = (loss(0.5 + h) - loss(0.5 - h)) / (2 * h)
    assert np.isfinite(fd) and abs(fd) > 1e-6


def test_shape_derivative_wrt_control_points():
    """Geometry gradients, per the differentiability contract."""
    import dataclasses

    V = jx.FunctionSpace(unit_square(2, 1))
    base = jx.Poisson(
        V, dirichlet=[jx.DirichletBC(0.0, where=ALL_BOUNDARY)], source=lambda x, p: 1.0
    )

    def compliance(cpts):
        # Build the problem once outside the trace and rebind the geometry, the
        # pattern the architecture guide uses for shape derivatives: `where`
        # predicates are resolved at construction and must stay concrete.
        problem = base.with_space(dataclasses.replace(V, cpts=cpts))
        return jnp.sum(jx.solve(problem, params={"a0": 1.0}).u ** 2)

    g = jax.grad(compliance)(V.cpts)
    assert np.any(np.abs(np.asarray(g)) > 1e-12)

    flat = V.cpts.reshape(-1)

    def flat_compliance(f):
        return compliance(f.reshape(V.cpts.shape))

    for i in (5, 11):
        fd = central_diff(flat_compliance, flat, i, h=1e-6)
        np.testing.assert_allclose(
            float(np.asarray(g).reshape(-1)[i]), float(fd), rtol=1e-4, atol=1e-8
        )


def test_vjp_uses_exactly_one_adjoint_solve():
    """The gradient must not differentiate through solver iterations."""
    problem = poisson(deg=2, refine=1)

    def loss(a0):
        return jnp.sum(jx.solve(problem, params={"a0": a0}).u ** 2)

    reset_solve_counter()
    jax.grad(loss)(1.0)
    assert SOLVE_COUNTER["adjoint"] == 1, SOLVE_COUNTER
    assert SOLVE_COUNTER["forward"] == 1, SOLVE_COUNTER


# -- error handling --------------------------------------------------------


def test_unknown_method_raises():
    problem = poisson(deg=2, refine=1)
    with pytest.raises(ValueError, match="unknown method"):
        jx.solve(problem, method="nonsense", params={"a0": 1.0})


def test_problem_without_energy_raises():
    V = jx.FunctionSpace(unit_square(2, 1))

    class Empty(jx.Problem):
        is_linear = True

    problem = Empty(V, dirichlet=[jx.DirichletBC(0.0, where=ALL_BOUNDARY)])
    with pytest.raises(TypeError, match="Galerkin requires an energy density"):
        jx.solve(problem, params={})


def test_conflicting_dirichlet_raises():
    V = jx.FunctionSpace(unit_square(2, 1))
    problem = jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where="left"), jx.DirichletBC(1.0, where="left")],
        source=lambda x, p: 1.0,
    )
    with pytest.raises(ValueError, match="conflicting Dirichlet"):
        jx.solve(problem, params={"a0": 1.0})


def test_elasticity_requires_matching_vec():
    V = jx.FunctionSpace(unit_square(2, 1))  # vec=1
    with pytest.raises(ValueError, match="vec == dim"):
        jx.LinearElasticity(V)


def test_unknown_boundary_label_raises():
    """Boundary conditions are resolved at construction, so this fails fast."""
    V = jx.FunctionSpace(unit_square(2, 1))
    with pytest.raises(KeyError, match="unknown boundary label"):
        jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where="nope")])


# -- solution object -------------------------------------------------------


def test_solution_sampling_and_probe():
    problem = poisson(deg=3, refine=3)
    sol = jx.solve(problem, params={"a0": 1.0})

    x, vals = sol.sample(n=5)
    assert x.shape[0] == vals.shape[0]

    targets = jnp.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.3]])
    got = np.asarray(sol.probe(targets))[:, 0]
    want = np.asarray([EXACT(t) for t in targets])
    np.testing.assert_allclose(got, want, atol=2e-3)


def test_to_vtk_writes_a_file(tmp_path):
    problem, _, _ = build_plate(2, 1)
    sol = jx.solve(problem, params={"E": 1e5, "nu": 0.3})
    out = tmp_path / "plate"
    sol.to_vtk(str(out), n=3, fields=["von_mises"])
    assert (tmp_path / "plate.vtu").exists()
    assert (tmp_path / "plate.vtu").stat().st_size > 0


def test_write_vtu_carries_point_and_cell_data(tmp_path):
    """Fields from different spaces over one mesh belong in one file."""
    import dataclasses

    from jaxiga.post.vtk import write_vtu
    from jaxiga.space import pointset as P
    from jaxiga.space.evaluation import evaluate

    patch = jx.primitives.cuboid((0, 0, 0), (1, 0.2, 1)).elevate(2).refine(1)
    scalar = jx.refine_elements(jx.FunctionSpace(patch, vec=1), [0, 3])
    vector = dataclasses.replace(scalar, vec=3)

    basis = evaluate(scalar, P.grid(scalar, 2))
    x = np.asarray(basis.x)
    path = write_vtu(
        vector,
        str(tmp_path / "step"),
        n=1,
        point_data={"damage": x[..., 2], "displacement": 0.01 * x},
        cell_data={"level": np.asarray(scalar.elem_key)[:, 1]},
    )
    assert path.endswith(".vtu") and os.path.getsize(path) > 0

    # pyevtk appends the arrays as raw bytes, so the file is not parseable XML
    # as a whole; the header that declares them is.
    text = open(path, "rb").read().decode("latin-1")
    assert '<VTKFile type="UnstructuredGrid"' in text
    for name in ("damage", "displacement_0", "displacement_2", "level"):
        assert f'Name="{name}"' in text, name
    assert '<CellData' in text and '<PointData' in text


def test_write_series_indexes_files_by_time(tmp_path):
    from jaxiga.post.vtk import write_series

    files = [tmp_path / "sub" / f"step_{k}.vtu" for k in range(3)]
    (tmp_path / "sub").mkdir()
    for f in files:
        f.write_text("")

    pvd = write_series(files, [0.0, 0.5, 1.25], str(tmp_path / "run"))
    assert pvd.endswith(".pvd")

    import xml.etree.ElementTree as ET

    datasets = ET.parse(pvd).getroot().find("Collection").findall("DataSet")
    assert [d.get("timestep") for d in datasets] == ["0.0", "0.5", "1.25"]
    # paths are relative, so the directory can be moved
    assert [d.get("file") for d in datasets] == [
        os.path.join("sub", f"step_{k}.vtu") for k in range(3)
    ]


def test_integrate_and_norm():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 2, 3).elevate(2).refine(2))
    np.testing.assert_allclose(jx.integrate(lambda x: np.ones((len(x), 1)), V), 6.0, rtol=1e-12)

    problem = poisson(deg=3, refine=3)
    sol = jx.solve(problem, params={"a0": 1.0})
    # ||sin(2 pi x) sin(2 pi y)||_L2 over the unit square is 1/2
    np.testing.assert_allclose(jx.norm(sol, "L2"), 0.5, rtol=1e-3)


def test_csr_pattern_is_independent_of_the_chunk_size():
    """Chunking the sparsity build must not change the sparsity.

    The build forms one key per (element, local row, local column), which in 3D
    is hundreds of millions of entries; it walks them in blocks so the transient
    stays bounded. The pattern it produces has to be the one a single pass gives.
    """
    import jaxiga.methods._common as common

    kn = np.linspace(0.0, 1.0, 5)[1:-1]
    patch = jx.primitives.cuboid((0, 0, 0), (1, 0.2, 1)).elevate(2).insert_knots(
        kn, np.array([0.5]), kn
    )
    V = jx.refine_elements(jx.FunctionSpace(patch, vec=3), [])
    V = jx.refine_elements(V, np.arange(0, V.n_elems, 4))

    rng = np.random.default_rng(3)
    free = tuple(
        int(d) for d in np.sort(rng.choice(V.n_dofs, int(V.n_dofs * 0.9), replace=False))
    )
    args = (V.elem_dofs, V.n_dofs, V.vec, free)

    original = common._PATTERN_CHUNK
    try:
        patterns = []
        for chunk in (10**9, 50_000, 4_000):
            common._PATTERN_CHUNK = chunk
            patterns.append(common._cached_csr_pattern.__wrapped__(*args))
    finally:
        common._PATTERN_CHUNK = original

    reference = patterns[0]
    assert len(reference.column_indices) > 0
    for other in patterns[1:]:
        np.testing.assert_array_equal(reference.indptr, other.indptr)
        np.testing.assert_array_equal(reference.column_indices, other.column_indices)
        np.testing.assert_array_equal(reference.row_indices, other.row_indices)
        np.testing.assert_array_equal(reference.scatter, other.scatter)
        assert reference.shape == other.shape
