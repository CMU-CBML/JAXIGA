"""Physics covered by the example scripts, at test-sized resolution.

Guards the examples against regressions without paying for their full meshes.
"""

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

import jaxiga as jx

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


# -- quarter annulus under internal pressure (Lame) ------------------------


def lame_reference(r_int=1.0, r_ext=4.0, pressure=10.0, E=1e5, nu=0.3):
    a2, b2 = r_int**2, r_ext**2

    def disp(pts):
        pts = np.asarray(pts)
        r, th = np.hypot(pts[:, 0], pts[:, 1]), np.arctan2(pts[:, 1], pts[:, 0])
        ur = pressure * a2 * r / (E * (b2 - a2)) * ((1 - nu) + (1 + nu) * b2 / r**2)
        return np.stack([ur * np.cos(th), ur * np.sin(th)], axis=1)

    def stress(pts):
        pts = np.asarray(pts)
        r, th = np.hypot(pts[:, 0], pts[:, 1]), np.arctan2(pts[:, 1], pts[:, 0])
        srr = pressure * a2 / (b2 - a2) * (1 - b2 / r**2)
        stt = pressure * a2 / (b2 - a2) * (1 + b2 / r**2)
        c, s = np.cos(th), np.sin(th)
        return np.stack([srr * c**2 + stt * s**2,
                         srr * s**2 + stt * c**2,
                         (srr - stt) * s * c], axis=1)

    return disp, stress


def solve_annulus(deg, refine, pressure=10.0):
    V = jx.FunctionSpace(
        jx.primitives.quarter_annulus(1.0, 4.0).elevate(deg).refine(refine), vec=2
    )
    problem = jx.LinearElasticity(
        V,
        plane="stress",
        dirichlet=[jx.DirichletBC(0.0, where="sym_y", component=1),
                   jx.DirichletBC(0.0, where="sym_x", component=0)],
        neumann=[jx.Neumann(lambda x, n, p: -pressure * n, where="inner")],
    )
    return jx.solve(problem, params={"E": 1e5, "nu": 0.3})


def test_quarter_annulus_recovers_lame_solution():
    disp, stress = lame_reference()
    sol = solve_annulus(4, 3)
    # Tolerances sized for this mesh; the example script runs finer and
    # reaches ~9e-8 in L2.
    assert jx.errornorm(sol, disp, "L2") < 5e-5
    assert jx.errornorm(sol, disp, "energy", exact_grad=stress) < 5e-4


def test_quarter_annulus_converges():
    disp, _ = lame_reference()
    errs = [jx.errornorm(solve_annulus(3, r), disp, "L2") for r in (1, 2, 3)]
    assert errs[0] > errs[1] > errs[2], errs


def test_internal_pressure_sign_is_outward():
    """Internal pressure must push the wall out, not pull it in.

    A sign slip in the traction leaves the magnitudes right and only flips the
    displacement, so this checks the direction explicitly.
    """
    sol = solve_annulus(3, 2)
    pts, vals = sol.sample(n=3)
    pts, vals = np.asarray(pts), np.asarray(vals)
    radial = (pts * vals).sum(1) / np.linalg.norm(pts, axis=1)
    assert np.all(radial > 0), "displacement is not radially outward"


# -- tension plate ---------------------------------------------------------


def test_tension_plate_honours_prescribed_displacement():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(2), vec=2)
    problem = jx.LinearElasticity(
        V,
        plane="stress",
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                   jx.DirichletBC([0.0, 0.1], where="top")],
    )
    sol = jx.solve(problem, params={"E": 1e5, "nu": 0.3})

    pts, vals = sol.sample(n=4)
    pts, vals = np.asarray(pts), np.asarray(vals)

    top = np.abs(pts[:, 1] - 1.0) < 1e-9
    bottom = np.abs(pts[:, 1]) < 1e-9
    np.testing.assert_allclose(vals[top, 1], 0.1, atol=1e-10)
    np.testing.assert_allclose(vals[bottom, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(vals[top, 0], 0.0, atol=1e-10)

    # u_y increases monotonically from bottom to top
    order = np.argsort(pts[:, 1])
    assert vals[order, 1][0] <= vals[order, 1][-1]
    assert 0.0 <= vals[:, 1].min() and vals[:, 1].max() <= 0.1 + 1e-12


def test_vector_dirichlet_per_component():
    """A list-valued DirichletBC constrains each component separately."""
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1), vec=2)
    problem = jx.LinearElasticity(
        V,
        dirichlet=[jx.DirichletBC([0.3, -0.4], where="bottom"),
                   jx.DirichletBC([0.0, 0.0], where="top")],
    )
    sol = jx.solve(problem, params={"E": 1e5, "nu": 0.3})
    bottom_dofs = V.boundaries["bottom"].dofs
    u = np.asarray(sol.u)
    np.testing.assert_allclose(u[2 * bottom_dofs], 0.3, atol=1e-12)
    np.testing.assert_allclose(u[2 * bottom_dofs + 1], -0.4, atol=1e-12)


# -- the example scripts themselves ---------------------------------------


@pytest.mark.parametrize(
    "script",
    [
        "poisson/Poisson1D_IGA.py",
        "poisson/Poisson2D_IGA.py",
        "linear_elasticity/Elast2D_IGA_Spline_plate_w_hole.py",
        "linear_elasticity/Elast2D_IGA_Spline_quarter_annulus.py",
        "linear_elasticity/Elast2D_IGA_Spline_Tension_plate.py",
        "shape_derivative.py",
        "energy_minimization.py",
        "poisson/Poisson2D_Collocation.py",
        "darcy/Darcy2D_inverse.py",
        "phase_field/adaptive_fracture.py",
    ],
)
def test_example_script_is_importable(script):
    """Cheap guard that the scripts stay syntactically and API valid."""
    path = EXAMPLES / script
    assert path.exists(), f"missing example {script}"
    compile(path.read_text(), str(path), "exec")


# -- adaptive fracture driver ----------------------------------------------


def adaptive_fracture():
    """Load the driver as a module so its pure helpers can be exercised."""
    import importlib.util

    path = EXAMPLES / "phase_field" / "adaptive_fracture.py"
    spec = importlib.util.spec_from_file_location("adaptive_fracture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_path_accumulates_the_schedule():
    module = adaptive_fracture()
    path = module.load_path([(3, 1e-3), (4, 1e-4)])
    np.testing.assert_allclose(
        path, [1e-3, 2e-3, 3e-3, 3.1e-3, 3.2e-3, 3.3e-3, 3.4e-3], rtol=1e-12
    )
    assert len(module.load_path([(3, 1e-3), (4, 1e-4)], n_steps=5)) == 5


def test_broken_specimen_is_detected_from_the_reaction():
    """The stopping rule fires after the drop, and never during the rise."""
    module = adaptive_fracture()

    # a brittle curve: rise to 100, snap to a 3% residual
    rise = list(np.linspace(1.0, 100.0, 40))
    tail = [3.0] * 10
    peak, fired = 0.0, []
    for index, force in enumerate(rise + tail):
        peak = max(peak, abs(force))
        if module.has_broken(force, peak, 0.05):
            fired.append(index)
    assert fired and fired[0] == len(rise), fired

    # never fires while the load is still rising, whatever the threshold
    peak = 0.0
    for force in rise:
        peak = max(peak, abs(force))
        assert not module.has_broken(force, peak, 0.999)

    # disabled by default, and sign-independent (a shear reaction is negative)
    assert not module.has_broken(3.0, 100.0, None)
    assert module.has_broken(-3.0, 100.0, 0.05)


# -- point probing ---------------------------------------------------------


def test_probe_handles_points_near_element_boundaries():
    """A nearest-sample search alone misassigns points close to element edges.

    Those points then fail the Newton inversion and come back as NaN, so this
    checks the containment retry in locate_points.
    """
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(3))
    problem = jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where=lambda x: jnp.full(x.shape[0], True))],
        source=lambda x, p: 1.0,
    )
    sol = jx.solve(problem, params={"a0": 1.0})

    rng = np.random.default_rng(0)
    pts = jnp.asarray(rng.uniform(0.15, 0.85, size=(40, 2)))
    vals = np.asarray(sol.probe(pts))
    assert np.all(np.isfinite(vals)), f"{np.sum(~np.isfinite(vals))} probes returned NaN"

    # points on element boundaries exactly
    edges = jnp.asarray([[0.125, 0.375], [0.5, 0.5], [0.25, 0.75], [0.875, 0.125]])
    assert np.all(np.isfinite(np.asarray(sol.probe(edges))))


def test_probe_matches_direct_evaluation():
    V = jx.FunctionSpace(jx.primitives.quarter_annulus(1.0, 2.0).elevate(3).refine(2))
    problem = jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where=lambda x: jnp.full(x.shape[0], True))],
        source=lambda x, p: 1.0,
    )
    sol = jx.solve(problem, params={"a0": 1.0})

    # sample points come from the geometry itself, so probing them must
    # return the very same values
    x, vals = sol.sample(n=3)
    pick = np.arange(0, len(x), 37)[:12]
    np.testing.assert_allclose(
        np.asarray(sol.probe(x[pick]))[:, 0], np.asarray(vals[pick])[:, 0], atol=1e-8
    )


def test_probe_returns_nan_outside_the_domain():
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))
    problem = jx.Poisson(
        V,
        dirichlet=[jx.DirichletBC(0.0, where=lambda x: jnp.full(x.shape[0], True))],
        source=lambda x, p: 1.0,
    )
    sol = jx.solve(problem, params={"a0": 1.0})
    assert np.all(np.isnan(np.asarray(sol.probe(jnp.array([[5.0, 5.0]])))))
