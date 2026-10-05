"""Cook's membrane, neo-Hookean: Newton and energy minimization on one problem.

The standard bending-dominated benchmark. A tapered panel, clamped on the left
edge and loaded by a vertical shear on the right, is solved with a compressible
neo-Hookean material; the reported quantity is the vertical displacement of the
upper right corner.

The point of the example is that ``method="galerkin"`` and ``method="energy"``
are two ways of finding the stationary point of the *same* discrete potential,
so they must agree to solver tolerance. Galerkin drives Newton with the
autodiff-consistent tangent; energy minimization runs L-BFGS on the potential
directly and never forms a tangent at all. Agreement is a real cross-check of
both the physics kernel and the two drivers, not a tautology -- they share only
the energy density.
"""

import time

import numpy as np
import jax.numpy as jnp

import jaxiga as jx

DEG, REFINE = 2, 3
E, NU = 240.0, 0.4
SHEAR = 6.25  # total traction per unit length on the right edge

# Cook's membrane: (0,0), (48,44), (48,60), (0,44), counter-clockwise.
patch = (
    jx.primitives.quadrilateral([[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]])
    .elevate(DEG)
    .refine(REFINE)
)
V = jx.FunctionSpace(patch, vec=2)
print(V.summary())

problem = jx.Hyperelasticity(
    V,
    dirichlet=[jx.DirichletBC([0.0, 0.0], where="left")],
    neumann=[jx.Neumann(lambda x, n, p: jnp.array([0.0, SHEAR]), where="right")],
)
params = {"E": E, "nu": NU}

results = {}
for method, kwargs in (
    ("galerkin", dict(tol=1e-10, max_steps=30)),
    ("energy", dict(tol=1e-12, max_steps=6000)),
):
    t0 = time.perf_counter()
    sol = jx.solve(problem, method=method, params=params, **kwargs)
    elapsed = time.perf_counter() - t0
    tip = sol.probe(np.array([[48.0, 60.0]]))[0]
    results[method] = sol
    detail = (
        f"{sol.stats.get('iterations')} Newton steps"
        if method == "galerkin"
        else f"{sol.stats.get('steps')} L-BFGS steps"
    )
    print(
        f"  {method:9s} tip displacement ({float(tip[0]): .6f}, {float(tip[1]): .6f})"
        f"   {detail}, {elapsed:.1f}s"
    )

a, b = results["galerkin"].u, results["energy"].u
print(
    f"\n  max |u_galerkin - u_energy| = {float(jnp.abs(a - b).max()):.3e} "
    f"(relative {float(jnp.abs(a - b).max() / jnp.abs(a).max()):.3e})"
)
print("  Newton residual history:",
      [f"{v:.2e}" for v in jx.history_list(results["galerkin"].stats)])

# The quadratic tail of that history is the signature of an exact tangent: the
# Hessian of the energy is differentiated, never approximated.
tip = results["galerkin"].probe(np.array([[48.0, 60.0]]))[0]
print(f"\n  tip vertical displacement {float(tip[1]):.6f}")
print("  (linear elasticity at the same load and mesh, for reference:)")
lin = jx.solve(
    jx.LinearElasticity(
        V,
        plane="strain",
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="left")],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([0.0, SHEAR]), where="right")],
    ),
    params=params,
)
tip_lin = lin.probe(np.array([[48.0, 60.0]]))[0]
print(f"  linear tip vertical displacement {float(tip_lin[1]):.6f}")

results["galerkin"].to_vtk("cooks_membrane", n=4)
print("  wrote cooks_membrane.vtu")
