"""Kirchhoff-Love plate: a fourth-order problem solved with C^1 splines.

Plate bending depends on the *second* derivatives of the deflection, so its
variational form needs a discretization in H^2 -- functions with continuous
first derivatives across element boundaries. Standard C^0 finite elements do
not provide that, which is why plate elements are a subject in their own right
(discrete Kirchhoff triangles, mixed formulations, C^0 interior penalty).

A spline of degree p at maximal smoothness is C^(p-1) *inside a patch* by
construction, so for p >= 2 the requirement is met with no special element at
all. Setting ``needs_hessian`` on the problem is the whole of it: the interior
Gauss set is then evaluated at order 2 and the kernel additionally receives
``hess_u``. The Galerkin tangent is still the autodiff Hessian of the energy,
so no other part of the pipeline changes.

Two closed-form solutions are reproduced:

1. simply supported under a sinusoidal load (Navier), where the L2 rate is
   ``min(p+1, 2p-2)``;
2. clamped under a uniform load, against Timoshenko's tabulated centre
   deflection ``w_max = 0.0012653 q a^4 / D``.

A clamped edge fixes the slope as well as the deflection, which on a spline
means constraining *two* layers of control points -- that is what ``ClampedBC``
does.

Multipatch C^1 coupling is out of scope: patches meet with C^0 continuity, so
a patch interface acts as a hinge. Both cases here are single-patch.
"""

import numpy as np
import jax.numpy as jnp

import jaxiga as jx

E, NU, THICK, Q0 = 210e3, 0.3, 0.01, 1.0
PARAMS = {"E": E, "nu": NU, "t": THICK}
D = E * THICK**3 / (12 * (1 - NU**2))  # bending stiffness

ALL = lambda x: np.ones(len(x), bool)
SINE_LOAD = lambda x, p: Q0 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
NAVIER = lambda x: Q0 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) / (4 * jnp.pi**4 * D)

print(f"bending stiffness D = {D:.6g} N mm")
print(f"Navier centre deflection = {Q0 / (4 * np.pi**4 * D):.8e}\n")

# -- 1. simply supported, sinusoidal load ----------------------------------
# A simply supported edge is just w = 0: the zero-moment condition is the
# natural boundary condition of the bending energy, so it needs no statement.
print("simply supported square plate, sinusoidal load")
print("   p   mesh      dofs    rel. L2 error    rate    theory")
for degree in (2, 3):
    errs, hs = [], []
    for n in (3, 4, 5):
        V = jx.FunctionSpace(
            jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0).elevate(degree).refine(n)
        )
        problem = jx.KirchhoffPlate(
            V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=SINE_LOAD
        )
        sol = jx.solve(problem, params=PARAMS)
        errs.append(jx.errornorm(sol, NAVIER, "L2"))
        hs.append(1.0 / 2**n)
        rate = (
            np.log(errs[-2] / errs[-1]) / np.log(hs[-2] / hs[-1]) if len(errs) > 1 else np.nan
        )
        print(
            f"  {degree}  {2**n:3d}x{2**n:<3d} {V.n_dofs:7d}   {errs[-1]:.6e}"
            f"   {rate:6.3f}   {min(degree + 1, 2 * degree - 2)}"
        )

# -- 2. clamped, uniform load ----------------------------------------------
# A clamped edge fixes the normal slope too, so the second control-point layer
# is constrained as well; that is exactly what ClampedBC resolves to.
print("\nclamped square plate, uniform load")
print(f"  Timoshenko w_max = 0.0012653 q a^4 / D = {0.0012653 * Q0 / D:.8e}")
print("   mesh      dofs    w(0.5, 0.5)      ratio")
for n in (3, 4, 5):
    V = jx.FunctionSpace(jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0).elevate(3).refine(n))
    problem = jx.KirchhoffPlate(
        V,
        dirichlet=[jx.ClampedBC(where=side) for side in ("left", "right", "bottom", "top")],
        source=lambda x, p: Q0,
    )
    sol = jx.solve(problem, params=PARAMS)
    w = float(sol.probe(np.array([[0.5, 0.5]]))[0, 0])
    print(f"  {2**n:3d}x{2**n:<3d} {V.n_dofs:7d}   {w:.8e}   {w / (0.0012653 * Q0 / D):.5f}")

# Bending moments come from the same energy by differentiation, like every
# other derived field in the package.
moments = sol.field("moments", ps=jx.pointset.grid(V, 3))
print(f"\n  peak |M_xx| = {float(jnp.abs(moments[..., 0]).max()):.6e} N mm/mm")
print(f"  peak |M_xy| = {float(jnp.abs(moments[..., 2]).max()):.6e} N mm/mm")

sol.to_vtk("plate_clamped", n=5)
print("  wrote plate_clamped.vtu")
