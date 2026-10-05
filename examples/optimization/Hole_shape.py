"""Shape optimization of a hole, driven by the geometry's own control points.

Control points are pytree *leaves* of the FunctionSpace, so the geometry is
differentiable on the same footing as any material parameter: one adjoint solve
gives the sensitivity of a stress functional with respect to every control
point at once, whatever their number.

The design is a square plate in uniaxial tension with a central hole. A
circular hole concentrates stress by a factor of three (Kirsch); an elliptical
hole elongated *along* the load direction does better, since for semi-axes
``a`` across the load and ``b`` along it the concentration factor is
``1 + 2a/b``. The optimizer is told none of that -- only to reduce a smooth
surrogate for the peak von Mises stress while holding the plate area fixed.

Peak stress is not differentiable, so the objective is the p-norm

    (mean over Omega of sigma_vm^k)^(1/k),   k = 8,

which is smooth and approaches the maximum from below. The area constraint uses
the *computed* area -- the quadrature weights of the deformed geometry -- so it
too flows through the control points rather than through a hand-written
formula. Holding the plate area fixed holds the hole area fixed, since the
outer boundary does not move.

``where`` predicates resolve against the geometry a Problem is built on, so the
pinned dofs are chosen once, on the circular hole, and stay the same design
after design -- which is exactly what ``with_space`` exists for.
"""

import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import optax

import jaxiga as jx
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

RADIUS, SIDE, DEG, REFINE = 1.0, 4.0, 2, 3
E, NU, TENSION = 1.0e5, 0.3, 10.0
KNORM, STEPS, PENALTY, TOL = 8.0, 40, 1.0e3, 1e-9

patches = [p.elevate(DEG).refine(REFINE) for p in jx.primitives.plate_with_hole(RADIUS, SIDE)]
V = jx.FunctionSpace(patches, vec=2)
print(V.summary())

on_hole = lambda x: np.abs(np.hypot(x[:, 0], x[:, 1]) - RADIUS) < 1e-6
vertical_edge = lambda x: np.abs(np.abs(x[:, 0]) - SIDE) < TOL

problem = jx.LinearElasticity(
    V,
    plane="stress",
    # Pure traction loading, so four pins remove the rigid-body modes.
    dirichlet=[
        jx.DirichletBC(0.0, where=lambda x: on_hole(x) & (np.abs(x[:, 1]) < TOL), component=1),
        jx.DirichletBC(0.0, where=lambda x: on_hole(x) & (np.abs(x[:, 0]) < TOL), component=0),
    ],
    # t = sigma n on the two vertical edges is uniaxial tension.
    neumann=[jx.Neumann(lambda x, n, p: TENSION * n, where=vertical_edge)],
)
params = {"E": E, "nu": NU}
ps = P.gauss(V)

# Design variables: one radial offset per hole control point. The hole is
# centred at the origin, so moving those points along their own radius deforms
# the hole while the outer boundary stays put.
cpts0 = np.asarray(V.cpts)
hole_dofs = np.asarray(V.boundary("hole").dofs)
radial = cpts0[hole_dofs] / np.linalg.norm(cpts0[hole_dofs], axis=1, keepdims=True)


def geometry(design):
    moved = jnp.asarray(cpts0)[hole_dofs] + design[:, None] * jnp.asarray(radial)
    return jnp.asarray(cpts0).at[hole_dofs].set(moved)


def analyse(design):
    """``(p-norm stress, max stress, area)``, all traced."""
    space = dataclasses.replace(V, cpts=geometry(design))
    sol = jx.solve(problem.with_space(space), params=params)
    basis = evaluate(space, ps)
    vm = sol.field("von_mises", basis=basis)
    area = jnp.sum(basis.w)
    pnorm = (jnp.sum(basis.w * vm**KNORM) / area) ** (1.0 / KNORM)
    return pnorm, jnp.max(vm), area


AREA0 = float(analyse(jnp.zeros(len(hole_dofs)))[2])


def objective(design):
    pnorm, _, area = analyse(design)
    return pnorm / TENSION + PENALTY * ((area - AREA0) / AREA0) ** 2


design = jnp.zeros(len(hole_dofs))
_, peak0, _ = analyse(design)
peak0 = float(peak0)
print(f"\ncircular hole: peak von Mises / applied = {peak0 / TENSION:.4f}")
print(f"  (Kirsch gives 3 for an infinite plate; this one is finite, side/radius = {SIDE})")
print(f"{len(hole_dofs)} design variables, plate area held at {AREA0:.6f}")

opt = optax.adam(0.02)
state = opt.init(design)
value_and_grad = jax.jit(jax.value_and_grad(objective))
report = jax.jit(analyse)

print("\n  step   objective     peak/applied   area error")
for step in range(STEPS):
    val, g = value_and_grad(design)
    if step % 5 == 0 or step == STEPS - 1:
        _, peak, area = report(design)
        print(f"  {step:4d}   {float(val):.6f}     {float(peak) / TENSION:.4f}"
              f"         {abs(float(area) - AREA0) / AREA0:.2e}")
    updates, state = opt.update(g, state, design)
    design = optax.apply_updates(design, updates)

_, peak1, area1 = report(design)
peak1, area1 = float(peak1), float(area1)
print(f"\npeak von Mises / applied: {peak0 / TENSION:.4f} -> {peak1 / TENSION:.4f} "
      f"({100 * (1 - peak1 / peak0):.1f}% lower), area changed by "
      f"{abs(area1 - AREA0) / AREA0:.1e}")

moved = np.asarray(geometry(design))[hole_dofs]
r = np.linalg.norm(moved, axis=1)
ang = np.degrees(np.arctan2(moved[:, 1], moved[:, 0])) % 180.0
along = float(r[np.argmin(np.minimum(ang, 180 - ang))])
across = float(r[np.argmin(np.abs(ang - 90))])
print(f"hole radius along the load {along:.4f}, across it {across:.4f} "
      f"(ratio {along / across:.3f})")
print("an ellipse elongated along the load is the known optimum, so a ratio > 1 "
      "means the optimizer found it unaided")
np.savetxt("hole_shape.csv", np.column_stack([moved, r]), delimiter=",",
           header="x,y,r", comments="")
print("wrote hole_shape.csv")
