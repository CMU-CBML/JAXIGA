"""Topology optimization of a cantilever by SIMP compliance minimization.

The differentiable-solve capability in one page: the design variable is a
per-element density, the objective is the compliance of the resulting
structure, and the sensitivity comes from ``jax.grad`` straight through the
linear solve. No adjoint problem is derived by hand, and no sensitivity formula
is written down -- the classical ``dc/drho = -u_e^T dK_e/drho u_e`` falls out of
the same implicit-differentiation rule that serves every other parameter.

Density reaches the physics as a :class:`PointField`: a value per quadrature
point that the pointwise kernel receives alongside ``x``. SIMP interpolation
``E(rho) = E_min + rho^p (E_0 - E_min)`` with ``p = 3`` then penalizes
intermediate densities into a nearly black-and-white layout.

The design is updated by the optimality criteria (OC) rule rather than by a
generic optimizer, and that choice is worth a note. Every compliance
sensitivity here is negative -- more material anywhere is always stiffer -- so
the design is driven entirely by their *relative* magnitudes. Adam normalizes
each coordinate to roughly +-lr, which throws that information away: the update
becomes nearly uniform, and the volume projection then subtracts it off almost
exactly, leaving the layout stuck. OC uses the magnitudes directly and enforces
the volume exactly by bisecting its Lagrange multiplier, which is why it is
still the standard update for this problem.
"""

import numpy as np
import jax
import jax.numpy as jnp

import jaxiga as jx
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

L, H, NEX, NEY = 2.0, 1.0, 48, 24
E0, EMIN, NU, PEN = 1.0, 1e-9, 0.3, 3.0
VOLFRAC, STEPS, RADIUS, MOVE, ETA = 0.45, 40, 1.6, 0.2, 0.5

patch = jx.primitives.rectangle(0.0, 0.0, L, H).elevate(1)
patch = patch.insert_knots(np.linspace(0, 1, NEX + 1)[1:-1], np.linspace(0, 1, NEY + 1)[1:-1])
V = jx.FunctionSpace(patch, vec=2)
load = lambda x, n, p: jnp.array([0.0, -1.0]) * jnp.exp(-((x[1] - H / 2) ** 2) / 0.004)
problem = jx.LinearElasticity(
    V, plane="stress",
    dirichlet=[jx.DirichletBC([0.0, 0.0], where="left")],
    neumann=[jx.Neumann(load, where="right")],
)
basis = evaluate(V, P.gauss(V))
bnd = evaluate(V, P.boundary_gauss(V, "right"))
n_elems, n_q = basis.n_elems, basis.n_q

# Density filter: a fixed sparse average over element centres, which removes
# the checkerboard mode and gives the layout a length scale.
centres = (V.elem_vertex[:, :2] + V.elem_vertex[:, 2:]) / 2 * np.array([L, H])
d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
W = np.maximum(0.0, RADIUS * min(L / NEX, H / NEY) - d)
W = jnp.asarray(W / W.sum(axis=1, keepdims=True))


def compliance(rho):
    """External work done by the traction -- the objective, and the energy norm."""
    stiffness = EMIN + (W @ rho) ** PEN * (E0 - EMIN)
    params = {"E": jx.PointField(jnp.broadcast_to(stiffness[:, None], (n_elems, n_q))),
              "nu": NU}
    sol = jx.solve(problem, params=params, linear=jx.LinearOptions(method="scipy"))
    u = sol.at(basis=bnd)
    t = jax.vmap(jax.vmap(load, in_axes=(0, 0, None)), in_axes=(0, 0, None))(
        bnd.x, bnd.normal, params)
    return jnp.sum(bnd.w * jnp.sum(t * u, axis=-1))


def oc_update(rho, dc):
    """Optimality criteria: exact volume, bounded move, monotone in practice."""
    lo, hi = 1e-9, 1e9
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        new = jnp.clip(rho * (-dc / mid) ** ETA, rho - MOVE, rho + MOVE)
        new = jnp.clip(new, 0.0, 1.0)
        if float(jnp.mean(new)) > VOLFRAC:
            lo = mid
        else:
            hi = mid
    return new


rho = jnp.full(n_elems, VOLFRAC)
value_and_grad = jax.jit(jax.value_and_grad(compliance))

print(f"{V.summary()}\n{n_elems} design variables, volume fraction {VOLFRAC}")
print("  step   compliance     mean rho   grey fraction   max change")
history, change = [], np.nan
for step in range(STEPS):
    c, g = value_and_grad(rho)
    history.append(float(c))
    grey = float(jnp.mean((rho > 0.05) & (rho < 0.95)))
    if step % 5 == 0 or step == STEPS - 1:
        print(f"  {step:4d}   {float(c):.6e}   {float(jnp.mean(rho)):.4f}     "
              f"{grey:.3f}          {change:.4f}")
    new = oc_update(rho, g)
    change = float(jnp.abs(new - rho).max())
    rho = new

print(f"\ncompliance {history[0]:.4e} -> {history[-1]:.4e} "
      f"({history[0] / history[-1]:.2f}x stiffer at the same volume)")
tail = history[len(history) // 4:]
print(f"monotone over the last {len(tail)} steps: "
      f"{all(b <= a + 1e-9 for a, b in zip(tail, tail[1:]))}")
print(f"final grey fraction {float(jnp.mean((rho > 0.05) & (rho < 0.95))):.3f} "
      f"(0 would be fully black-and-white)")
np.savetxt("simp_density.csv", np.column_stack([centres, np.asarray(rho)]),
           delimiter=",", header="x,y,rho", comments="")
print("wrote simp_density.csv")
