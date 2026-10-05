"""Shape derivatives: differentiating a solve with respect to the geometry.

Control points are pytree leaves of the FunctionSpace, so geometry gradients
need no special machinery -- `jax.grad` reaches them through the same adjoint
path used for material parameters. Each gradient costs one forward solve plus
one adjoint solve, whatever the number of design variables.

The `where` predicates of a Problem are resolved when it is constructed, so
build the problem once outside the differentiated function and rebind the
geometry with `with_space` inside it.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.solvers.linear import SOLVE_COUNTER, reset_solve_counter

DEG = 2
REFINEMENTS = 2

patch = jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0).elevate(DEG).refine(REFINEMENTS)
V = jx.FunctionSpace(patch)

base_problem = jx.Poisson(
    V,
    dirichlet=[jx.DirichletBC(0.0, where=lambda x: jnp.full(x.shape[0], True))],
    source=lambda x, p: 1.0,
)


def compliance(cpts):
    """A scalar objective depending on the geometry through the solve."""
    problem = base_problem.with_space(dataclasses.replace(V, cpts=cpts))
    sol = jx.solve(problem, params={"a0": 1.0})
    return jnp.sum(sol.u**2)


print(f"space: {V.summary()}")
print(f"compliance at the reference geometry: {float(compliance(V.cpts)):.10f}")

reset_solve_counter()
grad_cpts = jax.grad(compliance)(V.cpts)
print(f"solves used by one gradient: {SOLVE_COUNTER}")
print(f"sensitivity array shape: {grad_cpts.shape}")

# spot-check three entries against central differences
flat = V.cpts.reshape(-1)
flat_compliance = lambda f: compliance(f.reshape(V.cpts.shape))
h = 1e-6
print("\n  index      adjoint         finite difference    rel. error")
for i in (5, 11, 17):
    fd = float((flat_compliance(flat.at[i].add(h)) - flat_compliance(flat.at[i].add(-h))) / (2 * h))
    ad = float(np.asarray(grad_cpts).reshape(-1)[i])
    print(f"  {i:^5d}  {ad: .10e}  {fd: .10e}   {abs(ad - fd) / max(abs(fd), 1e-30):.2e}")

# The same machinery differentiates material parameters.
print("\nGradient with respect to the diffusion coefficient:")
d_a0 = jax.grad(
    lambda a0: jnp.sum(jx.solve(base_problem, params={"a0": a0}).u ** 2)
)(1.0)
fd_a0 = (
    jnp.sum(jx.solve(base_problem, params={"a0": 1.0 + h}).u ** 2)
    - jnp.sum(jx.solve(base_problem, params={"a0": 1.0 - h}).u ** 2)
) / (2 * h)
print(f"  adjoint {float(d_a0): .10e}   finite difference {float(fd_a0): .10e}")


# A coarse design net driving a fine analysis mesh: the refinement operator is
# linear, so gradients flow through it to the coarse control points.
coarse = jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0).elevate(DEG)
fine = coarse.refine(REFINEMENTS)
P = coarse.refinement_operator(fine)
print(f"\nrefinement operator maps {P.shape[1]} coarse -> {P.shape[0]} fine control points")

def coarse_compliance(coarse_cpts):
    w = jnp.asarray(coarse.weights)
    homogeneous = jnp.concatenate([coarse_cpts * w[:, None], w[:, None]], axis=1)
    fine_h = jnp.asarray(P) @ homogeneous
    return compliance(fine_h[:, :-1] / fine_h[:, -1:])

g_coarse = jax.grad(coarse_compliance)(coarse.ctrl_pts)
print(f"coarse-net sensitivity shape: {g_coarse.shape}")
