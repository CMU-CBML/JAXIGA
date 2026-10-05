"""Poisson on the unit square by the Galerkin method.

    -div(a0 grad u) = f   on (0,1)^2
                  u = 0   on the boundary

with the manufactured solution u = sin(2 pi x) sin(2 pi y).
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx

DEG = 3
REFINEMENTS = 5

exact = lambda x: jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
source = lambda x, p: 8 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])

patch = jx.primitives.quadrilateral(
    jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
).elevate(DEG).refine(REFINEMENTS)

V = jx.FunctionSpace(patch)
print(V.summary())

problem = jx.Poisson(
    V,
    dirichlet=[jx.DirichletBC(0.0, where=lambda x: jnp.full(x.shape[0], True))],
    source=source,
)

sol = jx.solve(problem, params={"a0": 1.0})

print("relative L2 error:", jx.errornorm(sol, exact, "L2"))
sol.to_vtk("poisson2d", n=6)
print("wrote poisson2d.vtu")


if __name__ == "__main__":
    # convergence study
    print("\n refine    n_dofs      rel L2 error   rate")
    prev = None
    for r in range(2, 6):
        Vr = jx.FunctionSpace(
            jx.primitives.quadrilateral(
                jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
            ).elevate(DEG).refine(r)
        )
        pr = jx.Poisson(
            Vr,
            dirichlet=[jx.DirichletBC(0.0, where=lambda x: jnp.full(x.shape[0], True))],
            source=source,
        )
        err = jx.errornorm(jx.solve(pr, params={"a0": 1.0}), exact, "L2")
        rate = "" if prev is None else f"{np.log2(prev / err):6.2f}"
        print(f" {r:^6d} {Vr.n_dofs:>8d}   {err:.6e}  {rate}")
        prev = err
    print(f"\n expected asymptotic rate for degree {DEG}: {DEG + 1}")
