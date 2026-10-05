"""1D Poisson: -u'' = k^2 pi^2 sin(k pi x) on (0, 8), u = 0 at both ends.
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx

DEG = 2
REFINEMENTS = 6
K = 1.0
LENGTH = 8.0

exact = lambda x: jnp.sin(K * jnp.pi * x[0])
source = lambda x, p: (K * jnp.pi) ** 2 * jnp.sin(K * jnp.pi * x[0])

V = jx.FunctionSpace(jx.primitives.interval(0.0, LENGTH).elevate(DEG).refine(REFINEMENTS))
print(V.summary())

problem = jx.Poisson(
    V,
    dirichlet=[jx.DirichletBC(0.0, where="left"), jx.DirichletBC(0.0, where="right")],
    source=source,
)

sol = jx.solve(problem, params={"a0": 1.0})
print("relative L2 error:", jx.errornorm(sol, exact, "L2"))


if __name__ == "__main__":
    # The solution has four full periods over (0, 8), so meshes coarser than
    # about 16 elements are pre-asymptotic and their rates are meaningless.
    print("\n refine   n_dofs     rel L2 error   rate")
    prev = None
    for r in range(4, 9):
        Vr = jx.FunctionSpace(jx.primitives.interval(0.0, LENGTH).elevate(DEG).refine(r))
        pr = jx.Poisson(
            Vr,
            dirichlet=[jx.DirichletBC(0.0, where="left"), jx.DirichletBC(0.0, where="right")],
            source=source,
        )
        err = jx.errornorm(jx.solve(pr, params={"a0": 1.0}), exact, "L2")
        rate = "" if prev is None else f"{np.log2(prev / err):6.2f}"
        print(f" {r:^6d} {Vr.n_dofs:>8d}   {err:.6e}  {rate}")
        prev = err
    print(f"\n expected asymptotic rate for degree {DEG}: {DEG + 1}")
