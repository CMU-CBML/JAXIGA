"""Poisson by collocation at Greville abscissae.

Collocation enforces the strong form -div(a0 grad u) - f = 0 at the Greville
points. The strong residual is derived from the same energy density the other
methods use, by automatic differentiation.

Two things to notice in the output:

* collocation converges at a *lower* rate than Galerkin -- p-1 for odd degree,
  p for even -- which is the classical Greville-collocation result, not a bug;
* on a multipatch space, flux-continuity equations at interface points are
  added automatically, making the system overdetermined, and it is solved by
  Gauss-Newton least squares.
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx

ALL = lambda x: jnp.full(x.shape[0], True)
exact = lambda x: jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])
source = lambda x, p: 8 * jnp.pi**2 * jnp.sin(2 * jnp.pi * x[0]) * jnp.sin(2 * jnp.pi * x[1])


def build(deg, refine, patches=None):
    if patches is None:
        patches = jx.primitives.rectangle(0, 0, 1, 1).elevate(deg).refine(refine)
    V = jx.FunctionSpace(patches)
    return jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=source)


problem = build(3, 3)
sol = jx.solve(problem, params={"a0": 1.0}, method="collocation")
print(problem.space.summary())
print("collocation stats:", sol.stats)
print("relative L2 error:", jx.errornorm(sol, exact, "L2"))


if __name__ == "__main__":
    print("\n--- convergence, single patch ---")
    print(" deg  refine   n_free   collocation L2   rate    Galerkin L2")
    for deg in (3, 4):
        prev = None
        for r in (2, 3, 4):
            pr = build(deg, r)
            c = jx.solve(pr, params={"a0": 1.0}, method="collocation")
            g = jx.solve(pr, params={"a0": 1.0})
            ec = jx.errornorm(c, exact, "L2")
            eg = jx.errornorm(g, exact, "L2")
            rate = "" if prev is None else f"{np.log2(prev / ec):5.2f}"
            print(f" {deg:^4d} {r:^7d} {c.stats['n_free']:>7d}   {ec:.4e}  {rate:>6s}   {eg:.4e}")
            prev = ec
        expected = deg - 1 if deg % 2 else deg
        print(f"      expected collocation rate for degree {deg}: {expected}\n")

    print("--- two patches: interface flux continuity is added automatically ---")
    for r in (1, 2, 3):
        pr = build(3, r, patches=[
            jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(r),
            jx.primitives.rectangle(1, 0, 2, 1).elevate(3).refine(r),
        ])
        two_exact = lambda x: jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
        pr = jx.Poisson(
            pr.space,
            dirichlet=[jx.DirichletBC(0.0, where=ALL)],
            source=lambda x, p: 2 * jnp.pi**2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
        )
        c = jx.solve(pr, params={"a0": 1.0}, method="collocation")
        print(f" refine={r}  interfaces={len(pr.space.topology.interfaces)}  "
              f"equations={c.stats['n_equations']} > free={c.stats['n_free']}  "
              f"L2={jx.errornorm(c, two_exact, 'L2'):.4e}")
