"""3D Poisson on a cube, solved by Galerkin and by energy minimization.

Nothing in the library is dimension-specific: the same `Poisson` class, the
same `solve`, the same error norms. Only the geometry changes.

Replaces Poisson3D_Cube_IGA.py and Poisson3D_Cube_EM.py, which were separate
scripts for what is now one argument.
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx

DEG = 2
REFINEMENTS = 3

ALL = lambda x: jnp.full(x.shape[0], True)
exact = lambda x: jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) * jnp.sin(jnp.pi * x[2])
source = lambda x, p: (
    3 * jnp.pi**2 * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) * jnp.sin(jnp.pi * x[2])
)


def build(deg, refine):
    patch = jx.primitives.cuboid([0, 0, 0], [1, 1, 1]).elevate(deg).refine(refine)
    V = jx.FunctionSpace(patch)
    return jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)], source=source)


problem = build(DEG, REFINEMENTS)
print(problem.space.summary())

sol = jx.solve(problem, params={"a0": 1.0})
print("galerkin relative L2 error:", jx.errornorm(sol, exact, "L2"))

sol_em = jx.solve(problem, params={"a0": 1.0}, method="energy", max_steps=4000)
diff = float(jnp.max(jnp.abs(sol_em.u - sol.u)))
print(f"energy method agrees to {diff:.3e} "
      f"({diff / float(jnp.max(jnp.abs(sol.u))):.3e} relative)")

sol.to_vtk("poisson3d_cube", n=3)
print("wrote poisson3d_cube.vtu")


if __name__ == "__main__":
    print("\n refine   n_dofs    rel L2 error   rate")
    prev = None
    for r in (1, 2, 3):
        pr = build(DEG, r)
        err = jx.errornorm(jx.solve(pr, params={"a0": 1.0}), exact, "L2")
        rate = "" if prev is None else f"{np.log2(prev / err):6.2f}"
        print(f" {r:^6d} {pr.space.n_dofs:>8d}   {err:.6e}  {rate}")
        prev = err
    print(f"\n expected asymptotic rate for degree {DEG}: {DEG + 1}")
