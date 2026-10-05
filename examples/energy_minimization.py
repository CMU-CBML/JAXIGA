"""Energy minimization, and why it agrees with Galerkin.

The Galerkin residual is the gradient of the discrete potential energy, so both
methods have the same stationary point. Solving a linear problem both ways is
therefore a built-in cross-check, and the difference below is at solver
tolerance rather than discretization error.

Energy minimization earns its keep on nonlinear problems: for the neo-Hookean
solid at the bottom, only the strain energy is written -- no residual, no
tangent.
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx

# -- linear elasticity, solved twice --------------------------------------

V = jx.FunctionSpace(jx.primitives.quarter_annulus(1.0, 4.0).elevate(3).refine(2), vec=2)

problem = jx.LinearElasticity(
    V,
    plane="stress",
    dirichlet=[jx.DirichletBC(0.0, where="sym_y", component=1),
               jx.DirichletBC(0.0, where="sym_x", component=0)],
    neumann=[jx.Neumann(lambda x, n, p: -10.0 * n, where="inner")],
)
params = {"E": 1e5, "nu": 0.3}

sol = jx.solve(problem, params=params)
sol_em = jx.solve(problem, params=params, method="energy", max_steps=3000)

print(V.summary())
print(f"galerkin : {sol.stats}")
print(f"energy   : steps={sol_em.stats['steps']} "
      f"energy={sol_em.stats['energy']:.12e} |grad|={sol_em.stats['grad_norm']:.3e}")

diff = float(jnp.max(jnp.abs(sol_em.u - sol.u)))
scale = float(jnp.max(jnp.abs(sol.u)))
print(f"\n||u_energy - u_galerkin||_inf = {diff:.3e}")
print(f"relative to ||u||_inf         = {diff / scale:.3e}   (same discrete minimum)")


# -- a nonlinear problem: only the strain energy is written ----------------


class NeoHookean(jx.Problem):
    """Compressible neo-Hookean solid."""

    def energy(self, grad_u, u, x, p):
        F = jnp.eye(2) + grad_u
        J = jnp.linalg.det(F)
        I1 = jnp.trace(F.T @ F)
        return p["mu"] / 2 * (I1 - 2) - p["mu"] * jnp.log(J) + p["lam"] / 2 * jnp.log(J) ** 2


Vn = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2), vec=2)
stretch = NeoHookean(
    Vn,
    dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
               jx.DirichletBC([0.0, 0.25], where="top")],
)
p_nl = {"mu": 384.6, "lam": 576.9}

print("\n--- neo-Hookean, 25% stretch ---")
nl_em = jx.solve(stretch, params=p_nl, method="energy", max_steps=5000)
nl_gal = jx.solve(stretch, params=p_nl)  # Newton with the autodiff-consistent tangent

print(f"energy   : steps={nl_em.stats['steps']} energy={nl_em.stats['energy']:.10e}")
print(f"newton   : iterations={nl_gal.stats['iterations']} "
      f"residual={nl_gal.stats['residual']:.3e}")
print("newton residual history:",
      [f"{h:.2e}" for h in jx.history_list(nl_gal.stats)])

d = float(jnp.max(jnp.abs(nl_em.u - nl_gal.u)))
print(f"\n||u_energy - u_newton||_inf = {d:.3e} "
      f"(relative {d / float(jnp.max(jnp.abs(nl_gal.u))):.3e})")
