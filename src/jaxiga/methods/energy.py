"""Energy minimization (Deep-Ritz style).

The whole driver is: minimize the discrete potential energy over the free dofs.
Because the Galerkin residual *is* the gradient of this objective, the two
methods have the same stationary point, and agreeing to solver tolerance is a
cross-validation of both.

The method comes into its own on nonlinear problems, where only the strain
energy has to be written -- no tangent, no residual.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxiga.methods._common import build_context, residual, total_energy
from jaxiga.solvers.implicit import implicit_correction
from jaxiga.solvers.linear import LinearOptions, sparse_solve
from jaxiga.solvers.optimize import minimize


def solve_energy(
    problem,
    params=None,
    initial_guess=None,
    quadrature=None,
    optimizer: str = "lbfgs",
    max_steps: int = 2000,
    tol: float = 1e-10,
    linear: LinearOptions | None = None,
    implicit: bool = True,
    **opts,
):
    """Minimize the total potential energy over the free dofs.

    Differentiation does not go through the optimizer. At the minimiser the
    gradient of the energy vanishes, which is exactly the Galerkin residual, so
    the converged point is a root that the implicit function theorem
    differentiates with one adjoint solve against the energy Hessian --
    independent of how many L-BFGS steps were taken, and identical to the
    derivative the Galerkin path produces at the same point.
    """
    from jaxiga.methods.galerkin import assemble_csr_values

    linear = linear or LinearOptions()
    ctx = build_context(problem, params, quadrature)
    u0 = ctx.dofmap.zeros() if initial_guess is None else initial_guess

    loss = lambda uf: total_energy(uf, params, ctx)  # noqa: E731
    try:
        loss(u0)
    except NotImplementedError as exc:
        raise TypeError(
            f"{type(problem).__name__} defines no energy density, so it cannot be "
            f"solved by minimization; define energy() for the variational methods, "
            "or use method='collocation' with flux/mass"
        ) from exc

    result = minimize(loss, u0, optimizer=optimizer, max_steps=max_steps, tol=tol, **opts)

    u_free = result.x
    if implicit:
        u_free = implicit_correction(
            u_free,
            lambda uu: residual(uu, params, ctx),
            lambda uu, rhs: sparse_solve(
                linear, ctx.assembly, assemble_csr_values(ctx, params, uu), rhs
            ),
            converged=result.converged,
        )

    value, grad = jax.value_and_grad(loss)(u_free)
    grad_norm = jnp.linalg.norm(grad)

    stats = {
        "method": "energy",
        "optimizer": optimizer,
        "steps": result.steps,
        "energy": value,
        "grad_norm": grad_norm,
        "residual": grad_norm,
        "converged": jnp.isfinite(value) & jnp.isfinite(grad_norm) & (grad_norm <= tol),
        "finite": jnp.isfinite(value) & jnp.isfinite(grad_norm) & jnp.all(jnp.isfinite(u_free)),
    }
    _warn_if_not_converged(stats, tol)
    return ctx.dofmap.lift(u_free), ctx, stats


def _warn_if_not_converged(stats, tol):
    """Warn outside a trace; silent when the values are tracers."""
    try:
        converged = bool(stats["converged"])
        gnorm = float(stats["grad_norm"])
        steps = int(stats["steps"])
    except (TypeError, ValueError):
        return
    if converged:
        return

    import warnings

    warnings.warn(
        f"energy minimization did not converge: final |grad| = {gnorm:.3e}, "
        f"tol = {tol:g}, after {steps} steps; do not use implicit sensitivities",
        RuntimeWarning,
        stacklevel=3,
    )
