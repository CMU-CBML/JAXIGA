"""Implicit differentiation of a converged solution.

A solver produces ``u*`` satisfying ``r(u*, theta) = 0``. The implicit function
theorem gives its sensitivity without touching the iteration that found it:

    du*/dtheta = -(dr/du)^{-1} (dr/dtheta).

:func:`implicit_correction` attaches exactly that derivative to an already
computed root, by evaluating one Newton correction from a *detached* copy of it:

    u* = ubar - A(ubar)^{-1} r(ubar),      ubar = stop_gradient(u*).

This is a first-order rule only; repeated differentiation of the detached
root does not give general higher sensitivities. At convergence ``r(ubar)``
is zero to solver tolerance, so the returned value
equals the input numerically; but differentiating the right-hand side gives
``-A^{-1} dr/dtheta``, which is the implicit-function-theorem result, and gives
nothing at all for the iterates, which carry no gradient. The backward pass is
therefore one adjoint solve regardless of how many iterations the forward solve
took, and solver iterations are never unrolled.

Why this formulation rather than a ``custom_vjp`` over the solve
---------------------------------------------------------------
A ``custom_vjp`` would have to be told, explicitly and exhaustively, which
values the residual depends on. Here the residual is an ordinary traced
function, so JAX finds them: material parameters in ``params``, but equally the
control points and weights closed over inside the quadrature data, which is what
makes shape derivatives work on nonlinear problems without special handling.
The cost is one extra *forward* solve, which buys that generality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def implicit_correction(u_star, residual_fn, tangent_solve, *, converged=None):
    """Attach first-order implicit sensitivities at a converged regular root.

    Higher derivatives of this detached-root expression are not supported.
    Accuracy requires converged primal and tangent/adjoint solves. If supplied,
    ``converged`` gates the correction: a failed iteration returns its detached
    last state, whose sensitivities must not be used.

    Parameters
    ----------
    u_star : array
        The converged root, as produced by any iteration whatsoever.
    residual_fn : callable
        ``r(u)``, closing over everything the derivative should reach. Must be
        traceable and differentiable in ``u`` and in those closed-over values.
    tangent_solve : callable
        ``(u, rhs) -> x`` solving ``(dr/du)(u) x = rhs``. Usually the same
        linear solve the forward iteration used, and itself carrying the
        adjoint rule that makes the backward pass one solve.

    Returns
    -------
    array
        Numerically ``u_star``, with the correct derivative attached.
    """
    u_bar = jax.lax.stop_gradient(u_star)

    def corrected(u):
        return u - tangent_solve(u, residual_fn(u))

    if converged is None:
        return corrected(u_bar)
    # Avoid staging a conditional for an eager solve. Besides its compilation
    # overhead, staging can trace a custom VJP more than once even though only
    # one branch executes. Under jit the predicate remains an array condition.
    try:
        accepted = bool(converged)
    except (TypeError, ValueError):
        accepted = None
    if accepted is not None:
        return corrected(u_bar) if accepted else u_bar
    return jax.lax.cond(converged, corrected, lambda u: u, u_bar)


def residual_norm(residual_fn, u):
    return jnp.linalg.norm(residual_fn(u))
