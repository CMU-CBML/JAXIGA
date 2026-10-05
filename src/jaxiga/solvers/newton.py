"""Newton's method with backtracking line search.

The iteration is a :func:`jax.lax.while_loop`, so the whole solve is one traced
program: it compiles, and it composes with ``jit`` in the way the linear path
does. Convergence tests are array comparisons rather than Python ``float``
conversions, which is what makes that possible.

The converged root is differentiated by the implicit function theorem
(:mod:`jaxiga.solvers.implicit`), never by unrolling the iteration: the iterates
are detached and a single adjoint solve with the tangent at the root supplies
the derivative. The cost and memory of a gradient are therefore independent of
the number of Newton steps taken.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.solvers.implicit import implicit_correction
from jaxiga.solvers.linear import LinearOptions, matrix_free_solve, sparse_solve

MAX_BACKTRACKS = 20


def newton(
    residual_fn,
    u0_free,
    params,
    *,
    tangent_fn,
    matvec_fn=None,
    diagonal_fn=None,
    lin_opts: LinearOptions | None = None,
    tol: float = 1e-8,
    max_steps: int = 25,
    line_search: bool = True,
    implicit: bool = True,
):
    """Solve ``residual_fn(u, params) = 0`` starting from ``u0_free``.

    ``tangent_fn(u, params)`` returns ``(pattern, data)`` of the tangent. The
    pattern may be a static CSR structure or backwards-compatible COO indices.
    Passing ``matvec_fn(u, params) -> (v -> A v)`` instead solves each step
    matrix-free, with ``diagonal_fn`` supplying the Jacobi diagonal.

    Returns ``(u_free, stats)``. ``stats`` holds JAX scalars, so it is usable
    under ``jit``; ``history`` is a fixed-length array padded with NaN.
    """
    lin_opts = lin_opts or LinearOptions()

    def solve_step(u, r):
        """One Newton step: solve ``A du = -r``."""
        if matvec_fn is not None:
            return matrix_free_solve(
                lin_opts,
                matvec_fn(u, params),
                -r,
                diagonal=None if diagonal_fn is None else diagonal_fn(u, params),
            )
        indices, data = tangent_fn(u, params)
        return sparse_solve(lin_opts, indices, data, -r)

    def norm_of(u):
        return jnp.linalg.norm(residual_fn(u, params))

    def backtrack(u, du, r_norm):
        """Halve the step until the residual decreases, or report a stall.

        A ``while_loop`` rather than a Python loop, so the whole Newton solve
        stays one traced program.
        """

        def cond(state):
            _, trial, k = state
            reject = ~jnp.isfinite(trial) | (trial >= r_norm)
            return reject & (k < MAX_BACKTRACKS)

        def body(state):
            alpha, _, k = state
            alpha = 0.5 * alpha
            return alpha, norm_of(u + alpha * du), k + 1

        alpha0 = jnp.asarray(1.0)
        state = (alpha0, norm_of(u + alpha0 * du), jnp.asarray(0))
        alpha, trial, _ = jax.lax.while_loop(cond, body, state)
        # The final allowed trial can succeed; judge it by its residual.
        accepted = jnp.isfinite(trial) & (trial < r_norm)
        return jnp.where(accepted, alpha, 0.0)

    n_slots = int(max_steps) + 1

    def cond(state):
        _, r_norm, step, alpha, _ = state
        return jnp.isfinite(r_norm) & (r_norm > tol) & (step < max_steps) & (alpha > 0.0)

    def body(state):
        u, r_norm, step, _, hist = state
        r = residual_fn(u, params)
        du = solve_step(u, r)
        alpha = backtrack(u, du, r_norm) if line_search else jnp.asarray(1.0)
        moved = alpha > 0.0
        u_next = jnp.where(moved, u + alpha * du, u)
        new_norm = jnp.where(moved, norm_of(u_next), r_norm)
        return u_next, new_norm, step + 1, alpha, hist.at[step + 1].set(new_norm)

    r0 = jnp.linalg.norm(residual_fn(u0_free, params))
    hist0 = jnp.full(n_slots, jnp.nan).at[0].set(r0)
    u, r_norm, steps, alpha, history = jax.lax.while_loop(
        cond, body, (u0_free, r0, jnp.asarray(0), jnp.asarray(1.0), hist0)
    )

    converged = r_norm <= tol
    stalled = (alpha <= 0.0) & jnp.logical_not(converged)

    if implicit:
        # Detach the iterates and attach the implicit-function-theorem
        # derivative: one adjoint solve, whatever `steps` turned out to be.
        # solve_step(u, r) solves A du = -r, i.e. it returns -A^{-1} r, so the
        # plain tangent solve A x = rhs is solve_step(u, -rhs).
        u = implicit_correction(
            u,
            lambda uu: residual_fn(uu, params),
            lambda uu, rhs: solve_step(uu, -rhs),
            converged=converged,
        )

    # Diagnostics describe the state returned to the caller, not the state
    # before the implicit correction.
    r_norm = norm_of(u)
    finite = jnp.isfinite(r_norm) & jnp.all(jnp.isfinite(u))
    converged = finite & (r_norm <= tol)

    stats = {
        "iterations": steps,
        "residual": r_norm,
        "converged": converged,
        "finite": finite,
        "iteration_limit": (steps >= max_steps) & ~converged,
        "stalled": stalled,
        "history": history,
    }
    _warn_if_not_converged(stats, tol, max_steps)
    return u, stats


def _warn_if_not_converged(stats, tol, max_steps):
    """Warn outside a trace; stay silent when the values are tracers.

    Under ``jit`` there is nothing to inspect at trace time, and a warning then
    would describe the abstract program rather than any particular solve.
    """
    try:
        converged = bool(stats["converged"])
        residual = float(stats["residual"])
        steps = int(stats["iterations"])
        stalled = bool(stats["stalled"])
    except Exception:
        return

    if converged:
        return

    import warnings

    if not np.isfinite(residual):
        reason = "a non-finite state was encountered"
    elif stalled:
        reason = "the line search found no finite decreasing step"
    elif steps >= max_steps:
        reason = f"the {max_steps}-step limit was reached"
    else:
        reason = "the final correction did not satisfy the tolerance"
    warnings.warn(
        f"Newton did not reach tol={tol:g}: {reason} after {steps} steps, "
        f"final residual {residual:.3e}",
        RuntimeWarning,
        stacklevel=3,
    )


def history_list(stats):
    """The residual history as a plain list, with the NaN padding removed."""
    h = np.asarray(stats["history"])
    return [float(v) for v in h[~np.isnan(h)]]
