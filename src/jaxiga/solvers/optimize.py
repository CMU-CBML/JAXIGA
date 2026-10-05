"""Unconstrained minimizers.

Thin wrappers over optax, replacing the hand-rolled ``utils/bfgs.py``,
``utils/lbfgs.py``, ``utils/jax_tfp_loss.py`` and the ad-hoc optimizer classes
in ``utils/Solvers.py``.

L-BFGS with a zoom line search is the default: the discrete potential energy is
smooth and, for linear problems, exactly quadratic, so a quasi-Newton method
reaches the minimizer to solver tolerance rather than merely near it. Adam is
offered for the warm-up phase that the legacy energy-minimization examples use
on stiff nonlinear problems.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import optax


@dataclasses.dataclass
class OptimizeResult:
    """Outcome of a minimization.

    The scalar fields are JAX values rather than Python floats, so a
    minimisation stays usable under ``jit``. ``history`` is a fixed-length
    array of gradient norms padded with NaN; :func:`history_list` trims it.
    """

    x: jnp.ndarray
    value: jnp.ndarray
    grad_norm: jnp.ndarray
    steps: jnp.ndarray
    converged: jnp.ndarray
    history: jnp.ndarray

    def __repr__(self):
        try:
            return (
                f"OptimizeResult(value={float(self.value):.6e}, "
                f"grad_norm={float(self.grad_norm):.3e}, "
                f"steps={int(self.steps)}, converged={bool(self.converged)})"
            )
        except Exception:  # traced
            return "OptimizeResult(<traced>)"


def history_list(result):
    """Gradient-norm history as a plain list, with the NaN padding removed."""
    import numpy as np

    h = np.asarray(result.history)
    return [float(v) for v in h[~np.isnan(h)]]


def _run(optimizer, loss_fn, x0, max_steps, tol, record_every, extra_args=False):
    """Iterate ``optimizer`` on ``loss_fn`` inside a single traced loop.

    A ``lax.while_loop`` rather than a Python loop, for two reasons. It keeps
    the whole minimisation traceable, so an energy solve composes with ``jit``
    the way a linear solve does. And it compiles the step once: L-BFGS's zoom
    line search calls ``value_fn`` several times per update, and re-tracing the
    discrete energy -- assembly, evaluation and all -- on each of those calls
    dominated the run time by two orders of magnitude.
    """
    value_and_grad = jax.value_and_grad(loss_fn)
    n_slots = int(max_steps) + 1

    def step_fn(state):
        x, opt_state, value, grad, step, hist = state
        updates, opt_state = optimizer.update(
            grad, opt_state, x, value=value, grad=grad, value_fn=loss_fn
        )
        x = optax.apply_updates(x, updates)
        value, grad = value_and_grad(x)
        gnorm = jnp.linalg.norm(grad)
        return x, opt_state, value, grad, step + 1, hist.at[step + 1].set(gnorm)

    def cond_fn(state):
        _, _, _, grad, step, _ = state
        return (jnp.linalg.norm(grad) > tol) & (step < max_steps)

    value, grad = value_and_grad(x0)
    hist0 = jnp.full(n_slots, jnp.nan).at[0].set(jnp.linalg.norm(grad))
    state = (x0, optimizer.init(x0), value, grad, jnp.asarray(0), hist0)
    x, _, value, grad, steps, hist = jax.lax.while_loop(cond_fn, step_fn, state)

    gnorm = jnp.linalg.norm(grad)
    return OptimizeResult(
        x=x,
        value=value,
        grad_norm=gnorm,
        steps=steps,
        converged=gnorm <= tol,
        history=hist,
    )


def minimize_lbfgs(loss_fn, x0, *, tol: float = 1e-10, max_steps: int = 2000,
                   memory_size: int = 20, record_every: int = 50) -> OptimizeResult:
    """Minimize with L-BFGS and a zoom line search.

    ``record_every`` is retained for compatibility; history records every step.
    """
    return _run(
        optax.lbfgs(memory_size=memory_size),
        loss_fn, x0, max_steps, tol, record_every, extra_args=True,
    )


def minimize_adam(loss_fn, x0, *, learning_rate: float = 1e-3, max_steps: int = 5000,
                  tol: float = 1e-10, record_every: int = 200) -> OptimizeResult:
    """Minimize with Adam.

    Adam does not use the line-search hooks, so its update is called plainly,
    and its step count is fixed rather than tolerance-driven -- which makes a
    ``scan`` the natural loop.
    ``record_every`` is retained for compatibility; history records every step.
    """
    optimizer = optax.adam(learning_rate)

    def step_fn(carry, _):
        x, state = carry
        value, grad = jax.value_and_grad(loss_fn)(x)
        updates, state = optimizer.update(grad, state, x)
        return (optax.apply_updates(x, updates), state), jnp.linalg.norm(grad)

    (x, _), gnorms = jax.lax.scan(
        step_fn, (x0, optimizer.init(x0)), None, length=max_steps
    )

    value, grad = jax.value_and_grad(loss_fn)(x)
    gnorm = jnp.linalg.norm(grad)
    return OptimizeResult(
        x=x,
        value=value,
        grad_norm=gnorm,
        steps=jnp.asarray(max_steps),
        converged=gnorm <= tol,
        history=gnorms,
    )


def minimize(loss_fn, x0, *, optimizer: str = "lbfgs", **kwargs) -> OptimizeResult:
    """Dispatch by name.

    ``"adam+lbfgs"`` runs a short Adam warm-up before L-BFGS, the pattern the
    legacy energy-minimization examples use to get past bad initial guesses.
    """
    if optimizer == "lbfgs":
        return minimize_lbfgs(loss_fn, x0, **kwargs)
    if optimizer == "adam":
        return minimize_adam(loss_fn, x0, **kwargs)
    if optimizer == "adam+lbfgs":
        warmup = int(kwargs.pop("warmup_steps", 500))
        lr = kwargs.pop("learning_rate", 1e-3)
        first = minimize_adam(loss_fn, x0, learning_rate=lr, max_steps=warmup,
                              record_every=max(warmup // 4, 1))
        second = minimize_lbfgs(loss_fn, first.x, **kwargs)
        # Adam records pre-update norms; L-BFGS starts at the warm-up's final
        # state. Concatenation therefore records each state once, including
        # L-BFGS's fixed-size NaN padding after convergence.
        second.history = jnp.concatenate([first.history, second.history])
        second.steps += first.steps
        return second
    raise ValueError(
        f"unknown optimizer {optimizer!r}; expected 'lbfgs', 'adam' or 'adam+lbfgs'"
    )
