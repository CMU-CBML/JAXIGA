"""Residual diagnostics that remain available under JAX transformations."""

import warnings

import jax.numpy as jnp


def linear_diagnostics(matvec, u, b, tol):
    """Check the actual equation, independently of a Krylov solver's status."""
    norm = jnp.linalg.norm(matvec(u) - b)
    scale = jnp.linalg.norm(b)
    relative = norm / jnp.where(scale > 0, scale, 1.0)
    finite = jnp.isfinite(norm) & jnp.all(jnp.isfinite(u))
    return {
        "residual": norm,
        "relative_residual": relative,
        "converged": finite & (norm <= tol * scale),
        "finite": finite,
    }


def warn_if_failed(name, converged, residual):
    """Warn eagerly; compiled callers inspect the returned array diagnostics."""
    try:
        ok, norm = bool(converged), float(residual)
    except (TypeError, ValueError):  # JAX tracers cannot be materialised
        return
    if not ok:
        warnings.warn(
            f"{name} did not converge (residual {norm:.3e}); "
            "do not use this result for implicit sensitivities",
            RuntimeWarning,
            stacklevel=3,
        )
