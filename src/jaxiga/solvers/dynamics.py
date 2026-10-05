"""Time integration for second-order dynamics.

Solves ``M a + K u = f(t)`` by the Newmark-beta and generalized-alpha families.
Both are written as :func:`jax.lax.scan` bodies, which buys three things at
once: the loop compiles to a single XLA program, it is differentiable through
time by construction (``scan`` transposes), and the adjoint is a reverse
``scan`` rather than an unrolled tape.

Because the step size is constant and the operators are linear, the effective
matrix ``(1 - alpha_m) M + (1 - alpha_f) beta dt^2 K`` is assembled **once**
outside the loop. Mass and stiffness share their sparsity pattern -- both come
from the same element connectivity -- so combining them is an operation on the
triplet values alone.

Memory
------
Reverse-mode through ``n_steps`` retains the per-step residuals. Pass
``checkpoint_every=k`` to run the loop as an outer scan over chunks of ``k``
checkpointed steps, trading recomputation for memory in the usual way.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.solvers.linear import LinearOptions, sparse_solve


@dataclasses.dataclass(frozen=True)
class Stepper:
    """Coefficients of a generalized-alpha / Newmark scheme."""

    beta: float
    gamma: float
    alpha_m: float = 0.0
    alpha_f: float = 0.0
    name: str = "newmark"

    def __str__(self):
        return (
            f"{self.name}(beta={self.beta:.4g}, gamma={self.gamma:.4g}, "
            f"alpha_m={self.alpha_m:.4g}, alpha_f={self.alpha_f:.4g})"
        )


def newmark(beta: float = 0.25, gamma: float = 0.5) -> Stepper:
    """Newmark-beta. The default is the average-acceleration rule.

    ``beta=1/4, gamma=1/2`` is unconditionally stable and conserves the
    discrete energy ``1/2 v^T M v + 1/2 u^T K u`` exactly for a linear
    undamped system, so any drift observed is round-off.
    """
    return Stepper(beta=beta, gamma=gamma, name="newmark")


def generalized_alpha(rho_inf: float = 0.8) -> Stepper:
    """Chung-Hulbert generalized-alpha with a prescribed high-frequency decay.

    ``rho_inf`` is the spectral radius at an infinite time step: 1 gives no
    numerical damping (and reduces to the average-acceleration rule), 0 gives
    asymptotic annihilation of the highest modes -- useful because a spline
    discretization's top modes are not physical.
    """
    if not 0.0 <= rho_inf <= 1.0:
        raise ValueError(f"rho_inf must lie in [0, 1], got {rho_inf}")
    alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
    return Stepper(
        beta=beta, gamma=gamma, alpha_m=alpha_m, alpha_f=alpha_f, name="generalized-alpha"
    )


def _matvec(indices, data, v, n):
    return jnp.zeros(n).at[indices[:, 0]].add(data * v[indices[:, 1]])


def integrate(
    indices,
    k_data,
    m_data,
    *,
    dt: float,
    n_steps: int,
    u0,
    v0,
    force=None,
    stepper: Stepper | None = None,
    linear: LinearOptions | None = None,
    checkpoint_every: int | None = None,
):
    """March ``M a + K u = f(t)`` forward in time.

    Parameters
    ----------
    indices : (nnz, 2) int array
        Shared COO pattern of the mass and stiffness matrices.
    k_data, m_data : (nnz,) arrays
        Stiffness and mass values on that pattern.
    dt, n_steps : float, int
        Constant step and number of steps.
    u0, v0 : (n,) arrays
        Initial displacement and velocity on the free dofs.
    force : callable, optional
        ``f(t) -> (n,)``. ``None`` means free vibration.
    stepper : Stepper
        Defaults to :func:`newmark`.
    linear : LinearOptions
        The per-step solve. The effective matrix is assembled once, so a direct
        method is refactored every step; ``"cg"`` avoids that on large problems.
    checkpoint_every : int, optional
        Steps per checkpointed chunk for reverse-mode differentiation.

    Returns
    -------
    times, U, V : ``(n_steps+1,)``, ``(n_steps+1, n)``, ``(n_steps+1, n)``
    """
    st = stepper or newmark()
    linear = linear or LinearOptions(method="dense")
    n = u0.shape[0]
    indices = np.asarray(indices)

    Kv = lambda v: _matvec(indices, k_data, v, n)  # noqa: E731
    Mv = lambda v: _matvec(indices, m_data, v, n)  # noqa: E731
    f_of = (lambda t: jnp.zeros(n)) if force is None else force

    # a0 from the equation of motion at t = 0
    a0 = sparse_solve(linear, indices, m_data, f_of(0.0) - Kv(u0))

    eff = (1.0 - st.alpha_m) * m_data + (1.0 - st.alpha_f) * st.beta * dt**2 * k_data

    def step(carry, i):
        u, v, a = carry
        t_next = (i + 1) * dt
        t_eval = t_next - st.alpha_f * dt

        u_pred = u + dt * v + dt**2 * (0.5 - st.beta) * a
        v_pred = v + dt * (1.0 - st.gamma) * a

        rhs = (
            f_of(t_eval)
            - st.alpha_m * Mv(a)
            - (1.0 - st.alpha_f) * Kv(u_pred)
            - st.alpha_f * Kv(u)
        )
        a_next = sparse_solve(linear, indices, eff, rhs)
        u_next = u_pred + st.beta * dt**2 * a_next
        v_next = v_pred + st.gamma * dt * a_next
        return (u_next, v_next, a_next), (u_next, v_next)

    carry0 = (u0, v0, a0)
    if checkpoint_every is None:
        _, (U, V) = jax.lax.scan(step, carry0, jnp.arange(n_steps))
    else:
        k = int(checkpoint_every)
        if n_steps % k:
            raise ValueError(
                f"checkpoint_every={k} must divide n_steps={n_steps}; the loop runs "
                f"as an outer scan over equal chunks"
            )

        @jax.checkpoint
        def chunk(carry, base):
            return jax.lax.scan(step, carry, base + jnp.arange(k))

        _, (U, V) = jax.lax.scan(
            chunk, carry0, jnp.arange(0, n_steps, k)
        )
        U = U.reshape(-1, n)
        V = V.reshape(-1, n)

    times = jnp.arange(n_steps + 1) * dt
    return times, jnp.concatenate([u0[None], U]), jnp.concatenate([v0[None], V])


def energy(indices, k_data, m_data, u, v):
    """``1/2 v^T M v + 1/2 u^T K u`` for one state or a whole history."""
    n = u.shape[-1]
    indices = np.asarray(indices)

    def one(uu, vv):
        return 0.5 * vv @ _matvec(indices, m_data, vv, n) + 0.5 * uu @ _matvec(
            indices, k_data, uu, n
        )

    if u.ndim == 1:
        return one(u, v)
    return jax.vmap(one)(u, v)


def natural_frequencies(indices, k_data, m_data, n_modes: int | None = None):
    """Angular frequencies of ``K phi = omega^2 M phi``, ascending.

    Densified: this is a validation and diagnostic helper, not a solver path.
    """
    import scipy.linalg

    indices = np.asarray(indices)
    n = int(indices.max()) + 1
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    np.add.at(K, (indices[:, 0], indices[:, 1]), np.asarray(k_data))
    np.add.at(M, (indices[:, 0], indices[:, 1]), np.asarray(m_data))
    vals = scipy.linalg.eigh(K, M, eigvals_only=True)
    omega = np.sqrt(np.maximum(vals, 0.0))
    return omega if n_modes is None else omega[:n_modes]
