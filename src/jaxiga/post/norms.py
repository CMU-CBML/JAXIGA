"""Error norms and integrals (Firedrake naming).

One dimension-generic implementation replaces the legacy ``comp_error_norm``,
``comp_error_norm_elast`` and their 1D/3D variants. Norms are **relative** by
default, matching those functions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate


def _eval_user_fn(fn, x):
    """Evaluate a user function over points ``(n_e, n_q, dim)``.

    Accepts either a vectorized function or one written for a single point.
    """
    flat = x.reshape(-1, x.shape[-1])
    try:
        out = fn(flat)
        out = jnp.asarray(out)
        if out.shape[0] != flat.shape[0]:
            raise ValueError
    except Exception:
        out = jax.vmap(fn)(flat)
        out = jnp.asarray(out)
    return out.reshape(*x.shape[:-1], -1)


def integrate(fn, space, quad: int | None = None) -> float:
    """``\\int_\\Omega fn(x) dOmega`` using a Gauss point set."""
    basis = evaluate(space, P.gauss(space, quad))
    vals = _eval_user_fn(fn, basis.x)
    return float(jnp.sum(basis.w[..., None] * vals))


def norm(sol, norm: str = "L2", quad: int | None = None) -> float:
    """Norm of the solution field itself."""
    basis = evaluate(sol.space, P.gauss(sol.space, quad))
    u = sol.at(basis=basis)

    if norm == "L2":
        return float(jnp.sqrt(jnp.sum(basis.w * jnp.sum(u**2, axis=-1))))
    if norm == "H1":
        g = sol.grad(basis=basis)
        return float(
            jnp.sqrt(
                jnp.sum(basis.w * (jnp.sum(u**2, -1) + jnp.sum(g**2, axis=(-1, -2))))
            )
        )
    raise ValueError(f"unknown norm {norm!r}")


def errornorm(
    sol,
    exact,
    norm: str = "L2",
    exact_grad=None,
    quad: int | None = None,
    relative: bool = True,
) -> float:
    """Error against an exact solution.

    Parameters
    ----------
    sol : Solution
    exact : callable
        ``u_exact(x)`` returning ``(vec,)`` for a point, or a vectorized
        equivalent over ``(n, dim)``.
    norm : {"L2", "H1", "energy"}
        ``"energy"`` uses the problem's energy density as the inner product and
        requires ``exact_grad``; for elasticity that is the exact stress, given
        in Voigt form, matching ``comp_error_norm_elast``.
    exact_grad : callable, optional
        Needed by the ``H1`` and ``energy`` norms.
    relative : bool
        Divide by the norm of the exact field (the legacy default).
    """
    space = sol.space
    basis = evaluate(space, P.gauss(space, quad))

    u_h = sol.at(basis=basis)
    u_ex = _eval_user_fn(exact, basis.x).reshape(u_h.shape)
    w = basis.w

    if norm == "L2":
        num = jnp.sum(w * jnp.sum((u_h - u_ex) ** 2, axis=-1))
        den = jnp.sum(w * jnp.sum(u_ex**2, axis=-1))
    elif norm in ("H1", "energy"):
        if exact_grad is None:
            raise ValueError(f"the {norm!r} norm needs exact_grad")
        g_h = sol.grad(basis=basis)
        g_ex = _eval_user_fn(exact_grad, basis.x)

        if norm == "H1":
            g_ex = g_ex.reshape(g_h.shape)
            num = jnp.sum(w * jnp.sum((g_h - g_ex) ** 2, axis=(-1, -2)))
            den = jnp.sum(w * jnp.sum(g_ex**2, axis=(-1, -2)))
        else:
            num, den = _energy_norm(sol, basis, g_h, g_ex, w)
    else:
        raise ValueError(f"unknown norm {norm!r}")

    if not relative:
        return float(jnp.sqrt(num))
    return float(jnp.sqrt(num / den))


def _energy_norm(sol, basis, grad_h, exact_voigt, w):
    """Energy-norm numerator/denominator via the problem's constitutive law.

    ``exact_voigt`` is the exact stress in Voigt form. The strain energy of the
    error is ``1/2 (s_h - s_ex) : C^{-1} : (s_h - s_ex)``, computed here as
    ``1/2 de : C : de`` with ``de`` the corresponding strain difference.
    """
    from jaxiga.forms import materials

    problem = sol.problem
    if not hasattr(problem, "C"):
        raise ValueError(
            f"the energy norm needs a problem with a constitutive matrix; "
            f"{type(problem).__name__} has none"
        )

    C = problem.C(sol.params)
    Cinv = jnp.linalg.inv(C)

    strain_h = jax.vmap(jax.vmap(materials.strain_voigt))(grad_h)
    stress_h = jnp.einsum("ij,eqj->eqi", C, strain_h)
    stress_ex = exact_voigt.reshape(stress_h.shape)

    diff = stress_h - stress_ex
    num = jnp.sum(w * jnp.einsum("eqi,ij,eqj->eq", diff, Cinv, diff))
    den = jnp.sum(w * jnp.einsum("eqi,ij,eqj->eq", stress_ex, Cinv, stress_ex))
    return num, den
