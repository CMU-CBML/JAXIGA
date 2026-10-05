"""A posteriori error estimation and the adaptive refinement driver.

The explicit residual estimator

    eta_e^2 = h_e^2 * sum_q w_q |strong_residual(u_h)|^2

reuses the collocation machinery verbatim -- :func:`strong_residual_d2R` with
an ``order=2`` evaluation -- so no new mathematics enters the package.

Spline continuity reduces the omitted interface contributions. Inside a
patch the discretization is C^{p-1} for p >= 2, so flux jumps vanish across
smooth interior knots when the coefficients are continuous. The implemented
indicator uses only the volume term. It omits boundary residuals and jumps at
coefficient discontinuities, patch interfaces and repeated knots with C^0
continuity. It can therefore under-count error near these sets and bias marking
away from them; see :func:`residual_indicator`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.methods._common import expand_dofs_traced
from jaxiga.methods.collocation import _voigt_to_full, strong_residual_d2R
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate
from jaxiga.space.function_space import pad_coeffs
from jaxiga.space.hierarchical import dorfler_mark, refine_elements


def residual_indicator(sol, params=None, quad: int | None = None) -> np.ndarray:
    """Per-element explicit residual indicators ``eta_e``.

    Parameters
    ----------
    sol : Solution
        A solved field; its problem supplies the strong form by autodiff of the
        same energy density the solve used.
    params : optional
        Parameters to evaluate the residual at; defaults to the ones the
        solution was computed with.
    quad : int, optional
        Gauss points per direction. Defaults to ``p + 2``, one more than the
        solve, since the residual is rougher than the solution.

    Returns
    -------
    (n_elems,) array of non-negative indicators.

    Notes
    -----
    Volume term only. Across a C^0 line -- a patch interface, or a repeated
    interior knot -- the flux jump is a genuine part of the error that this
    does not see. It still *ranks* elements sensibly there, because a
    singularity sitting on such a line also blows the volume residual up in the
    surrounding elements, which is what marking needs.
    """
    problem = sol.problem
    if problem is None:
        raise ValueError("residual_indicator needs a Solution that carries its problem")
    space = problem.space
    if space.cell_type in {"simplex", "mixed"}:
        raise NotImplementedError("simplex error estimation requires interelement flux jumps, not implemented")
    params = sol.params if params is None else params
    n_q = quad if quad is not None else max(space.degree) + 2

    basis = evaluate(space, P.gauss(space, n_q), order=2)
    vec, dim = space.vec, space.dim

    u_local = pad_coeffs(sol.u, vec)[expand_dofs_traced(basis.dofs, vec)].reshape(
        basis.n_elems, basis.n_local, vec
    )
    u_q = jnp.einsum("eqn,enc->eqc", basis.R, u_local)
    grad_q = jnp.einsum("eqdn,enc->eqcd", basis.dR, u_local)
    d2_q = jnp.einsum("eqkn,enc->eqck", basis.d2R, u_local)

    def one(u, grad_u, d2, x):
        full = jax.vmap(_voigt_to_full, in_axes=(0, None))(d2, dim)  # (vec, dim, dim)
        return strong_residual_d2R(problem, u, grad_u, full, x, params)

    res = jax.vmap(jax.vmap(one))(u_q, grad_q, d2_q, basis.x)  # (n_e, n_q, vec)

    measure = jnp.sum(basis.w, axis=1)  # element volume
    h = measure ** (1.0 / dim)
    eta2 = h**2 * jnp.sum(basis.w * jnp.sum(res**2, axis=-1), axis=1)
    return np.asarray(jnp.sqrt(jnp.maximum(eta2, 0.0)))


def adapt(
    make_problem,
    params=None,
    *,
    space=None,
    n_cycles: int = 5,
    frac: float = 0.3,
    exact=None,
    exact_grad=None,
    solve_kwargs=None,
    verbose: bool = False,
):
    """Solve-estimate-mark-refine, returning the last solution and its space.

    Parameters
    ----------
    make_problem : callable
        ``FunctionSpace -> Problem``. A **factory**, not a problem: a ``Problem``
        resolves its ``where`` clauses against a fixed space at construction, so
        each cycle needs a fresh one.
    params : optional
        Passed to every solve.
    space : FunctionSpace, optional
        Where to start. Defaults to ``make_problem.space`` if the factory
        carries one.
    n_cycles : int
        Solves performed. ``n_cycles - 1`` refinements happen between them.
    frac : float
        Dorfler bulk-marking fraction.
    exact, exact_grad : callable, optional
        If given, the true error is recorded per cycle alongside the estimate.
    solve_kwargs : dict, optional
        Forwarded to :func:`jaxiga.solve`.

    Returns
    -------
    sol, space, history
        ``history`` is a list of dicts with ``n_dofs``, ``n_elems``, ``eta``
        and, when an exact solution was supplied, ``L2`` and ``H1``.

    Notes
    -----
    Marking changes array shapes, so this is an ordinary Python loop and
    ``jax.grad`` does not flow across cycles. Each adapted space is otherwise a
    full citizen of the differentiability contract: gradients with respect to
    ``params`` and ``cpts`` on any fixed adapted space work as usual.
    """
    from jaxiga.methods import solve as _solve
    from jaxiga.post.norms import errornorm

    if not callable(make_problem):
        raise TypeError(
            "adapt() takes a factory FunctionSpace -> Problem, not a Problem: a "
            "problem is bound to one space at construction"
        )

    if space is None:
        space = getattr(make_problem, "space", None)
    if space is None:
        raise TypeError(
            "adapt() needs a starting space: pass space=V, or set "
            "make_problem.space = V on the factory"
        )

    history = []
    sol = None
    for cycle in range(n_cycles):
        problem = make_problem(space)
        sol = _solve(problem, params=params, **(solve_kwargs or {}))
        eta = residual_indicator(sol, params)

        entry = {
            "cycle": cycle,
            "n_dofs": space.n_dofs,
            "n_elems": space.n_elems,
            "eta": float(np.sqrt(np.sum(eta**2))),
        }
        if exact is not None:
            entry["L2"] = errornorm(sol, exact, "L2")
            if exact_grad is not None:
                entry["H1"] = errornorm(sol, exact, "H1", exact_grad=exact_grad)
        history.append(entry)
        if verbose:
            extra = "".join(f"  {k}={entry[k]:.4e}" for k in ("L2", "H1") if k in entry)
            print(
                f"  cycle {cycle}: {entry['n_dofs']:6d} dofs, "
                f"{entry['n_elems']:6d} elems, eta={entry['eta']:.4e}{extra}"
            )

        if cycle + 1 < n_cycles:
            space = refine_elements(space, dorfler_mark(eta, frac))

    return sol, space, history
