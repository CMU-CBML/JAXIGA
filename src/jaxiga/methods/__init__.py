"""Solution methods and the top-level ``solve`` entry point."""

from __future__ import annotations

from jaxiga.solvers.linear import LinearOptions

_IMPLEMENTED = {"galerkin", "energy", "collocation"}
_PLANNED = set()


def solve(
    problem,
    *,
    method: str = "galerkin",
    params=None,
    linear: LinearOptions | None = None,
    initial_guess=None,
    **method_opts,
):
    """Solve a problem and return a :class:`~jaxiga.post.solution.Solution`.

    Parameters
    ----------
    problem : Problem
    method : {"galerkin", "energy", "collocation"}
    params : pytree
        Passed unchanged to every kernel; anything inside it is differentiable.
    linear : LinearOptions
        Options for the linear solves. Used by the Galerkin tangent solves, by
        the collocation normal equations, and -- for every method -- by the
        adjoint solve that implicit differentiation performs.
    initial_guess : array, optional
        Free-dof starting vector for nonlinear problems.
    """
    from jaxiga.post.solution import Solution

    if method in _PLANNED:
        raise NotImplementedError(
            f"method={method!r} is not implemented yet (phase 3); "
            f"available now: {sorted(_IMPLEMENTED)}"
        )
    if method not in _IMPLEMENTED:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_IMPLEMENTED)}")

    if method == "energy":
        from jaxiga.methods.energy import solve_energy

        u, ctx, stats = solve_energy(
            problem, params, linear=linear, initial_guess=initial_guess, **method_opts
        )
    elif method == "collocation":
        from jaxiga.methods.collocation import solve_collocation

        u, ctx, stats = solve_collocation(
            problem, params, linear=linear, initial_guess=initial_guess, **method_opts
        )
    else:
        from jaxiga.methods.galerkin import solve_galerkin

        u, ctx, stats = solve_galerkin(
            problem, params, linear=linear, initial_guess=initial_guess, **method_opts
        )

    return Solution(space=problem.space, u=u, params=params, problem=problem, stats=stats)


__all__ = ["solve", "LinearOptions"]
