"""Check implicit gradients across Galerkin, energy and collocation methods.

Run from the repository root::

    python -m paper.experiments.implicit_differentiation

See paper/README.md for outputs, caching and the suggested reading order.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.solvers.linear import SOLVE_COUNTER, reset_solve_counter

from ._common import ALL, RESULTS, run_example, write_table


def implicit_differentiation():
    """Gradient, adjoint-solve count and jit-ability, per solution method."""
    print("[7b] implicit differentiation, per method")

    lin_V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2))
    linear = jx.Poisson(
        lin_V,
        dirichlet=[jx.DirichletBC(0.0, where=ALL)],
        source=lambda x, p: p["f"] * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
    )
    col_V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(2))
    collocated = jx.Poisson(
        col_V,
        dirichlet=[jx.DirichletBC(0.0, where=ALL)],
        source=lambda x, p: p["f"] * jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]),
    )
    # A nonlinear, overdetermined collocation system: three Dirichlet sides and
    # one Neumann side, so the least-squares residual does not vanish. Both
    # conditions are needed for the Gauss-Newton gap of eq (gauss_newton_gap)
    # to be non-zero.
    class _NonlinearPoisson(jx.Problem):
        is_linear = False

        def energy(self, grad_u, u, x, params):
            return 0.5 * (1.0 + params["k"] * u[0] ** 2) * jnp.sum(grad_u**2)

    nlc_V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(3).refine(2))
    nonlinear_collocation = _NonlinearPoisson(
        nlc_V,
        dirichlet=[
            jx.DirichletBC(0.0, where="left"),
            jx.DirichletBC(0.0, where="right"),
            jx.DirichletBC(0.0, where="bottom"),
        ],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([p["g"]]), where="top")],
        source=lambda x, p: p["f"] * jnp.sin(jnp.pi * x[0]),
    )

    nl_V = jx.FunctionSpace(
        jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2), vec=2
    )
    nonlinear = jx.Hyperelasticity(
        nl_V,
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="left")],
        neumann=[
            jx.Neumann(
                lambda x, n, p: jnp.array([p["load"], 0.3 * p["load"]]), where="right"
            )
        ],
    )

    cases = [
        ("Galerkin, linear", linear, {"a0": 1.3, "f": 2.0}, "f", dict()),
        (
            "Galerkin, Newton",
            nonlinear,
            {"E": 200.0, "nu": 0.3, "load": 5.0},
            "load",
            dict(),
        ),
        (
            "energy minimisation",
            linear,
            {"a0": 1.3, "f": 2.0},
            "f",
            dict(method="energy", tol=1e-12, max_steps=3000),
        ),
        (
            "energy, nonlinear",
            nonlinear,
            {"E": 200.0, "nu": 0.3, "load": 5.0},
            "load",
            dict(method="energy", tol=1e-11, max_steps=6000),
        ),
        ("collocation", collocated, {"a0": 1.0, "f": 2.0}, "f", dict(method="collocation")),
        (
            "collocation, nonlinear",
            nonlinear_collocation,
            {"k": 3.0, "f": 40.0, "g": 5.0},
            "g",
            dict(method="collocation", max_steps=40),
        ),
    ]

    rows = []
    for name, problem, params, key, kwargs in cases:
        loss = lambda p: jnp.sum(jx.solve(problem, params=p, **kwargs).u ** 2)  # noqa: E731

        reset_solve_counter()
        adjoint = float(jax.grad(loss)(params)[key])
        n_adjoint = SOLVE_COUNTER["adjoint"]

        h = 1e-6 * max(1.0, abs(params[key]))
        up = dict(params, **{key: params[key] + h})
        dn = dict(params, **{key: params[key] - h})
        fd = float((loss(up) - loss(dn)) / (2 * h))

        compiled = float(jax.jit(jax.grad(loss))(params)[key])
        jit_gap = abs(compiled - adjoint) / max(abs(adjoint), 1e-30)

        rows.append((name, key, adjoint, fd, abs(adjoint - fd) / abs(fd), n_adjoint, jit_gap))
        print(
            f"    {name:22s} d/d{key:5s} rel.diff {rows[-1][4]:.1e}  "
            f"adjoint solves {n_adjoint}  jit gap {jit_gap:.1e}"
        )

    lines = [
        r"\begin{tabular}{l l l l c c}", r"\hline",
        r"method & param. & adjoint & central diff. & rel. diff. & adjoint solves \\",
        r"\hline",
    ]
    for name, key, ad, fd, rel, n_adj, _ in rows:
        lines.append(
            f"{name} & \\texttt{{{key}}} & {ad:.6e} & {fd:.6e} & {rel:.1e} & {n_adj} \\\\"
        )
    gn_error = _gauss_newton_gap(nonlinear_collocation, {"k": 3.0, "f": 40.0, "g": 5.0})
    print(f"    Gauss-Newton would give {gn_error:.1e} relative error on the last "
          f"row (exact Jacobian: {rows[-1][4]:.1e})")

    lines += [
        r"\hline",
        r"\multicolumn{6}{l}{\footnotesize Every gradient also agrees with its "
        r"\texttt{jax.jit} compilation to "
        + f"{max(r[6] for r in rows):.0e}"
        + r" relative, so the whole} \\",
        r"\multicolumn{6}{l}{\footnotesize solve is one traced program. On the "
        r"last row the Gauss--Newton Jacobian $J^{\top}J$ would give} \\",
        r"\multicolumn{6}{l}{\footnotesize "
        + f"{gn_error:.1e}"
        + r" relative error instead of "
        + f"{rows[-1][4]:.1e}"
        + r"; see eq.~(\ref{eq:gauss_newton_gap}).} \\",
        r"\hline", r"\end{tabular}",
    ]
    write_table("implicit_differentiation", "\n".join(lines))

    RESULTS["collocation_gauss_newton_error"] = gn_error
    RESULTS["max_adjoint_solves"] = max(r[5] for r in rows)
    RESULTS["max_method_grad_error"] = max(r[4] for r in rows)
    RESULTS["max_jit_gap"] = max(r[6] for r in rows)


def _gauss_newton_gap(problem, params):
    """Relative gradient error the Gauss-Newton Jacobian would introduce.

    Computes the implicit derivative twice at the converged solution, once with
    the exact Hessian of the least-squares objective and once with ``J^T J``,
    and reports how far the latter falls from a central difference.
    """
    from jaxiga.methods._common import build_context
    from jaxiga.methods.collocation import (
        _block_residual,
        _rows_and_columns,
        build_blocks,
    )

    key = "g"
    sol = jx.solve(problem, method="collocation", params=params, max_steps=40)
    ctx = build_context(problem, params)
    blocks = build_blocks(problem, params, ctx, None)
    layout, _ = _rows_and_columns(problem, blocks, ctx)

    def residual_vector(u_free, p):
        u = ctx.dofmap.lift(u_free)
        out = []
        for block, _ in zip(blocks, layout):
            r = _block_residual(problem, block, u, p, "d2R") * block.scale
            out.append(r.reshape(-1)[np.flatnonzero(block.comp_mask.reshape(-1))])
        return jnp.concatenate(out)

    u_star = sol.u[ctx.dofmap.free]
    lsq = lambda v: 0.5 * jnp.sum(residual_vector(v, params) ** 2)  # noqa: E731
    J = np.asarray(jax.jacfwd(lambda v: residual_vector(v, params))(u_star))
    A_gn = J.T @ J

    lam = np.linalg.solve(A_gn.T, 2.0 * np.asarray(u_star))
    _, vjp = jax.vjp(lambda p: jax.grad(lambda v: 0.5 * jnp.sum(residual_vector(v, p) ** 2))(u_star), params)
    (pbar,) = vjp(jnp.asarray(lam))
    gn_grad = -float(pbar[key])

    loss = lambda p: jnp.sum(  # noqa: E731
        jx.solve(problem, method="collocation", params=p, max_steps=40).u ** 2
    )
    h = 1e-6 * max(1.0, abs(params[key]))
    fd = float(
        (loss(dict(params, **{key: params[key] + h}))
         - loss(dict(params, **{key: params[key] - h}))) / (2 * h)
    )
    return abs(gn_grad - fd) / abs(fd)


if __name__ == "__main__":
    run_example("implicit_differentiation", implicit_differentiation)
