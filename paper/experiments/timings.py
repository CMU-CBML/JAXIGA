"""Measure setup, solve, compilation and gradient costs on Poisson problems.

Run from the repository root::

    python -m paper.experiments.timings

See paper/README.md for outputs, caching and the suggested reading order.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.methods._common import build_context

from ._common import RESULTS, run_example, write_table
from .poisson_convergence import poisson_problem


def timings(repeats=3):
    """Cost of setup, solve and gradient.

    Every measured quantity is warmed up first, so the reported numbers are
    steady-state and exclude JIT compilation; compilation is reported
    separately, since for one-off solves it dominates. Comparing a warmed
    gradient against a cold solve would make the adjoint look faster than the
    forward problem, which it is not.
    """
    print("[7] Timings")

    def best_of(fn, n=repeats):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    rows = []
    for deg in (2, 3, 4):
        for r in (3, 4, 5):
            problem, _ = poisson_problem(2, deg, r)
            params = {"a0": 1.0}

            def setup():
                ctx = build_context(problem, params)
                ctx.interior.R.block_until_ready()

            def solve():
                jx.solve(problem, params=params).u.block_until_ready()

            loss = lambda a: jnp.sum(jx.solve(problem, params={"a0": a}).u ** 2)
            grad = jax.grad(loss)

            def grad_call():
                jax.block_until_ready(grad(1.0))

            # Cold cost including compilation. The caches must be cleared
            # explicitly: without this the number measures "first call in this
            # process", which depends on what happened to be compiled earlier.
            jax.clear_caches()
            t0 = time.perf_counter()
            solve()
            t_cold = time.perf_counter() - t0

            setup(); solve(); grad_call()          # warm up all three
            t_setup = best_of(setup)
            t_solve = best_of(solve)
            t_grad = best_of(grad_call)

            rows.append((deg, r, problem.space.n_dofs, problem.space.n_elems,
                         t_setup, t_solve, t_grad, t_grad / t_solve, t_cold))
            print(f"    p={deg} r={r}: setup {t_setup:.3f}s solve {t_solve:.3f}s "
                  f"grad {t_grad:.3f}s (x{t_grad/t_solve:.2f}), first call {t_cold:.2f}s")

    lines = [
        r"\begin{tabular}{c c r r l l l c c}", r"\hline",
        r"$p$ & ref. & dofs & elems & setup [s] & solve [s] & gradient [s] & "
        r"$\frac{\text{grad}}{\text{solve}}$ & first call [s] \\",
        r"\hline",
    ]
    for deg, r, n, ne, ts, tv, tg, ratio, tc in rows:
        lines.append(f"{deg} & {r} & {n} & {ne} & {ts:.3f} & {tv:.3f} & {tg:.3f} & "
                     f"{ratio:.2f} & {tc:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write_table("timings", "\n".join(lines))
    RESULTS["max_grad_ratio"] = max(r[7] for r in rows)
    RESULTS["min_grad_ratio"] = min(r[7] for r in rows)


if __name__ == "__main__":
    run_example("timings", timings)
