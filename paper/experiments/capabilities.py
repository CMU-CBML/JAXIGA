"""Demonstrate matrix-free assembly, chunking, batched solves and time integration.

Run from the repository root::

    python -m paper.experiments.capabilities

See paper/README.md for outputs, caching and the suggested reading order.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxiga as jx

from ._common import RESULTS, SEED, run_example, write_table
from .poisson_convergence import poisson_problem


def capabilities():
    print("[10] assembly variants, batching, dynamics")
    from jaxiga.methods._common import build_context as _ctx
    from jaxiga.methods.galerkin import assemble_triplets, tangent_matvec
    from jaxiga.solvers import dynamics
    from jaxiga.solvers.linear import LinearOptions

    rows = []

    # -- matrix-free vs assembled ----------------------------------------
    problem, _ = poisson_problem(2, 2, 4)
    ref = jx.solve(problem, params={"a0": 1.0}, linear=LinearOptions(method="dense"))
    ctx = _ctx(problem, {"a0": 1.0})
    indices, data = assemble_triplets(ctx, {"a0": 1.0})
    n = len(ctx.dofmap.free)
    K = np.zeros((n, n))
    np.add.at(K, (indices[:, 0], indices[:, 1]), np.asarray(data))
    v = np.random.default_rng(SEED).normal(size=n)
    mv = np.asarray(tangent_matvec(ctx, {"a0": 1.0})(jnp.asarray(v)))
    mf_matvec = float(np.abs(mv - K @ v).max() / np.abs(K @ v).max())

    mf = jx.solve(problem, params={"a0": 1.0},
                  linear=LinearOptions(method="cg", matrix_free=True, tol=1e-13))
    rows.append((r"matrix-free matvec vs assembled $Kv$", f"{mf_matvec:.1e}",
                 "relative"))
    rows.append((r"matrix-free vs assembled solution", 
                 f"{float(jnp.abs(mf.u - ref.u).max()):.1e}", "absolute"))

    # -- chunked assembly -------------------------------------------------
    ch = jx.solve(problem, params={"a0": 1.0},
                  linear=LinearOptions(method="dense"), chunk=8)
    rows.append((r"chunked vs eager assembly (8 elements/chunk)",
                 f"{float(jnp.abs(ch.u - ref.u).max()):.1e}", "absolute"))

    # -- batched solves ---------------------------------------------------
    small, _ = poisson_problem(2, 2, 3)
    f = lambda a0: jx.solve(small, params={"a0": a0},
                            linear=LinearOptions(method="dense")).u
    thetas = jnp.linspace(0.5, 4.0, 8)
    loop = jnp.stack([f(a) for a in thetas])
    batch = jax.vmap(f)(thetas)
    rows.append((r"batched (\texttt{vmap}) vs looped solves, 8 parameters",
                 f"{float(jnp.abs(batch - loop).max()):.1e}", "absolute"))

    # -- dynamics ---------------------------------------------------------
    E_bar, rho_bar, L_bar = 2.0, 3.0, 1.0
    Vb = jx.FunctionSpace(jx.primitives.interval(0.0, L_bar).elevate(3).refine(5))
    bar = jx.Poisson(Vb, dirichlet=[jx.DirichletBC(0.0, where="left"),
                                    jx.DirichletBC(0.0, where="right")])
    idx, mdat, kdat, bctx = jx.mass_matrix(bar, rho=rho_bar, params={"a0": E_bar})
    omega = dynamics.natural_frequencies(idx, kdat, mdat, 5)
    exact_omega = np.array([(k + 1) * np.pi / L_bar * np.sqrt(E_bar / rho_bar)
                            for k in range(5)])
    freq_err = float(np.abs(omega / exact_omega - 1).max())
    rows.append((r"bar natural frequencies, first five modes",
                 f"{freq_err:.1e}", "relative"))

    nf = len(bctx.dofmap.free)
    xb = np.asarray(Vb.cpts).reshape(-1)[bctx.dofmap.free]
    u0 = jnp.asarray(np.sin(np.pi * xb / L_bar))
    _, U, W = dynamics.integrate(idx, kdat, mdat, dt=2e-3, n_steps=1000,
                                 u0=u0, v0=jnp.zeros(nf),
                                 linear=LinearOptions(method="dense"))
    en = dynamics.energy(idx, kdat, mdat, U, W)
    drift = float(jnp.abs(en - en[0]).max() / en[0])
    rows.append((r"Newmark energy drift, 1000 steps ($\beta=\tfrac14$, $\gamma=\tfrac12$)",
                 f"{drift:.1e}", "relative"))

    for label, value, _ in rows:
        print(f"    {label}: {value}")

    lines = [r"\begin{tabular}{l l l}", r"\hline",
             r"Check & Discrepancy & Measure \\", r"\hline"]
    for label, value, measure in rows:
        lines.append(f"{label} & {value} & {measure} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write_table("capabilities", "\n".join(lines))

    RESULTS["matrix_free_matvec"] = mf_matvec
    RESULTS["newmark_energy_drift"] = drift
    RESULTS["bar_frequency_error"] = freq_err


if __name__ == "__main__":
    run_example("capabilities", capabilities)
