"""Recover a stiffness inclusion by differentiating through a time integration loop.

Run from the repository root::

    python -m paper.experiments.full_waveform_inversion

See paper/README.md for outputs, caching and the suggested reading order.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

import jaxiga as jx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import FIGURES, RESULTS, run_example, write_table


FWI_E0, FWI_RHO, FWI_L = 1.0, 1.0, 1.0
FWI_DT, FWI_STEPS, FWI_CKPT = 5e-3, 360, 40
FWI_FREQ, FWI_T0, FWI_WIDTH = 3.5, 0.35, 0.12
FWI_TRUE = {"contrast": 0.60, "centre": 0.62}


class _Bar(jx.Problem):
    """A 1D elastic bar whose stiffness carries a smooth Gaussian inclusion."""

    is_linear = True

    def energy(self, grad_u, u, x, params):
        stiffness = FWI_E0 * (
            1.0
            + params["contrast"]
            * jnp.exp(-(((x[0] - params["centre"]) / FWI_WIDTH) ** 2))
        )
        return 0.5 * stiffness * jnp.sum(grad_u**2)


def full_waveform_inversion():
    """Recover a stiffness anomaly by differentiating through 360 time steps."""
    from jaxiga.methods._common import build_context
    from jaxiga.methods.galerkin import assemble_triplets, mass_triplets, rhs
    from jaxiga.solvers import dynamics
    from jaxiga.solvers.linear import LinearOptions

    print("[7c] full-waveform inversion")
    V = jx.FunctionSpace(jx.primitives.interval(0.0, FWI_L).elevate(2).refine(7))
    problem = _Bar(
        V,
        dirichlet=[jx.DirichletBC(0.0, where="right")],
        neumann=[jx.Neumann(lambda x, n, p: jnp.array([1.0]), where="left")],
    )
    ctx0 = build_context(problem, FWI_TRUE)
    load = rhs(problem, FWI_TRUE, ctx0)
    _, m_data = mass_triplets(ctx0, FWI_RHO)
    n_free = len(ctx0.dofmap.free)
    probe_dof = int(np.argmin(np.asarray(V.cpts).reshape(-1)[ctx0.dofmap.free]))

    def ricker(t):
        a = (jnp.pi * FWI_FREQ * (t - FWI_T0)) ** 2
        return (1.0 - 2.0 * a) * jnp.exp(-a)

    def trace(p):
        ctx = build_context(problem, p)
        indices, k_data = assemble_triplets(ctx, p)
        _, _, W = dynamics.integrate(
            indices, k_data, m_data,
            dt=FWI_DT, n_steps=FWI_STEPS,
            u0=jnp.zeros(n_free), v0=jnp.zeros(n_free),
            force=lambda t: ricker(t) * load,
            linear=LinearOptions(method="dense"),
            checkpoint_every=FWI_CKPT,
        )
        return W[:, probe_dof]

    traced = jax.jit(trace)
    observed = traced(FWI_TRUE)
    flat = traced({"contrast": 0.0, "centre": 0.5})
    signal = float(jnp.linalg.norm(observed - flat) / jnp.linalg.norm(observed))

    def misfit(p):
        return 0.5 * jnp.sum((trace(p) - observed) ** 2) / jnp.sum(observed**2)

    guess = {"contrast": 0.20, "centre": 0.50}

    # adjoint through the whole time loop, against a central difference
    grads = jax.grad(misfit)(guess)
    fd = {}
    h = 1e-5
    for key in guess:
        up = dict(guess, **{key: guess[key] + h})
        dn = dict(guess, **{key: guess[key] - h})
        fd[key] = float((misfit(up) - misfit(dn)) / (2 * h))

    value_and_grad = jax.jit(jax.value_and_grad(misfit))
    start = float(misfit(guess))
    opt = optax.adam(0.03)
    state = opt.init(guess)
    for _ in range(90):
        _, g = value_and_grad(guess)
        updates, state = opt.update(g, state, guess)
        guess = optax.apply_updates(guess, updates)
    final = float(misfit(guess))

    print(f"    inclusion signal {100 * signal:.1f}% of the trace norm")
    for key in FWI_TRUE:
        err = abs(float(guess[key]) / FWI_TRUE[key] - 1.0)
        print(f"    {key:9s} true {FWI_TRUE[key]:.3f} -> recovered "
              f"{float(guess[key]):.4f} ({100 * err:.1f}% off)")
    print(f"    misfit {start:.3e} -> {final:.3e}")

    lines = [
        r"\begin{tabular}{l l l}", r"\hline",
        r"Quantity & Value & Note \\", r"\hline",
        rf"inclusion echo & {100 * signal:.1f}\% of trace norm & the entire signal \\",
    ]
    for key in FWI_TRUE:
        lines.append(
            rf"{key} & {FWI_TRUE[key]:.3f} $\to$ {float(guess[key]):.4f} & "
            rf"{100 * abs(float(guess[key]) / FWI_TRUE[key] - 1.0):.1f}\% error \\"
        )
    lines += [
        rf"misfit & {start:.3e} $\to$ {final:.3e} & {start / final:.0f}$\times$ lower \\",
        r"\hline",
    ]
    for key in guess:
        rel = abs(float(grads[key]) - fd[key]) / max(abs(fd[key]), 1e-30)
        lines.append(
            rf"$\partial$misfit$/\partial${key} & {float(grads[key]):.6e} & "
            rf"vs.\ central diff., {rel:.0e} rel. \\"
        )
    lines += [r"\hline", r"\end{tabular}"]
    write_table("fwi", "\n".join(lines))

    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    t = np.arange(FWI_STEPS + 1) * FWI_DT
    ax.plot(t, np.asarray(observed), lw=1.1, label="observed")
    ax.plot(t, np.asarray(traced(guess)), "--", lw=1.1, label="recovered")
    ax.plot(t, np.asarray(observed - flat), lw=0.8, c="0.55",
            label="inclusion echo (observed $-$ homogeneous)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity at the source")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(os.path.join(FIGURES, "fwi.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/fwi.pdf")

    RESULTS["fwi_signal_fraction"] = signal
    RESULTS["fwi_misfit_reduction"] = start / final
    RESULTS["fwi_max_param_error"] = max(
        abs(float(guess[k]) / FWI_TRUE[k] - 1.0) for k in FWI_TRUE
    )
    RESULTS["fwi_max_grad_error"] = max(
        abs(float(grads[k]) - fd[k]) / max(abs(fd[k]), 1e-30) for k in guess
    )


if __name__ == "__main__":
    run_example("full_waveform_inversion", full_waveform_inversion)
