"""Recover a spatial permeability field from pressure observations.

Run from the repository root::

    python -m paper.experiments.darcy_inverse

See paper/README.md for outputs, caching and the suggested reading order.
"""

import os
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

import jaxiga as jx
from jaxiga.solvers.optimize import minimize_lbfgs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import ALL, FIGURES, RESULTS, SEED, run_example


N_MODES = 3


def log_perm(x, theta):
    modes = jnp.array([
        jnp.cos(i * jnp.pi * x[0]) * jnp.cos(j * jnp.pi * x[1])
        for i in range(N_MODES) for j in range(N_MODES)
    ])
    return jnp.dot(theta, modes)


class Darcy(jx.Problem):
    is_linear = True

    def energy(self, grad_u, u, x, params):
        return 0.5 * jnp.exp(log_perm(x, params["log_k"])) * jnp.sum(grad_u**2)


def darcy_inverse():
    print("[5] Darcy inverse problem")
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(3))
    problem = Darcy(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)],
                    source=lambda x, p: 1.0)

    rng = np.random.default_rng(SEED)
    theta_true = jnp.asarray(rng.normal(scale=0.4, size=N_MODES**2))
    truth = jx.solve(problem, params={"log_k": theta_true})
    x_obs = jnp.asarray(rng.uniform(0.15, 0.85, size=(40, 2)))
    targets = jx.locate_points(V, x_obs)
    p_obs = truth.probe(x_obs, targets=targets)[:, 0]

    def loss(theta):
        sol = jx.solve(problem, params={"log_k": theta})
        return jnp.mean((sol.probe(x_obs, targets=targets)[:, 0] - p_obs) ** 2) \
            + 1e-6 * jnp.sum(theta**2)

    theta0 = jnp.zeros(N_MODES**2)
    t0 = time.perf_counter()
    res = minimize_lbfgs(jax.jit(loss), theta0, tol=1e-12, max_steps=300)
    elapsed = time.perf_counter() - t0

    grid = jnp.asarray([[i/40, j/40] for i in range(1, 40) for j in range(1, 40)])
    k_true = jnp.exp(jax.vmap(log_perm, in_axes=(0, None))(grid, theta_true))
    k_rec = jnp.exp(jax.vmap(log_perm, in_axes=(0, None))(grid, res.x))
    rel = float(jnp.linalg.norm(k_rec - k_true) / jnp.linalg.norm(k_true))
    print(f"    initial loss {float(loss(theta0)):.3e} -> {res.value:.3e} "
          f"in {res.steps} steps ({elapsed:.1f}s); field error {rel:.2e}")

    g = np.asarray(grid)
    n = 39
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), layout="constrained")
    kt = np.asarray(k_true).reshape(n, n)
    kr = np.asarray(k_rec).reshape(n, n)
    vmin, vmax = min(kt.min(), kr.min()), max(kt.max(), kr.max())
    for ax, field, title in zip(axes[:2], (kt, kr), ("true $k(x)$", "recovered $k(x)$")):
        im = ax.imshow(field.T, origin="lower", extent=[0, 1, 0, 1],
                       vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(title); ax.set_xlabel("$x$")
        plt.colorbar(im, ax=ax)
    axes[0].set_ylabel("$y$")
    im = axes[2].imshow(np.abs(kt - kr).T, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
    axes[2].scatter(np.asarray(x_obs)[:, 0], np.asarray(x_obs)[:, 1], s=6, c="w",
                    edgecolors="k", linewidths=0.3, label="measurements")
    axes[2].set_title("absolute error"); axes[2].set_xlabel("$x$")
    # below the panel: inside it the box would hide part of the error field
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8,
                   frameon=False, handletextpad=0.3)
    plt.colorbar(im, ax=axes[2])
    fig.savefig(os.path.join(FIGURES, "darcy_inverse.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/darcy_inverse.pdf")

    RESULTS["darcy"] = {"initial_loss": float(loss(theta0)), "final_loss": res.value,
                        "steps": res.steps, "field_rel_error": rel, "seconds": elapsed}


if __name__ == "__main__":
    run_example("darcy_inverse", darcy_inverse)
