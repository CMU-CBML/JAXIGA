"""Differentiate through elasticity and Poisson solves and check parameter and shape gradients.

Run from the repository root::

    python -m paper.experiments.differentiability

See paper/README.md for outputs, caching and the suggested reading order.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.solvers.linear import SOLVE_COUNTER, reset_solve_counter

from ._common import ALL, RESULTS, run_example, write_table
from .plate_with_hole import build_plate


def differentiability():
    print("[4] Differentiability")
    rows = []

    # (a) material parameter
    problem, _, _ = build_plate(2, 1)

    def compliance(theta):
        return jnp.sum(jx.solve(problem, params={"E": theta[0], "nu": theta[1]}).u ** 2)

    theta = jnp.array([1e5, 0.3])
    g = jax.grad(compliance)(theta)
    for i, (name, h) in enumerate([("material $E$", 1.0), (r"material $\nu$", 1e-6)]):
        fd = (compliance(theta.at[i].add(h)) - compliance(theta.at[i].add(-h))) / (2 * h)
        rows.append((name, float(g[i]), float(fd)))

    # (b) source amplitude
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(2))
    pr = jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where=ALL)],
                    source=lambda x, p: p["amp"] * jnp.sin(jnp.pi*x[0]) * jnp.sin(jnp.pi*x[1]))

    def src_loss(a):
        return jnp.sum(jx.solve(pr, params={"a0": 1.0, "amp": a}).u ** 2)

    h = 1e-5
    rows.append(("source amplitude", float(jax.grad(src_loss)(3.0)),
                 float((src_loss(3.0 + h) - src_loss(3.0 - h)) / (2 * h))))

    # (c) geometry (control points)
    import dataclasses

    Vg = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(2).refine(1))
    base = jx.Poisson(Vg, dirichlet=[jx.DirichletBC(0.0, where=ALL)],
                      source=lambda x, p: 1.0)

    def geo(cpts):
        return jnp.sum(jx.solve(base.with_space(dataclasses.replace(Vg, cpts=cpts)),
                                params={"a0": 1.0}).u ** 2)

    flat = Vg.cpts.reshape(-1)
    gg = np.asarray(jax.grad(geo)(Vg.cpts)).reshape(-1)
    i = 5
    fd = float((geo((flat.at[i].add(1e-6)).reshape(Vg.cpts.shape))
                - geo((flat.at[i].add(-1e-6)).reshape(Vg.cpts.shape))) / 2e-6)
    rows.append(("control point", float(gg[i]), fd))

    # adjoint solve count
    reset_solve_counter()
    jax.grad(src_loss)(3.0)
    n_solves = dict(SOLVE_COUNTER)

    lines = [
        r"\begin{tabular}{l l l l}", r"\hline",
        r"differentiated w.r.t. & adjoint & central differences & rel. difference \\",
        r"\hline",
    ]
    for name, ad, fd in rows:
        rel = abs(ad - fd) / max(abs(fd), 1e-30)
        lines.append(f"{name} & {ad:.8e} & {fd:.8e} & {rel:.1e} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    write_table("gradient_verification", "\n".join(lines))
    print(f"    solves per gradient: {n_solves}")
    RESULTS["solves_per_gradient"] = n_solves
    RESULTS["max_grad_rel_error"] = max(
        abs(a - f) / max(abs(f), 1e-30) for _, a, f in rows
    )


if __name__ == "__main__":
    run_example("differentiability", differentiability)
