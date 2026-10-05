"""Parametric sweep and forward UQ: many solves as one batched program.

``vmap`` maps a solve over a batch of parameters. That is a capability the
architecture gets for free and that a conventional solver cannot offer: the
sparsity pattern is static and shared across the batch, so only the matrix
*values* and the right-hand side carry a batch axis, and the whole sweep
becomes one XLA program instead of a Python loop over independent solves.

Two uses are shown:

1. a deterministic sweep over a material parameter;
2. forward uncertainty quantification -- push a sample of random material
   properties through the solver and read off the distribution of a
   quantity of interest.

Applicability. Batching covers *linear* problems: the Newton loop makes Python
decisions on residual norms, which a batch tracer cannot answer. Every linear
backend batches, including ``method="scipy"`` -- its host callback runs once
per batch member rather than in parallel, so it is correct but gains no speed.
``"dense"`` and the iterative BCOO paths batch properly.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp

import jaxiga as jx

DEG, REFINE = 2, 3
E_MEAN, E_SD, NU = 1.0e5, 1.0e4, 0.3
LOAD, N_SAMPLES = 100.0, 256

patch = jx.primitives.quadrilateral(
    [[0.0, 0.0], [10.0, 0.0], [10.0, 1.0], [0.0, 1.0]]
).elevate(DEG).refine(REFINE)
V = jx.FunctionSpace(patch, vec=2)
print(V.summary())

problem = jx.LinearElasticity(
    V, plane="stress",
    dirichlet=[jx.DirichletBC([0.0, 0.0], where="left")],
    neumann=[jx.Neumann(lambda x, n, p: jnp.array([0.0, -LOAD]), where="right")],
)
tip = np.array([[10.0, 0.5]])
# Locating which element holds a physical point is a host-side search, so it
# cannot run inside jit or vmap. The geometry does not depend on the parameter
# being swept, so locate once here and pass the result in.
tip_target = jx.locate_points(V, tip)


def tip_deflection(E):
    """Quantity of interest: downward tip displacement of the cantilever."""
    sol = jx.solve(problem, params={"E": E, "nu": NU},
                   linear=jx.LinearOptions(method="dense"))
    return -sol.probe(tip, targets=tip_target)[0, 1]


# -- 1. deterministic sweep ------------------------------------------------
values = jnp.linspace(0.5 * E_MEAN, 2.0 * E_MEAN, 16)

t0 = time.perf_counter()
loop = jnp.stack([tip_deflection(e) for e in values])
t_loop = time.perf_counter() - t0

batched = jax.jit(jax.vmap(tip_deflection))
batched(values[:2]).block_until_ready()  # warm up the compilation
t0 = time.perf_counter()
batch = batched(values).block_until_ready()
t_batch = time.perf_counter() - t0

print(f"\nsweep over {len(values)} stiffness values")
print(f"  max |batch - loop| = {float(jnp.abs(batch - loop).max()):.3e}")
print(f"  python loop {t_loop:.3f}s, batched {t_batch:.3f}s "
      f"({t_loop / t_batch:.1f}x measured speedup)")
print("  E / E_mean    tip deflection    E * deflection (should be constant)")
for e, d in zip(values[::5], batch[::5]):
    print(f"    {float(e) / E_MEAN:.3f}       {float(d):.6e}      {float(e * d):.6e}")

# -- 2. forward uncertainty quantification ---------------------------------
key = jax.random.PRNGKey(0)
samples = E_MEAN + E_SD * jax.random.normal(key, (N_SAMPLES,))
t0 = time.perf_counter()
qoi = batched(samples).block_until_ready()
print(f"\nforward UQ: {N_SAMPLES} samples in {time.perf_counter() - t0:.3f}s")
print(f"  E     ~ N({E_MEAN:.3g}, {E_SD:.3g})  ->  {float(jnp.mean(samples)):.5g} "
      f"+- {float(jnp.std(samples)):.4g}")
print(f"  tip deflection  mean {float(jnp.mean(qoi)):.6e}  sd {float(jnp.std(qoi)):.4e}")
q = np.percentile(np.asarray(qoi), [5, 50, 95])
print(f"  5th / 50th / 95th percentile: {q[0]:.6e}  {q[1]:.6e}  {q[2]:.6e}")

# The response is 1/E, so first-order propagation predicts the spread; the gap
# is the curvature of 1/E, which the sampling captures and linearization misses.
linear_sd = float(jnp.mean(qoi)) * E_SD / E_MEAN
print(f"  first-order estimate of the sd: {linear_sd:.4e} "
      f"({100 * (float(jnp.std(qoi)) / linear_sd - 1):+.1f}% vs sampled)")

np.savetxt("uq_samples.csv", np.column_stack([np.asarray(samples), np.asarray(qoi)]),
           delimiter=",", header="E,tip_deflection", comments="")
print("wrote uq_samples.csv")
