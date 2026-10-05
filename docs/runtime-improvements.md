# Runtime improvements

The paper's EPYC/A100 results remain archived measurements. They have not been
replaced by timings from the CPU workstation used for this revision. The
roughly 0.19 s small-system cost is specific to the measured eager path, not a
hardware-wide lower bound. Cached fracture figures represent separate runs;
saved results and the manuscript are excluded from this public source tree.

## Implemented in this revision

- Matrix-free Galerkin statistics no longer construct the element-to-CSR scatter
  map just to report `nnz`. An unassembled operator reports `nnz=None`.
- The implicit correction avoids staging a conditional in eager execution when
  convergence is already concrete. Under `jit`, it keeps an array conditional.
- Failed iterative roots skip their correction solve. Convergence diagnostics
  make deliberately truncated solves visible instead of advertising their
  inaccurate gradients as valid performance results.
- The README demonstrates compiling the complete repeated-solve function.
  `paper/benchmark_execution.py` measures this path against eager calls without
  changing the paper's timing records.

## Local CPU result (14 September 2026)

On this ARM64 macOS workstation, cubic 2D Poisson solves with a fixed space,
varying diffusion coefficient, float64 and the default SciPy/SuperLU backend
produced the following medians over 15 synchronized warm repetitions:

| Total dofs | Eager | Whole-function `jit` | Speedup | Maximum solution difference |
|---|---:|---:|---:|---:|
| 121 | 24.50 ms | 1.08 ms | 22.7x | 1.0e-15 |
| 1,225 | 39.10 ms | 12.64 ms | 3.1x | 4.0e-15 |

Raw timing records are excluded from this public source tree. Use the commands
below to generate measurements on your machine.
The benefit decreases as sparse factorisation becomes a larger share of work.
First-call measurements follow a JAX cache clear within the same process; they
are not fresh-process startup measurements. These results compare execution
paths in this revised checkout, not old and new library versions. They do not
predict the A100 speedup or replace any paper result.

## Highest-value next changes

| Priority | Change | Where it helps | Implementation and validation |
|---|---|---|---|
| 1 | Reuse one compiled solve across a parameter sweep | CPU and GPU; especially small and medium systems | Construct the space/problem outside the loop, keep shapes stable, and pass changing arrays as arguments. Measure cold cost and synchronized warm samples separately. |
| 2 | Reuse constant-operator factorisations | CPU dynamics and multiple right-hand sides | `solvers/dynamics.py` assembles the effective matrix once, but the host sparse callback still factorises at each invocation. Introduce an explicit prepared solver scoped to unchanged matrix values; validate transpose solves and parameter gradients. Changing geometry, coefficients or time step must invalidate it. |
| 3 | Compute Jacobi diagonals directly | CPU/GPU matrix-free solves; high degree and 3D | `tangent_diagonal` still forms local tangent blocks before extracting their diagonals. Add matching diagonal kernels for built-in quadratic energies, retaining the generic fallback when energy is overridden. Compare diagonals, solutions, gradients and peak memory. |
| 4 | Improve preconditioning and reuse | Large GPU solves, refined meshes, damaged material | Profile iteration counts first. Evaluate block Jacobi or an appropriate multigrid preconditioner, with refresh rules for changing stiffness. Keep CG restricted to SPD operators; compare time at matched residual and gradient accuracy. |
| 5 | Use tensor contractions and bound assembly batches | CPU/GPU high-degree 3D | Apply sum factorisation where the basis/extraction permits it. Chunk local work and measure temporary allocations as well as final matrix storage. THB extraction needs separate benchmarking; tensor-product speedups cannot simply be assumed there. |
| 6 | Assemble collocation from structural connectivity | Larger collocation problems | Build the sparse Jacobian pattern from basis support, never from numerical nonzeros. Explore QR/LSQR/LSMR or operator-based steps instead of dense normal equations; preserve the exact stationarity Hessian for implicit derivatives and add globalisation for nonlinear problems. |
| 7 | Reduce adaptive recompilation and retained executables | Adaptive fracture, especially 3D GPU runs | Profile remesh setup, tracing, compilation and solve separately. Pass mesh-dependent numerical arrays into kernels, bound executable caches and avoid retaining obsolete closures. Bucketing/padding is useful only if saved compilation exceeds its extra work and memory. |

Factorisation reuse is already used within unchanged displacement sweeps of the
fracture driver; the second item targets the separate constant-operator
dynamics path. An assembled sparse solve can remain faster than matrix-free
AD: skipping the global matrix does not remove basis evaluation or local
preconditioning costs. The current A100 crossover of 3,000 unknowns should be
remeasured for each relevant device, degree, backend and conditioning regime.

## Reproducible measurements

Run from the repository root in an environment with JAXIGA installed:

```sh
python paper/benchmark_execution.py --refine 3 --repeats 15 --output tmp/benchmarks/execution-small.json
python paper/benchmark_execution.py --refine 5 --repeats 15 --output tmp/benchmarks/execution-medium.json
```

The script refuses to overwrite existing results. It records software versions,
device, revision/dirty state, a hash of the measured Python sources, thread
environment, first-call cost, individual
warm samples, medians and numerical agreement. Inputs vary in value while the
space remains fixed. The timing includes the public solve path and its default
linear backend; the jitted path may eliminate diagnostics unused by the returned
solution. No single-core affinity is imposed. Set `JAX_PLATFORMS=cpu` to select
CPU on a GPU host; use fresh filenames for accelerator measurements.

The warm timing synchronizes each output and separates initial compilation,
following the [JAX benchmarking guidance](https://docs.jax.dev/en/latest/201/profiling.html).
These are repeated fixed-space microbenchmarks, not replacements for the plate
or fracture experiments. Keep float64, discretisation, stopping tolerances and
solver choice explicit in comparisons. Accurate CG derivatives require both
primal and adjoint convergence, as specified in the
[JAX CG documentation](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.sparse.linalg.cg.html).

Before publishing new performance claims, also collect peak memory, compilation
counts, actual CPU affinity/thread configuration, raw repetitions and a source
snapshot or content hash. A commit with an unspecified dirty tree is insufficient
to reconstruct a run. For cross-code claims, measure matched error directly;
keep extrapolated comparisons explicitly labelled.
