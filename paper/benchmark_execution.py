"""Compare eager and fully jitted repeated solves without changing paper data.

Example (CPU; choose a new output filename for each measurement):
    python paper/benchmark_execution.py --output tmp/benchmarks/execution.json

Run with JAX_PLATFORMS=cpu to select the CPU on a machine with a GPU installed.
The script also supports accelerators. Cold costs include compilation; warm
samples are synchronized.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy

import jaxiga as jx

ROOT = Path(__file__).resolve().parents[1]


def source_hash():
    """Identify the measured Python sources even in an uncommitted checkout."""
    digest = hashlib.sha256()
    paths = [*sorted((ROOT / "src").rglob("*.py")), Path(__file__).resolve()]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def measure(degree, refine, repeats):
    V = jx.FunctionSpace(jx.primitives.rectangle(0, 0, 1, 1).elevate(degree).refine(refine))
    problem = jx.Poisson(
        V, dirichlet=[jx.DirichletBC(0., where=lambda x: jnp.ones(x.shape[0], bool))],
        source=lambda x, p: 2*jnp.pi**2*jnp.sin(jnp.pi*x[0])*jnp.sin(jnp.pi*x[1]),
    )
    solve = lambda a: jx.solve(problem, params={"a0": a}).u
    values = [jnp.asarray(1.0 + 0.1*i) for i in range(repeats)]
    result, answers = {}, {}
    for name, fn in (("eager", solve), ("jit", jax.jit(solve))):
        jax.clear_caches()
        start = time.perf_counter()
        first = fn(values[0]).block_until_ready()
        cold = time.perf_counter() - start
        samples = []
        for a in values:
            start = time.perf_counter()
            answer = fn(a).block_until_ready()
            samples.append(time.perf_counter() - start)
        answers[name] = np.asarray(first)
        result[name] = {"first_call_s": cold, "warm_samples_s": samples,
                        "warm_median_s": float(np.median(samples))}
    discrepancy = float(np.max(np.abs(answers["eager"]-answers["jit"])))
    np.testing.assert_allclose(answers["eager"], answers["jit"], rtol=1e-10, atol=1e-12)
    result.update(degree=degree, refine=refine, dofs=V.n_dofs,
                  solution_max_abs_difference=discrepancy,
                  warm_speedup=result["eager"]["warm_median_s"]/result["jit"]["warm_median_s"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--refine", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new path to preserve previous measurements")
    if args.repeats < 1 or args.degree < 2 or args.refine < 0:
        parser.error("repeats >= 1, degree >= 2 and refine >= 0 are required")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    environment = {
        "commit": revision.stdout.strip(), "working_tree_dirty": bool(dirty.stdout.strip()),
        "source_sha256": source_hash(),
        "python": platform.python_version(), "platform": platform.platform(),
        "jax": jax.__version__, "jaxlib": __import__("jaxlib").__version__,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")},
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "fixed space, varying coefficient; default linear backend; synchronized output; no single-core affinity imposed",
    }
    record = {"environment": environment, "measurement": measure(args.degree, args.refine, args.repeats)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(record, f, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
