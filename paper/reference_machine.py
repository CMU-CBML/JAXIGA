"""Plate-with-a-hole timings on the machine and backend this is launched with.

Run once per backend; the backend is selected before JAX is imported, since a
process that has already initialised CUDA cannot be talked out of it:

    python3 reference_machine.py out.json --backend gpu
    python3 reference_machine.py out.json --backend cpu

The sweep is degrees two to four over the four refinements the cross-code
comparison uses, with degree four carried two levels further, to $36{,}448$
unknowns, where the two solver routes separate.

Each level is timed twice: once with the library default, Jacobi-preconditioned
conjugate gradients on the device, and once with the sparse LU it replaced,
which is a host callback on either backend. Reporting both is what lets the
figure show that the accelerator is worth having only once the solve stops
leaving the device --- the direct route is within a few percent of its CPU
timing no matter which backend assembles the matrix.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
parser.add_argument("--max-refine", type=int, default=6)
args = parser.parse_args()

# Before any JAX import: on this machine the CUDA plugin is installed, so the
# CPU run has to be asked for explicitly rather than merely not asked for.
if args.backend == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jaxiga as jx
from jaxiga.methods._common import build_context
from jaxiga.methods.galerkin import assemble_csr_values, default_linear
from jaxiga.solvers.linear import LinearOptions

from generate_results import build_plate, plate_l2_error

PARAMS = {"E": 1e5, "nu": 0.3}
REPEATS = 3
DIRECT = LinearOptions(method="scipy")


def median_of(fn, repeats=REPEATS):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def timed_solve(problem, linear):
    """Median warm solve time, tracing and compilation excluded."""
    jx.solve(problem, params=PARAMS, linear=linear).u.block_until_ready()
    return median_of(
        lambda: jx.solve(problem, params=PARAMS, linear=linear).u.block_until_ready()
    )


def cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def gpu_model():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()
        return out[0] if out else "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# Every degree is carried the full range: the interesting part of the curve is
# where the system is large enough for the iterative route to pay, and stopping
# the lower degrees at the cross-code range would cut them off before it.
levels = [(deg, r) for deg in (2, 3, 4) for r in range(1, args.max_refine + 1)]

rows = []
for deg, refine in levels:
    problem, exact_disp, exact_stress = build_plate(deg, refine)
    ctx = build_context(problem, PARAMS)

    assemble_csr_values(ctx, PARAMS).block_until_ready()
    t_assemble = median_of(
        lambda: assemble_csr_values(ctx, PARAMS).block_until_ready()
    )

    t_default = timed_solve(problem, default_linear(problem))
    t_direct = timed_solve(problem, DIRECT)

    sol = jx.solve(problem, params=PARAMS)
    row = {
        "deg": deg,
        "refine": refine,
        "dofs": int(problem.space.n_dofs),
        "elems": int(problem.space.n_elems),
        "nnz": int(ctx.assembly.nnz),
        "l2": plate_l2_error(sol, exact_disp),
        "energy": float(
            jx.errornorm(sol, exact_disp, "energy", exact_grad=exact_stress)
        ),
        "time": t_default,
        "time_direct": t_direct,
        "assemble": t_assemble,
    }
    rows.append(row)
    print(f"    p={deg} r={refine}: dofs {row['dofs']:6d}  CG {t_default:7.3f}s  "
          f"direct {t_direct:8.3f}s  assemble {t_assemble:7.4f}s  "
          f"L2 {row['l2']:.4e}", flush=True)

devices = jax.devices()
info = {
    "backend": jax.default_backend(),
    "devices": [str(d) for d in devices],
    "device_name": gpu_model() if jax.default_backend() == "gpu" else cpu_model(),
    "cpu": cpu_model(),
    "cpu_count": os.cpu_count(),
    "jaxiga": jx.__version__,
    "jax": jax.__version__,
    "jaxlib": getattr(__import__("jaxlib"), "__version__", "unknown"),
    "numpy": np.__version__,
    "scipy": __import__("scipy").__version__,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "x64_enabled": bool(jax.config.jax_enable_x64),
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

with open(args.out, "w") as fh:
    json.dump({"versions": info, "runs": rows}, fh, indent=2, default=float)
print(f"  wrote {args.out} ({info['backend']}, {info['device_name']})")
