"""Shared output paths, cached data, result records and provenance."""

import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

import jaxiga as jx

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Outputs remain under paper/, independent of the working directory.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = os.path.join(HERE, "tables")
FIGURES = os.path.join(HERE, "figures")
FIGDATA = os.path.join(HERE, "figdata")
os.makedirs(TABLES, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(FIGDATA, exist_ok=True)

# Every random draw in the paper studies comes from this seed.
SEED = 20260901


def provenance():
    """Everything needed to interpret a difference in the numbers below.

    Recorded rather than described: floating-point results depend on the JAX and
    XLA versions, on the backend and device, and on 64-bit mode being on, and a
    reader comparing against a rerun needs to know which of those changed.
    """
    import platform
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "-C", HERE, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", HERE, "status", "--porcelain"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
    except Exception:
        commit, dirty = "unknown", None

    devices = jax.devices()
    return {
        "jaxiga": jx.__version__,
        "commit": commit,
        "working_tree_dirty": dirty,
        "jax": jax.__version__,
        "jaxlib": getattr(__import__("jaxlib"), "__version__", "unknown"),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in devices],
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "seed": SEED,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

ALL = lambda x: jnp.full(x.shape[0], True)
RESULTS = {}

def figure_data(name, compute):
    """Compute and cache plot data separately from drawing a figure.

    The solve writes JSON records and compressed arrays to ``paper/figdata/``.
    Later runs reuse those files unless ``JAXIGA_RECOMPUTE_FIGURES`` is set.
    No saved data is bundled with the public source tree. Separating solving
    from plotting lets a figure be redrawn without repeating the simulation.

    ``compute`` returns ``(record, arrays)``. ``record`` is JSON -- scalars,
    labels and the load--displacement curves, the things the paper quotes and
    a reviewer would want to read in a diff. ``arrays`` is a dict of NumPy
    arrays for the bulk field data, which is not reviewable as text and goes
    to a compressed ``.npz`` instead.
    """
    record_path = os.path.join(FIGDATA, name + ".json")
    array_path = os.path.join(FIGDATA, name + ".npz")
    stale = not (os.path.exists(record_path) and os.path.exists(array_path))
    if os.environ.get("JAXIGA_RECOMPUTE_FIGURES") or stale:
        if compute is None:
            raise RuntimeError(
                f"no cached figure data for {name!r} and nothing to compute it "
                f"with: run generate_results.py to produce {record_path}"
            )
        record, arrays = compute()
        # Serialise before writing either file, so a failure cannot leave the
        # pair half-updated and disagreeing with each other.
        payload = json.dumps(record, indent=2, default=float)
        np.savez_compressed(array_path, **arrays)
        with open(record_path, "w") as fh:
            fh.write(payload)
        print(f"    wrote {os.path.relpath(record_path, HERE)} and "
              f"{os.path.relpath(array_path, HERE)}")
        return record, arrays
    with open(record_path) as fh:
        record = json.load(fh)
    with np.load(array_path) as fh:
        arrays = {k: fh[k] for k in fh.files}
    print(f"    using cached {os.path.relpath(record_path, HERE)} "
          f"(set JAXIGA_RECOMPUTE_FIGURES to re-run the solve)")
    return record, arrays


def write_table(name, text):
    with open(os.path.join(TABLES, name + ".tex"), "w") as fh:
        fh.write(text)
    print(f"  wrote {TABLES}/{name}.tex")


def rate(errs):
    return [np.nan] + [np.log2(a / b) for a, b in zip(errs, errs[1:])]


def _write_provenance_table(info):
    """A LaTeX fragment recording the environment the numbers came from."""
    rows = [
        ("JAXIGA", f"{info['jaxiga']} (commit \\texttt{{{info['commit'][:12]}}}"
                   + (", dirty tree)" if info["working_tree_dirty"] else ")")),
        ("JAX / jaxlib", f"{info['jax']} / {info['jaxlib']}"),
        ("NumPy / SciPy", f"{info['numpy']} / {info['scipy']}"),
        ("Python", info["python"]),
        ("Platform", info["platform"].replace("_", "\\_")),
        ("Backend, device", f"{info['backend']}, {info['devices'][0]}"),
        ("Precision", "float64" if info["x64_enabled"] else "float32"),
        ("Random seed", str(info["seed"])),
    ]
    fx = RESULTS.get("fenicsx_versions")
    if fx:
        rows.insert(3, ("DOLFINx / basix / UFL",
                        f"{fx['dolfinx']} / {fx['basix']} / {fx['ufl']} "
                        f"(Python {fx['python']})"))
    mach = RESULTS.get("machine_versions")
    if mach:
        # Record benchmark devices separately from the current process backend.
        gpu, cpu = mach["gpu"], mach["cpu"]
        rows.append(("Benchmark machine, CPU",
                     cpu["device_name"].replace("_", "\\_")))
        rows.append(("\\quad accelerator", f"{gpu['device_name']} (CUDA)"))
    lines = [r"\begin{tabular}{l l}", r"\hline"]
    lines += [f"{k} & {v} \\\\" for k, v in rows]
    lines += [r"\hline", r"\end{tabular}"]
    write_table("provenance", "\n".join(lines))


def save_results(name=None):
    """Save a complete run, or an isolated example, with its environment.

    Individual runs go to results/<name>.json so they cannot replace the
    complete paper summary or its provenance table.
    """
    RESULTS["provenance"] = provenance()
    # Encode before opening: an unserialisable value must not truncate a
    # previously completed run's summary.
    payload = json.dumps(RESULTS, indent=2, default=float)
    if name is None:
        path = os.path.join(HERE, "results_summary.json")
    else:
        directory = os.path.join(HERE, "results")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name + ".json")
    with open(path, "w") as fh:
        fh.write(payload)
    if name is None:
        _write_provenance_table(RESULTS["provenance"])
    return path


def run_example(name, compute, argv=None):
    """Command-line entry point shared by the independently runnable studies."""
    import argparse

    parser = argparse.ArgumentParser(
        description=compute.__doc__ or f"Run the {name} paper example.",
        epilog="See paper/README.md for output files and cache controls.",
    )
    parser.parse_args(argv)
    RESULTS.clear()
    start = time.perf_counter()
    compute()
    path = save_results(name)
    print(f"\n{name} in {time.perf_counter() - start:.0f}s -> {path}")
