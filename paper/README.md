# Paper examples

The studies used in the paper live in [`experiments/`](experiments/), with one
runnable module per study. Each file contains the problem setup, solver calls,
checks and figure/table generation. Shared output and plotting utilities are in
the files whose names start with `_`.

Install JAXIGA and Matplotlib from the repository root:

```bash
python -m pip install -e . matplotlib
```

Run a single example from the repository root:

```bash
python -m paper.experiments.method_comparison
```

The existing driver can also select an example, or run every study:

```bash
python paper/generate_results.py --list
python paper/generate_results.py --example method_comparison
python paper/generate_results.py
```

`--help` shows usage without running a study. To run from another directory,
use the absolute path to `paper/generate_results.py`. If you have not installed
the library, set `PYTHONPATH` to the absolute path of the repository's `src/`.

## Suggested reading order

Start with `poisson_problem` in
[`poisson_convergence.py`](experiments/poisson_convergence.py). It shows the
basic sequence: create a geometry, build a function space, prescribe boundary
conditions and define a source. For a small first solve, run this from the
repository root:

```python
import jaxiga as jx
from paper.experiments.poisson_convergence import poisson_problem

problem, exact = poisson_problem(dim=2, deg=2, refine=2)
solution = jx.solve(problem, params={"a0": 1.0})
print("Relative L2 error:", jx.errornorm(solution, exact, "L2"))
```

Then read `method_comparison.py` to see how the same problem can use three
solution methods. The inverse and adaptive examples build on that workflow.
Importing an example does not execute its study.

| Module in `paper.experiments` | What it demonstrates |
| --- | --- |
| [`poisson_convergence`](experiments/poisson_convergence.py) | Manufactured solutions and convergence in 1D, 2D and 3D |
| [`imported_geometry`](experiments/imported_geometry.py) | All-quad perforated CAD plate and all-hex native NURBS connecting rod; mesh and cell-average stress |
| [`method_comparison`](experiments/method_comparison.py) | Galerkin, energy minimisation and collocation on the same problem |
| [`plate_with_hole`](experiments/plate_with_hole.py) | Linear elasticity, the Kirsch solution, adaptivity and independent reference solvers |
| [`differentiability`](experiments/differentiability.py) | Material, source and geometry gradients checked against finite differences |
| [`implicit_differentiation`](experiments/implicit_differentiation.py) | Adjoint gradients across methods, including nonlinear collocation |
| [`darcy_inverse`](experiments/darcy_inverse.py) | Recovering permeability from pressure measurements |
| [`full_waveform_inversion`](experiments/full_waveform_inversion.py) | Recovering a stiffness inclusion through a time integration loop |
| [`adaptive_lshape`](experiments/adaptive_lshape.py) | Local refinement around a corner singularity |
| [`kirchhoff_plate`](experiments/kirchhoff_plate.py) | Fourth-order plate bending with smooth splines |
| [`kirchhoff_love_shell`](experiments/kirchhoff_love_shell.py) | Kirchhoff–Love shell: flat-plate consistency and the Scordelis–Lo roof |
| [`capabilities`](experiments/capabilities.py) | Matrix-free and chunked assembly, batching and dynamics |
| [`phase_field`](experiments/phase_field.py) | Brittle fracture in a notched plate under tension |
| [`adaptive_fracture`](experiments/adaptive_fracture.py) | Crack-following refinement compared with predefined meshes |
| [`fracture_3d`](experiments/fracture_3d.py) | A notched 3D specimen and its two-dimensional limits |
| [`linear_solver_3d`](experiments/linear_solver_3d.py) | Sparse direct versus conjugate-gradient solvers |
| [`timings`](experiments/timings.py) | Setup, solve, compilation and differentiation costs |

These modules retain the paper's mesh sizes and parameter sweeps. A full study
can take substantially longer than the small solve above. Fresh fracture runs
can take hours on a GPU, and the 3D cases need substantial memory.

## Outputs and cached results

All output locations are relative to `paper/`, regardless of the working
directory:

- `tables/*.tex` and `figures/*.pdf`: the study's paper tables and figures.
- `results/<example>.json`: results and environment provenance from a single
  example, written by either single-example command above. These local run
  summaries are ignored by Git.
- `results_summary.json` and `tables/provenance.tex`: written only by a complete
  run of `generate_results.py`.
- `figdata/*.json` and `figdata/*.npz`: locally generated simulation results and plotting
  arrays. No saved results are bundled; the first run computes them. Later
  runs reuse them unless `JAXIGA_RECOMPUTE_FIGURES=1` is set.

Individual examples regenerate their own tables and figures, but leave the
complete paper summary and its provenance table intact. To recompute a fracture
study rather than use its cached data:

```bash
JAXIGA_RECOMPUTE_FIGURES=1 python -m paper.experiments.adaptive_fracture
```

The plate-with-hole comparison requires FEniCSx and JAX-FEM results in
`fenicsx_reference.json` and `jaxfem_reference.json`. Generate these using
`reference_fenicsx.py` and `reference_jaxfem.py` in their separate environments,
or set `JAXIGA_FENICSX_PYTHON` and `JAXIGA_JAXFEM_PYTHON` when running the study.
See the [repository README](../README.md#testing) for the reference environments.
`reference_machine.py` generates the CPU/GPU timing results automatically when
`machine_timings.json` is absent; this full comparison requires a CUDA GPU.
Set `JAXIGA_REMEASURE_MACHINE=1` to repeat the measurements.
The remaining studies can run independently through their individual modules.
The manuscript, saved benchmark results and generated figures are excluded.

The imported-geometry figure replaces the cube solution illustration in the
manuscript; the Poisson convergence study remains. Replot it with
`python -m paper.experiments.imported_geometry`. Its locally generated `figdata` records
include plotting arrays, geometry/solver settings and environment provenance.
Use `JAXIGA_RECOMPUTE_FIGURES=1` to regenerate them (requires the `gmsh` extra;
on macOS, use `OMP_NUM_THREADS=1`). Without cached data, the first run computes the solution and requires the
`gmsh` extra. The figure uses only quad/hex analysis cells.
The standalone library examples additionally export VTK for ParaView.

The older paper drivers can still import helpers from `generate_results`.
For new code, import them directly from the relevant experiment module.
