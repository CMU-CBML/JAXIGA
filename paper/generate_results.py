"""Run all paper studies, or select one with --example.

    python paper/generate_results.py
    python paper/generate_results.py --example method_comparison

The implementations live in paper/experiments/. This module also re-exports
existing helpers for the paper's benchmark and figure-generation scripts.
See paper/README.md for the example index and output locations.
"""

import argparse
import sys
import time
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Compatibility imports: existing paper drivers use these names.
# ruff: noqa: F401
from paper.experiments._common import (
    ALL,
    FIGDATA,
    FIGURES,
    HERE,
    RESULTS,
    SEED,
    TABLES,
    _write_provenance_table,
    figure_data,
    provenance,
    rate,
    write_table,
)
from paper.experiments._fracture_common import (
    _CAPTURE_KEYS,
    _PLOT_DTYPE,
    _crack_panel,
    _crack_panel_data,
    _fracture_case,
    _fracture_mesh,
    _run_fracture,
    _summary_only,
)
from paper.experiments._plotting import (
    box_slice_lines,
    element_triangulation,
    mesh_lines,
    slice_pointset,
)
from paper.experiments.adaptive_fracture import (
    _adaptive_fracture_compute,
    _adaptive_fracture_figure,
    adaptive_fracture,
)
from paper.experiments.adaptive_lshape import (
    LSHAPE_ALPHA,
    _lshape_theta,
    _tail_slope,
    adaptive_lshape,
    lshape_exact,
    lshape_exact_grad,
    lshape_patches,
    lshape_problem,
)
from paper.experiments.capabilities import (
    capabilities,
)
from paper.experiments.darcy_inverse import (
    Darcy,
    N_MODES,
    darcy_inverse,
    log_perm,
)
from paper.experiments.differentiability import (
    differentiability,
)
from paper.experiments.fracture_3d import (
    _cube3d_crack_panel,
    _cube3d_crack_panel_data,
    _cube3d_figure,
    _fracture_3d_compute,
    fracture_3d,
)
from paper.experiments.full_waveform_inversion import (
    FWI_CKPT,
    FWI_DT,
    FWI_E0,
    FWI_FREQ,
    FWI_L,
    FWI_RHO,
    FWI_STEPS,
    FWI_T0,
    FWI_TRUE,
    FWI_WIDTH,
    _Bar,
    full_waveform_inversion,
)
from paper.experiments.implicit_differentiation import (
    _gauss_newton_gap,
    implicit_differentiation,
)
from paper.experiments.kirchhoff_plate import (
    PLATE_D,
    PLATE_E,
    PLATE_NU,
    PLATE_PARAMS,
    PLATE_Q,
    PLATE_T,
    kirchhoff_plate,
)
from paper.experiments.kirchhoff_love_shell import kirchhoff_love_shell
from paper.experiments.linear_solver_3d import (
    linear_solver_3d,
)
from paper.experiments.method_comparison import (
    method_comparison,
)
from paper.experiments.phase_field import (
    _CachedLU,
    _phase_field_compute,
    _phase_field_figure,
    phase_field,
)
from paper.experiments.plate_with_hole import (
    _default_route_time,
    _plate_fields_figure,
    _plate_work_precision,
    _rigid_modes,
    build_plate,
    fenicsx_reference,
    jaxfem_reference,
    kirsch,
    machine_timings,
    plate_adaptive,
    plate_exact,
    plate_l2_error,
    plate_problem,
    plate_space,
    plate_with_hole,
    rigid_aligned_difference,
    strain_energies,
)
from paper.experiments.poisson_convergence import (
    KWAVE,
    poisson_convergence,
    poisson_problem,
    poisson_solution_figure,
)
from paper.experiments.timings import (
    timings,
)
from paper.experiments.imported_geometry import imported_geometry

from paper.experiments._common import run_example, save_results

# Preserve the order of the original complete paper run.
EXAMPLES = (
    "poisson_convergence",
    "imported_geometry",
    "method_comparison",
    "plate_with_hole",
    "differentiability",
    "implicit_differentiation",
    "full_waveform_inversion",
    "darcy_inverse",
    "phase_field",
    "adaptive_fracture",
    "fracture_3d",
    "linear_solver_3d",
    "adaptive_lshape",
    "kirchhoff_plate",
    "kirchhoff_love_shell",
    "capabilities",
    "timings",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=EXAMPLES, help="run only this study")
    parser.add_argument("--list", action="store_true", help="list available studies")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(EXAMPLES))
        return 0
    if args.example:
        run_example(args.example, globals()[args.example], argv=[])
        return 0

    RESULTS.clear()
    start = time.perf_counter()
    for name in EXAMPLES:
        globals()[name]()
    path = save_results()
    print(f"\nall results in {time.perf_counter() - start:.0f}s -> "
          f"{TABLES}/, {FIGURES}/, {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
