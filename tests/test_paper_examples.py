"""Paper entry points, shared results and cached figures after modularisation."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def paper(monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(ROOT))
    driver = importlib.import_module("paper.generate_results")
    driver.RESULTS.clear()
    yield driver
    driver.RESULTS.clear()


def test_legacy_driver_imports_share_helpers_and_results(paper, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "paper"))
    legacy = importlib.import_module("generate_results")
    common = importlib.import_module("paper.experiments._common")
    assert legacy.RESULTS is paper.RESULTS is common.RESULTS
    for name, module in (
        ("build_plate", "plate_with_hole"),
        ("plate_l2_error", "plate_with_hole"),
        ("_run_fracture", "_fracture_common"),
        ("_fracture_case", "_fracture_common"),
        ("_phase_field_figure", "phase_field"),
        ("_adaptive_fracture_figure", "adaptive_fracture"),
        ("_cube3d_figure", "fracture_3d"),
    ):
        example = importlib.import_module(f"paper.experiments.{module}")
        assert getattr(legacy, name) is getattr(example, name)
    assert Path(common.HERE) == ROOT / "paper"


def test_selected_example_keeps_complete_summary(paper, monkeypatch, tmp_path):
    common = importlib.import_module("paper.experiments._common")
    monkeypatch.setattr(common, "HERE", str(tmp_path))
    monkeypatch.setattr(common, "provenance", lambda: {"backend": "test"})
    full_summary = tmp_path / "results_summary.json"
    full_summary.write_text('{"previous": true}')
    monkeypatch.setattr(
        paper, "method_comparison", lambda: paper.RESULTS.update(agreement=1e-12)
    )
    paper.RESULTS["stale"] = 1

    assert paper.main(["--example", "method_comparison"]) == 0

    result = json.loads((tmp_path / "results/method_comparison.json").read_text())
    assert result == {"agreement": 1e-12, "provenance": {"backend": "test"}}
    assert full_summary.read_text() == '{"previous": true}'


def test_complete_run_preserves_order_and_aggregates(paper, monkeypatch, tmp_path):
    common = importlib.import_module("paper.experiments._common")
    monkeypatch.setattr(common, "HERE", str(tmp_path))
    monkeypatch.setattr(common, "provenance", lambda: {"backend": "test"})
    provenance_tables = []
    monkeypatch.setattr(common, "_write_provenance_table", provenance_tables.append)
    calls = []
    for name in paper.EXAMPLES:
        def compute(name=name):
            calls.append(name)
            paper.RESULTS[name] = len(calls)
        monkeypatch.setattr(paper, name, compute)

    assert paper.main([]) == 0

    assert calls == list(paper.EXAMPLES)
    result = json.loads((tmp_path / "results_summary.json").read_text())
    assert list(result) == list(paper.EXAMPLES) + ["provenance"]
    assert provenance_tables == [{"backend": "test"}]


def test_unserialisable_result_preserves_previous_summary(paper, monkeypatch, tmp_path):
    common = importlib.import_module("paper.experiments._common")
    monkeypatch.setattr(common, "HERE", str(tmp_path))
    monkeypatch.setattr(common, "provenance", lambda: {})
    path = tmp_path / "results_summary.json"
    path.write_text('{"previous": true}')
    paper.RESULTS["live_solver_object"] = object()
    with pytest.raises(TypeError):
        common.save_results()
    assert path.read_text() == '{"previous": true}'


@pytest.mark.parametrize("name,filename", [
    ("phase_field", "phase_field.pdf"),
    ("adaptive_fracture", "adaptive_fracture.pdf"),
    ("fracture_3d", "cube3d.pdf"),
])
def test_cached_example_runs_without_solving(paper, monkeypatch, tmp_path, name, filename):
    common = importlib.import_module("paper.experiments._common")
    example = importlib.import_module(f"paper.experiments.{name}")
    # Saved simulation outputs are intentionally absent from the public tree.
    if not all((Path(common.FIGDATA) / f"{name}{suffix}").exists()
               for suffix in (".json", ".npz")):
        pytest.skip("requires locally generated figure data")
    monkeypatch.delenv("JAXIGA_RECOMPUTE_FIGURES", raising=False)
    monkeypatch.setattr(example, "FIGURES", str(tmp_path))
    monkeypatch.setattr(common, "HERE", str(tmp_path))
    monkeypatch.setattr(common, "provenance", lambda: {"backend": "test"})
    monkeypatch.setattr(
        example, f"_{name}_compute", lambda: pytest.fail("cached data should avoid solving")
    )

    common.run_example(name, getattr(example, name), argv=[])

    assert (tmp_path / filename).read_bytes().startswith(b"%PDF")
    result = json.loads((tmp_path / "results" / f"{name}.json").read_text())
    cached = json.loads((Path(common.FIGDATA) / f"{name}.json").read_text())
    assert result[name] == cached["results"]


def test_cli_entry_points(paper, tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "src")]))
    for command, expected in (
        ([str(ROOT / "paper/generate_results.py"), "--list"], "poisson_convergence"),
        (["-m", "paper.experiments.method_comparison", "--help"], "usage:"),
    ):
        result = subprocess.run(
            [sys.executable, *command], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=60, check=True,
        )
        assert expected in result.stdout
        assert "RuntimeWarning" not in result.stderr


def test_poisson_setup_solves_and_converges(paper):
    import jaxiga as jx

    errors = []
    for refine in (2, 3):
        problem, exact = paper.poisson_problem(dim=1, deg=2, refine=refine)
        solution = jx.solve(problem, params={"a0": 1.0})
        errors.append(float(jx.errornorm(solution, exact, "L2")))
    assert 0 < errors[1] < errors[0] / 4
