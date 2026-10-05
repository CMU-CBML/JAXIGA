"""Verify linear elasticity against the Kirsch solution and independent solvers.

Run from the repository root::

    python -m paper.experiments.plate_with_hole

See paper/README.md for outputs, caching and the suggested reading order.
"""

import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

import jaxiga as jx
from jaxiga.methods._common import build_context
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import FIGURES, HERE, RESULTS, run_example
from ._plotting import element_triangulation, mesh_lines


def kirsch(rad=1.0, trac=10.0, Emod=1e5, nu=0.3):
    """The Kirsch field: an infinite plate with a circular hole under tension.

    Returns ``(disp_np, stress_np, stress_jnp, disp_jnp)``. Both stress
    functions are vectorised and return Voigt order ``(sxx, syy, sxy)`` along
    the last axis.
    """

    def _stress(x, y, lib):
        r, th = lib.hypot(x, y), lib.arctan2(y, x)
        c2, s2 = lib.cos(2*th), lib.sin(2*th)
        srr = trac/2*(1-rad**2/r**2) + trac/2*(1-4*rad**2/r**2+3*rad**4/r**4)*c2
        stt = trac/2*(1+rad**2/r**2) - trac/2*(1+3*rad**4/r**4)*c2
        srt = -trac/2*(1+2*rad**2/r**2-3*rad**4/r**4)*s2
        # polar -> Cartesian in closed form rather than a 3x3 solve per point,
        # so the whole field evaluates in one vectorised pass
        c, s = lib.cos(th), lib.sin(th)
        return lib.stack([srr*c**2 + stt*s**2 - 2*srt*s*c,
                          srr*s**2 + stt*c**2 + 2*srt*s*c,
                          (srr - stt)*s*c + srt*(c**2 - s**2)], axis=-1)

    def _disp(x, y, lib):
        r, th = lib.hypot(x, y), lib.arctan2(y, x)
        ux = ((1+nu)/Emod*trac*(r*lib.cos(th)/(1+nu) + 2*rad**2*lib.cos(th)/((1+nu)*r)
              + rad**2*lib.cos(3*th)/(2*r) - rad**4*lib.cos(3*th)/(2*r**3)))
        uy = ((1+nu)/Emod*trac*(-nu*r*lib.sin(th)/(1+nu)
              - (1-nu)*rad**2*lib.sin(th)/((1+nu)*r)
              + rad**2*lib.sin(3*th)/(2*r) - rad**4*lib.sin(3*th)/(2*r**3)))
        return ux, uy

    return (lambda x, y: _disp(x, y, np),
            lambda x, y: _stress(x, y, np),
            lambda x, y: _stress(x, y, jnp),
            lambda x, y: jnp.stack(_disp(x, y, jnp)))


def plate_space(deg, refine, rad=1.0, side=4.0, vec=2, extra_u=0):
    """The four-sector NURBS discretisation of the plate, cut along the diagonals.

    Each patch is a $90^\\circ$ arc of the hole opposite one straight side of
    the square, so neither of its curved-to-straight boundaries has a corner in
    its interior and no repeated interior knot is needed: the knot vectors are
    ``[0,0,0,1,1,1]`` by ``[0,0,1,1]``. Cutting along the axes instead would put
    a corner of the square in the middle of each patch's outer edge, forcing a
    repeated knot and a $C^0$ line across every patch.
    """
    patches = []
    for k in range(4):
        patch = jx.primitives.plate_with_hole_sector(rad, side, k).elevate(deg).refine(refine)
        for _ in range(extra_u):
            uniq = np.unique(patch.knot_arrays()[0])
            patch = patch.insert_knots((uniq[:-1] + uniq[1:]) / 2, [])
        patches.append(patch)
    return jx.FunctionSpace(patches, vec=vec)


def plate_exact(rad=1.0):
    """``(exact_disp, exact_stress)`` over arrays of points, for the norms."""
    disp_np, stress_np, _, _ = kirsch(rad=rad)
    p = lambda pts: np.asarray(pts)
    return (lambda pts: np.stack(disp_np(p(pts)[:, 0], p(pts)[:, 1]), 1),
            lambda pts: stress_np(p(pts)[:, 0], p(pts)[:, 1]))


def plate_problem(V, rad=1.0):
    """The Kirsch boundary-value problem on a given space.

    A factory of this shape --- space in, problem out --- is what the adaptive
    loop needs, since a ``Problem`` resolves its ``where`` clauses against a
    fixed space at construction.

    This is a pure Neumann problem, so the three rigid-body modes have to be
    removed. The diagonal cut puts no control point where the axes meet the
    hole, so instead the analytical displacement is prescribed at two of the
    hole corners. Three conditions on a three-dimensional null space shift the
    discrete solution by a rigid motion and change nothing else, so the stress
    field is untouched and the displacement is anchored to the exact solution.
    """
    exact_disp, _ = plate_exact(rad)
    c = rad / np.sqrt(2.0)
    u1, u2 = exact_disp(np.array([[-c, c]]))[0], exact_disp(np.array([[c, c]]))[0]
    at = lambda px, py: (lambda x: np.hypot(x[:, 0] - px, x[:, 1] - py) < 1e-9)
    _, _, stress_jnp, _ = kirsch(rad=rad)

    def traction(x, n, p):
        s = stress_jnp(x[0], x[1])
        return jnp.stack([n[0]*s[0] + n[1]*s[2], n[0]*s[2] + n[1]*s[1]])

    return jx.LinearElasticity(
        V, plane="stress",
        dirichlet=[jx.DirichletBC(float(u1[0]), where=at(-c, c), component=0),
                   jx.DirichletBC(float(u1[1]), where=at(-c, c), component=1),
                   jx.DirichletBC(float(u2[1]), where=at(c, c), component=1)],
        neumann=[jx.Neumann(traction, where="outer")],
    )


def _rigid_modes(x):
    """The three planar rigid-body modes sampled at ``x``, shape ``(3, ..., 2)``."""
    ones = np.ones(x.shape[:-1])
    zeros = np.zeros(x.shape[:-1])
    return np.stack([np.stack([ones, zeros], -1),
                     np.stack([zeros, ones], -1),
                     np.stack([-x[..., 1], x[..., 0]], -1)])


def plate_l2_error(sol, exact_disp):
    """Relative $L^2$ displacement error with the rigid-body gauge fixed.

    This is a pure Neumann problem, so the displacement is defined only up to a
    rigid motion and any $L^2$ error has to say which one it means. Removing
    the rigid motion that best fits the error picks the gauge that minimises
    it, and makes the number independent of how the three modes were pinned:
    on a layout where both are expressible, the symmetric point constraints
    used by the finite element references and a three-corner anchoring give
    errors differing by a factor of five raw, and identical to five digits once
    the gauge is fixed. The symmetric anchoring already realises this gauge, so
    the value stays directly comparable to the reference codes.
    """
    basis = evaluate(sol.space, P.gauss(sol.space))
    x, w = np.asarray(basis.x), np.asarray(basis.w)
    u_ex = exact_disp(x.reshape(-1, 2)).reshape(*x.shape[:-1], 2)
    err = np.asarray(sol.at(basis=basis)) - u_ex
    r = _rigid_modes(x)
    coeff = np.linalg.solve(np.einsum("ieqc,jeqc,eq->ij", r, r, w),
                            np.einsum("ieqc,eqc,eq->i", r, err, w))
    num = np.sum(w * np.sum(err**2, -1)) - np.einsum("ieqc,eqc,eq->i", r, err, w) @ coeff
    return float(np.sqrt(num / np.sum(w * np.sum(u_ex**2, -1))))


def rigid_aligned_difference(a, b, pts):
    """Relative difference between two displacement fields, gauge removed.

    Used for the cross-code probe comparison, where the two solutions may carry
    different rigid-body offsets for the same reason.
    """
    r = _rigid_modes(np.asarray(pts))
    d = np.asarray(a) - np.asarray(b)
    coeff = np.linalg.solve(np.einsum("ipc,jpc->ij", r, r), np.einsum("ipc,pc->i", r, d))
    d = d - np.einsum("i,ipc->pc", coeff, r)
    return float(np.linalg.norm(d) / np.linalg.norm(b))


def build_plate(deg, refine, rad=1.0, side=4.0):
    exact_disp, exact_stress = plate_exact(rad)
    return plate_problem(plate_space(deg, refine, rad, side), rad), exact_disp, exact_stress


def strain_energies(sol, exact_stress):
    """Total strain energy of the discrete solution, and of the exact field.

    A scalar functional both codes can report without evaluating anything at a
    shared point, so it compares the whole pipeline -- geometry, quadrature,
    constitutive law and solve -- in one number. The exact value is the same
    integral taken over the analytical stress, which additionally puts JAXIGA's
    exact NURBS geometry and FEniCSx's curved triangles side by side.
    """
    from jaxiga.forms import materials

    basis = evaluate(sol.space, P.gauss(sol.space))
    C = sol.problem.C(sol.params)
    Cinv = jnp.linalg.inv(C)
    eps = jax.vmap(jax.vmap(materials.strain_voigt))(sol.grad(basis=basis))
    sig = jnp.einsum("ij,eqj->eqi", C, eps)
    discrete = 0.5 * jnp.sum(basis.w * jnp.sum(sig * eps, -1))

    x = np.asarray(basis.x).reshape(-1, basis.x.shape[-1])
    sig_ex = jnp.asarray(exact_stress(x)).reshape(sig.shape)
    exact = 0.5 * jnp.sum(
        basis.w * jnp.einsum("eqi,ij,eqj->eq", sig_ex, Cinv, sig_ex)
    )
    return float(discrete), float(exact)


def fenicsx_reference():
    """The FEniCSx solution of the same benchmark, as a plain dictionary.

    DOLFINx cannot share an interpreter with JAXIGA -- it is a compiled
    MPI/PETSc stack pinned to its own NumPy build -- so ``reference_fenicsx.py``
    runs under its own interpreter and communicates through JSON. Point
    ``JAXIGA_FENICSX_PYTHON`` at that interpreter to regenerate the file;
    without it the committed JSON is used, so the rest of this script runs
    anywhere.
    """
    path = os.path.join(HERE, "fenicsx_reference.json")
    exe = os.environ.get("JAXIGA_FENICSX_PYTHON")
    if exe and os.path.exists(exe):
        env = dict(os.environ)
        extra = os.environ.get("JAXIGA_FENICSX_DYLD_LIBRARY_PATH")
        if extra:
            env["DYLD_LIBRARY_PATH"] = extra
        print("    running reference_fenicsx.py under", exe)
        subprocess.run([exe, os.path.join(HERE, "reference_fenicsx.py"), path],
                       check=True, env=env)
    elif not os.path.exists(path):
        raise RuntimeError(
            "no FEniCSx reference: set JAXIGA_FENICSX_PYTHON to a DOLFINx "
            f"interpreter, or restore {path}"
        )
    with open(path) as fh:
        data = json.load(fh)
    if not exe:
        print(f"    using cached {os.path.basename(path)} "
              f"(dolfinx {data['versions']['dolfinx']})")
    return data


def jaxfem_reference():
    """Load or regenerate the independently pinned JAX-FEM benchmark."""
    path = os.path.join(HERE, "jaxfem_reference.json")
    exe = os.environ.get("JAXIGA_JAXFEM_PYTHON")
    if exe and not os.path.exists(exe):
        raise RuntimeError(f"JAXIGA_JAXFEM_PYTHON does not exist: {exe}")
    if exe:
        print("    running reference_jaxfem.py under", exe)
        subprocess.run(
            [exe, os.path.join(HERE, "reference_jaxfem.py"), path], check=True
        )
    elif not os.path.exists(path):
        raise RuntimeError(
            "no JAX-FEM reference: create paper/jaxfem-environment.yml and set "
            f"JAXIGA_JAXFEM_PYTHON, or restore {path}"
        )
    with open(path) as fh:
        data = json.load(fh)
    if not exe:
        versions = data["versions"]
        print(
            f"    using cached {os.path.basename(path)} "
            f"(jax-fem {versions['jax_fem']}, commit "
            f"{versions['jax_fem_commit'][:8]})"
        )
    return data


def machine_timings():
    """The same benchmark on the machine this is run on, CPU and GPU.

    Two subprocesses rather than one process: a backend is chosen before JAX is
    imported and cannot be changed afterwards, so a CPU and a GPU measurement
    cannot share an interpreter. Cached like the cross-code references, so the
    figure regenerates anywhere; set ``JAXIGA_REMEASURE_MACHINE`` to re-run it
    on the machine at hand.
    """
    path = os.path.join(HERE, "machine_timings.json")
    if os.environ.get("JAXIGA_REMEASURE_MACHINE") or not os.path.exists(path):
        driver = os.path.join(HERE, "reference_machine.py")
        out = {}
        for backend in ("gpu", "cpu"):
            dst = os.path.join(HERE, f"machine_timings_{backend}.json")
            print(f"    measuring this machine on {backend}")
            subprocess.run([sys.executable, driver, dst, "--backend", backend],
                           check=True)
            with open(dst) as fh:
                out[backend] = json.load(fh)
            os.remove(dst)
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
    with open(path) as fh:
        data = json.load(fh)
    for backend, run in data.items():
        print(f"    {backend}: {run['versions']['device_name']}, "
              f"{len(run['runs'])} levels to {run['runs'][-1]['dofs']} dofs")
    return data


def plate_with_hole():
    print("[3] Plate with a hole: JAXIGA against FEniCSx and JAX-FEM")
    ref = fenicsx_reference()
    jaxfem = jaxfem_reference()
    machine = machine_timings()
    RESULTS["fenicsx_versions"] = ref["versions"]
    RESULTS["jaxfem_versions"] = jaxfem["versions"]
    RESULTS["machine_versions"] = {k: v["versions"] for k, v in machine.items()}
    rr = ref["reference"]

    # The probe grid is subsampled: locating a physical point in a NURBS patch
    # is a host-side search followed by a Newton inversion, so a few hundred
    # points cost seconds while thousands cost minutes, and the comparison is
    # a norm over the set either way.
    points = np.asarray(ref["probe_points"])[::5]
    u_fem = np.asarray(rr["u_probe"])[::5, :2]
    disp_np, _, _, _ = kirsch()
    u_ex = np.stack(disp_np(points[:, 0], points[:, 1]), 1)
    rel = lambda a, b: rigid_aligned_difference(a, b, points)
    ref_probe_err = rel(u_fem, u_ex)
    print(f"    FEniCSx reference: {rr['dofs']} dofs, {ref_probe_err:.3e} from "
          f"the analytical solution over {len(points)} probe points")

    rows = []
    for deg in (2, 3, 4):
        for refine in (1, 2, 3, 4):
            problem, exact_disp, exact_stress = build_plate(deg, refine)
            params = {"E": 1e5, "nu": 0.3}
            # Timing convention, matched to the FEniCSx side: warm up once so
            # that tracing and XLA compilation are excluded, then take the
            # median of repeated calls. Function-space construction, like mesh
            # construction on the other side, is one-time setup and excluded.
            jx.solve(problem, params=params).u.block_until_ready()
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                sol = jx.solve(problem, params=params)
                sol.u.block_until_ready()
                ts.append(time.perf_counter() - t0)
            elapsed = float(np.median(ts))

            l2 = plate_l2_error(sol, exact_disp)
            en = jx.errornorm(sol, exact_disp, "energy", exact_grad=exact_stress)
            se, se_exact = strain_energies(sol, exact_stress)

            # Probing is the expensive step, and the cross-code agreement it
            # measures is a statement about the fields, not about refinement,
            # so it is done on the middle levels only.
            probe = None
            if refine in (2, 3):
                u_iga = np.asarray(sol.probe(points))
                probe = (rel(u_iga, u_fem), rel(u_iga, u_ex))

            rows.append({
                "deg": deg, "refine": refine, "dofs": problem.space.n_dofs,
                "l2": l2, "energy": en, "time": elapsed,
                "strain_energy": se, "exact_energy": se_exact, "probe": probe,
            })
            msg = (f"    p={deg} r={refine}: dofs {problem.space.n_dofs:5d} "
                   f"{elapsed:6.3f}s  L2 {l2:.4e}  energy {en:.4e}")
            if probe:
                msg += f"; vs FEniCSx {probe[0]:.4e} vs exact {probe[1]:.4e}"
            print(msg)

    # Every curve in the figure is now measured on this machine, so the check
    # that matters is between the two backends of it: same code, same
    # discretisation, different processor, and the errors have to agree.
    gpu = {(r["deg"], r["refine"]): r for r in machine["gpu"]["runs"]}
    drift = max(abs(r["l2"] / gpu[(r["deg"], r["refine"])]["l2"] - 1)
                for r in machine["cpu"]["runs"])
    print(f"    the two backends agree on the errors to {drift:.1e} relative")
    RESULTS["backend_error_drift"] = drift

    _plate_work_precision(ref, jaxfem, machine)

    digits = lambda a, b: -np.log10(abs(a - b) / abs(b) + 1e-17)
    agree = [digits(*r["probe"]) for r in rows if r["probe"]]
    worst = max(r["probe"][0] for r in rows if r["probe"])
    print(f"    cross-code agreement at the probe points: "
          f"{min(agree):.1f} to {max(agree):.1f} digits, "
          f"largest relative difference {worst:.2e}")

    fine = rows[-1]
    e_iga, e_fem, e_ex = fine["strain_energy"], rr["strain_energy"], fine["exact_energy"]
    fine_ctx = build_context(problem, params)
    raw_entries = int(
        np.count_nonzero(fine_ctx.assembly.scatter < fine_ctx.assembly.nnz)
    )
    unique_nnz = fine_ctx.assembly.nnz
    assembly_reduction = raw_entries / unique_nnz
    print(f"    strain energy: JAXIGA {e_iga:.12e}  FEniCSx {e_fem:.12e}  "
          f"exact {e_ex:.12e}")
    print(f"      JAXIGA vs exact {abs(e_iga/e_ex - 1):.2e}, "
          f"FEniCSx vs exact {abs(e_fem/e_ex - 1):.2e}, "
          f"exact integral computed by the two codes differs by "
          f"{abs(rr['exact_energy']/e_ex - 1):.2e}")
    print(f"    assembly: {raw_entries} local entries -> {unique_nnz} CSR nonzeros "
          f"({assembly_reduction:.2f}x reduction)")

    # accuracy per unit time, read off the two finest comparable points
    RESULTS["plate_vs_fenicsx"] = {
        "reference_dofs": rr["dofs"],
        "reference_probe_error": ref_probe_err,
        "agreeing_digits": [float(min(agree)), float(max(agree))],
        "max_relative_difference": worst,
        "strain_energy_jaxiga": e_iga,
        "strain_energy_fenicsx": e_fem,
        "strain_energy_exact": e_ex,
        "exact_energy_two_codes": abs(rr["exact_energy"] / e_ex - 1),
        "assembly_raw_entries": raw_entries,
        "assembly_unique_nnz": unique_nnz,
        "assembly_reduction": assembly_reduction,
        "superlu_ordering": "MMD_AT_PLUS_A (SymmetricMode)",
        "jaxiga": machine["cpu"]["runs"],
        "jaxiga_this_machine": [
            {k: r[k] for k in ("deg", "refine", "dofs", "l2", "energy", "time")}
            for r in rows
        ],
        "machine_scaling": {k: v["runs"] for k, v in machine.items()},
        "fenicsx": [{k: r[k] for k in ("degree", "cells", "dofs", "l2", "energy", "time")}
                    for r in ref["runs"]],
        "jaxfem": [{k: r[k] for k in ("degree", "cells", "dofs", "l2", "energy", "time")}
                   for r in jaxfem["runs"]],
    }

    # Where the error sits, and whether the patch coupling is what puts it
    # there; then the adaptive run that the figure shows.
    # The adaptive loop needs room: the first few cycles are coarser than the
    # uniform mesh they are compared with, and localisation only pays once the
    # solution is resolved well enough for the indicator to see where the error
    # actually is. The budget is therefore set above the finest uniform run.
    ref_run = next(r for r in rows if r["deg"] == 4 and r["refine"] == 4)
    sol_ad, V_ad, hist = plate_adaptive(deg=4, budget=4000)
    # dofs the uniform family would need for the adaptive error, read off its
    # own last two points, which is the comparison that does not depend on
    # landing at exactly the same size
    q0, q1 = [r for r in rows if r["deg"] == 4][-2:]
    slope = np.log(q1["energy"] / q0["energy"]) / np.log(q1["dofs"] / q0["dofs"])
    need = q1["dofs"] * (hist[-1]["energy"] / q1["energy"]) ** (1.0 / slope)
    gain = float(need / hist[-1]["n_dofs"])
    print(f"    adaptive p=4: {hist[-1]['n_dofs']} dofs, {hist[-1]['n_elems']} elements, "
          f"{hist[-1]['levels']} levels, energy error {hist[-1]['energy']:.4e}; the "
          f"uniform family needs {need:.0f} dofs for that ({gain:.1f}x fewer)")
    RESULTS["plate_adaptive"] = {
        "history": hist,
        "uniform_dofs": ref_run["dofs"],
        "uniform_energy": ref_run["energy"],
        "uniform_dofs_for_same_error": float(need),
        "dof_gain": gain,
        "n_local": V_ad.n_local,
        "n_bernstein": V_ad.n_bernstein,
    }
    print(f"    adapted mesh: n_local_max {V_ad.n_local} "
          f"(tensor-product: {V_ad.n_bernstein})")
    RESULTS["plate_adaptive"].update(_plate_fields_figure(
        sol_ad, jx.solve(plate_problem(plate_space(4, 3)), params={"E": 1e5, "nu": 0.3})
    ))


def plate_adaptive(deg=4, budget=2112, frac=0.5, start=1, max_cycles=40):
    """Solve, estimate, mark, refine, stopping just under a dof budget.

    Written out rather than delegating to :func:`jaxiga.adapt` because the
    stopping rule is a budget matched to a uniform mesh and the quantity of
    interest is the energy norm; the ingredients are the same public ones the
    driver uses. Refinement is mirrored across the patch interfaces by
    ``refine_elements`` itself, so the adapted meshes stay conforming there and
    the space is still glued dof for dof.
    """
    params = {"E": 1e5, "nu": 0.3}
    exact_disp, exact_stress = plate_exact()
    V, hist = plate_space(deg, start), []
    for _ in range(max_cycles):
        sol = jx.solve(plate_problem(V), params=params)
        hist.append({
            "n_dofs": int(V.n_dofs), "n_elems": int(V.n_elems),
            "energy": jx.errornorm(sol, exact_disp, "energy", exact_grad=exact_stress),
            "levels": V.hierarchy.n_levels if V.is_hierarchical else 1,
        })
        nxt = jx.refine_elements(V, jx.dorfler_mark(jx.residual_indicator(sol), frac))
        if nxt.n_dofs > budget:
            return sol, V, hist
        V = nxt
    raise RuntimeError("adaptive loop did not reach the dof budget")

def _plate_fields_figure(sol, sol_uniform, n_samp=6):
    """von Mises stress and its error on the adapted mesh, and the mesh itself."""
    from matplotlib.collections import LineCollection

    _, exact_stress = plate_exact()

    def sample(s):
        basis = evaluate(s.space, P.grid(s.space, n_samp))
        x = np.asarray(basis.x).reshape(-1, 2)
        vm = np.asarray(s.field("von_mises", basis=basis)).reshape(-1)
        sx, sy, sxy = exact_stress(x).T
        vm_ex = np.sqrt(sx**2 - sx*sy + sy**2 + 3*sxy**2)
        triang = matplotlib.tri.Triangulation(
            x[:, 0], x[:, 1], element_triangulation(basis.n_elems, n_samp)
        )
        return triang, vm, np.abs(vm - vm_ex)

    triang, vm, err = sample(sol)
    err_uniform = sample(sol_uniform)[2]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    sc = axes[0].tripcolor(triang, vm, shading="gouraud", cmap="viridis")
    axes[0].set_title("von Mises stress", fontsize=9)
    plt.colorbar(sc, ax=axes[0])

    sc2 = axes[1].tripcolor(triang, err, shading="gouraud", cmap="magma")
    axes[1].set_title("absolute error vs Kirsch", fontsize=9)
    plt.colorbar(sc2, ax=axes[1])

    # the four patch interfaces lie on the diagonals; drawing them first, wide
    # and pale, lets the mesh lines over them show the grading being mirrored
    c = 1.0 / np.sqrt(2.0)
    axes[2].add_collection(LineCollection(
        [[(sx*c, sy*c), (sx*4.0, sy*4.0)] for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))],
        lw=2.2, colors="tab:blue", alpha=0.35))
    axes[2].add_collection(LineCollection(mesh_lines(sol.space), lw=0.25, colors="0.3"))
    axes[2].set_xlim(-4.15, 4.15)
    axes[2].set_ylim(-4.15, 4.15)
    axes[2].set_title(f"adapted mesh ({sol.space.n_elems} elements)", fontsize=9)
    plt.colorbar(sc2, ax=axes[2]).ax.set_visible(False)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.tick_params(labelsize=8)
    for ax in axes[:2]:
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="w", lw=0.8))
    axes[0].set_ylabel("$y$")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(FIGURES, "plate_von_mises.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/plate_von_mises.pdf (peak error {err.max():.3e} on the "
          f"adapted mesh against {err_uniform.max():.3e} on the uniform one)")
    return {"peak_vm_error": float(err.max()),
            "peak_vm_error_uniform": float(err_uniform.max())}


def _default_route_time(row, backend):
    """The time the library's default solver actually takes on that backend.

    The sweep measures both routes at every level; which one a user gets is
    decided by :func:`jaxiga.methods.galerkin.default_linear`, so the figure
    applies the same rule rather than picking a route by hand. Plotting the
    iterative time on a CPU, or below the crossover, would show a curve nobody
    runs.
    """
    from jaxiga.methods.galerkin import CG_CROSSOVER

    if backend != "cpu" and row["dofs"] >= CG_CROSSOVER:
        return row["time"]
    return row["time_direct"]


def _plate_work_precision(ref, jaxfem, machine):
    """Error against solve time for all three codes, in both norms.

    A work-precision diagram rather than a table: the codes discretise
    differently, so degrees of freedom are not comparable between them, but
    time to a given accuracy is. All three run on the one machine, the two JAX
    codes on its accelerator and FEniCSx on a CPU core of it, since DOLFINx has
    no accelerator path to give it.
    """
    width, k = 6.4, 1.0
    # Sized to the 390 pt text width so the figure is included close to 1:1.
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.5 * k))
    blues = [plt.cm.Blues(v) for v in (0.55, 0.75, 0.95)]
    oranges = [plt.cm.Oranges(v) for v in (0.45, 0.65, 0.85)]
    marks = ("o", "s", "^")
    gpu_runs = machine["gpu"]["runs"]
    handles = []

    for col, key in enumerate(("l2", "energy")):
        ax = axes[col]
        for deg, c, m in zip((2, 3, 4), blues, marks):
            r = [x for x in gpu_runs if x["deg"] == deg]
            h, = ax.loglog([_default_route_time(x, "gpu") for x in r],
                           [x[key] for x in r], color=c,
                           marker=m, ms=3.2, lw=1.0, label=f"JAXIGA $p={deg}$")
            if col == 0:
                handles.append(h)
        for deg, c, m in zip((1, 2, 3), oranges, marks):
            r = [x for x in ref["runs"] if x["degree"] == deg]
            h, = ax.loglog([x["time"] for x in r], [x[key] for x in r], color=c,
                           marker=m, ms=3.2, lw=1.0, ls="--", mfc="none",
                           label=f"FEniCSx $P_{deg}$")
            if col == 0:
                handles.append(h)
        r = jaxfem["runs"]
        h, = ax.loglog(
            [x["time"] for x in r], [x[key] for x in r],
            color=plt.cm.Greens(0.72), marker="D", ms=3.2, lw=1.0,
            ls="-.", mfc="none", label="JAX-FEM $P_2$",
        )
        if col == 0:
            handles.append(h)
        ax.set_xlabel("solution time (s)", fontsize=9 * k)
        ax.set_title({"l2": r"relative $L^2$ error",
                      "energy": "relative energy error"}[key], fontsize=9 * k, pad=3)
        ax.grid(True, which="major", alpha=0.3)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=100, numticks=6))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(labelsize=7.5 * k)

    # Column-major fill: each JAXIGA degree sits above a FEniCSx one, with the
    # accelerated curve and JAX-FEM sharing the last column.
    order = [0, 3, 1, 4, 2, 5, 6]
    fig.legend([handles[i] for i in order], [handles[i].get_label() for i in order],
               loc="lower center", ncol=4, fontsize=7.2 * k,
               frameon=False, bbox_to_anchor=(0.5, -0.24), handlelength=1.8,
               columnspacing=1.1)
    fig.subplots_adjust(wspace=0.32)
    fig.savefig(os.path.join(FIGURES, "plate_work_precision.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/plate_work_precision.pdf")


if __name__ == "__main__":
    run_example("plate_with_hole", plate_with_hole)
