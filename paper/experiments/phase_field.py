"""Resolve brittle fracture in a single-edge-notched plate under tension.

Run from the repository root::

    python -m paper.experiments.phase_field

See paper/README.md for outputs, caching and the suggested reading order.
"""

import os
import time

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

import jaxiga as jx
from jaxiga.methods._common import PointField, build_context, reactions
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._common import FIGURES, RESULTS, figure_data, run_example
from ._fracture_common import _crack_panel_data
from ._plotting import element_triangulation


class _CachedLU:
    """A SuperLU factorisation reused until iterative refinement stops working.

    Factorising costs about forty times what the triangular solves cost, the
    sparsity never changes across a run, and with $10^{-6}$ load increments the
    matrix barely moves between staggered sweeps. Reusing the factorisation and
    refining the result therefore removes most of the linear-algebra cost; the
    residual test keeps the solve exact, and a factorisation that no longer
    works is simply rebuilt.
    """

    def __init__(self, pattern, tol=1e-5):
        self.indptr, self.cols = pattern.indptr, pattern.column_indices
        self.shape, self.tol = pattern.shape, tol
        self.lu, self.refactor, self.calls = None, 0, 0

    def solve(self, data, b):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        self.calls += 1
        A = sp.csr_matrix((np.asarray(data), self.cols, self.indptr),
                          shape=self.shape).tocsc()
        nb = np.linalg.norm(b) or 1.0
        if self.lu is not None:
            x = self.lu.solve(b)
            r = b - A @ x
            if np.linalg.norm(r) > self.tol * nb:
                x = x + self.lu.solve(r)
                r = b - A @ x
            if np.linalg.norm(r) <= self.tol * nb:
                return x
        self.lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A",
                            options={"SymmetricMode": True})
        self.refactor += 1
        return self.lu.solve(b)


def _phase_field_compute():
    """Single-edge-notched plate in tension, on an isotropically refined mesh.

    Three things decide whether this reproduces the brittle response the
    benchmark is known for, and all three were wrong in the first version of
    this example.

    *Resolution across the crack band* sets the peak load. At $\\ell/h = 1$ the
    AT2 profile cannot form, damage is suppressed and the specimen never
    cracks; the sequence over $\\ell/h = 1, 2, 4$ is $771$, $641$, $616$~N.
    Refinement here is hierarchical from a square base mesh, so elements stay
    square and the same $\\ell/h$ holds along the crack as across it --- a
    tensor-product graded mesh can only resolve one direction.

    *The load increments* must be small enough that the staggered iteration
    starts each step near its fixed point. Fixed increments of $10^{-6}$
    through the peak need two sweeps per step; large adaptive increments need
    hundreds, because alternate minimisation converges linearly at a rate that
    degrades with increment size.

    *The inner convergence test* has to be one the iteration can actually pass.
    On the softening branch the crack advances during the sweeps, so a bound on
    $|\\Delta\\phi|$ never converges at fixed load; the relative residual of the
    damage system at the previous damage field does.
    """
    import dataclasses

    import scipy.sparse as sp
    from jaxiga.forms.library import edge_crack_distance, tensile_energy
    from jaxiga.methods.galerkin import assemble_csr_values, rhs
    from jaxiga.post.solution import Solution
    from jaxiga.space.hierarchical import build_hierarchical_space

    E, NU, GC, ELL = 210e3, 0.3, 2.7, 0.0125
    BASE, LEVELS, BAND = 20, 4, 2.0          # base mesh, levels, band half-width / l
    NSTEP, DU1, DU2, SWITCH = 1200, 1e-4, 1e-6, 45
    TOL_STAG, MAX_STAG, LU_TOL = 1e-4, 100, 1e-5

    # -- isotropic base mesh, crack path refined hierarchically ------------
    kn = np.linspace(0.0, 1.0, BASE + 1)[1:-1]
    patch = jx.primitives.rectangle(0, 0, 1, 1).elevate(2).insert_knots(kn, kn)
    Vd = jx.refine_elements(jx.FunctionSpace(patch, vec=1), [])
    for _ in range(LEVELS):
        ev = np.asarray(Vd.elem_vertex)
        near = (ev[:, 3] >= 0.5 - BAND * ELL) & (ev[:, 1] <= 0.5 + BAND * ELL)
        marked = np.flatnonzero(near & (np.asarray(Vd.elem_key)[:, 1] < LEVELS))
        if not marked.size:
            break
        Vd = jx.refine_elements(Vd, marked)
    Vu = build_hierarchical_space(Vd.patches, Vd.hierarchy, vec=2)
    basis = evaluate(Vu, P.gauss(Vu))
    top = np.asarray(Vu.boundaries["top"].dofs)
    ev = np.asarray(Vd.elem_vertex)
    h_min = float(min((ev[:, 2] - ev[:, 0]).min(), (ev[:, 3] - ev[:, 1]).min()))
    band = np.abs(0.5 * (ev[:, 1] + ev[:, 3]) - 0.5) < 0.5 * ELL
    aspect = float(np.max((ev[band, 2] - ev[band, 0]) / (ev[band, 3] - ev[band, 1])))
    print(f"    {BASE}x{BASE} base, {LEVELS} levels: {Vd.n_elems} elements, "
          f"{Vu.n_dofs} dofs, l/h = {ELL/h_min:.1f}, band aspect {aspect:.2f}:1")

    # -- contexts and factorisations built once ----------------------------
    # The basis at the Gauss points and the sparsity are fixed for the whole
    # run; only the prescribed displacement changes, and it is linear in the
    # applied value, so one base vector scales. Going through jx.solve would
    # rebuild all of it on each of the ~5000 solves.
    disp = jx.PhaseFieldDisplacement(
        Vu, plane="strain",
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                   jx.DirichletBC([0.0, 1.0], where="top")])
    damage = jx.PhaseFieldDamage(Vd)
    zero = jnp.zeros((basis.n_elems, basis.n_q))
    ctx_u = build_context(disp, {"E": E, "nu": NU, "phi": PointField(zero)})
    ctx_d = build_context(damage, {"Gc": GC, "l": ELL, "H": PointField(zero)})
    base_values = ctx_u.dofmap.values
    free_u, pres_u = np.asarray(ctx_u.dofmap.free), np.asarray(ctx_u.dofmap.prescribed)
    lu_u, lu_d = _CachedLU(ctx_u.assembly, LU_TOL), _CachedLU(ctx_d.assembly, LU_TOL)

    @jax.jit
    def assemble_u(u_top, ph):
        dm = dataclasses.replace(ctx_u.dofmap, values=base_values * u_top)
        c = dataclasses.replace(ctx_u, dofmap=dm)
        pu = {"E": E, "nu": NU, "phi": PointField(ph)}
        return rhs(disp, pu, c), assemble_csr_values(c, pu)

    @jax.jit
    def lift_u(uf, u_top):
        return jnp.zeros(Vu.n_dofs).at[free_u].set(uf).at[pres_u].set(base_values * u_top)

    @jax.jit
    def assemble_d(hist):
        pd = {"Gc": GC, "l": ELL, "H": PointField(hist)}
        return rhs(damage, pd, ctx_d), assemble_csr_values(ctx_d, pd)

    dist = np.asarray(jnp.vectorize(edge_crack_distance((0.5, 0.5)),
                                    signature="(2)->()")(basis.x))
    hist = jnp.asarray(np.where(dist <= ELL / 2,
                                1e3 * GC * (1 - dist / (ELL / 2)) / (2 * ELL), 0.0))
    phi, sol_d = zero, None

    u, curve, iters = 0.0, [(0.0, 0.0)], []
    t0 = time.perf_counter()
    for istep in range(1, NSTEP + 1):
        u += DU1 if istep < SWITCH else DU2
        used, prev = MAX_STAG, None
        for it in range(MAX_STAG):
            b, data = assemble_u(u, phi)
            uf = lu_u.solve(np.asarray(data), np.asarray(b))
            pu = {"E": E, "nu": NU, "phi": PointField(phi)}
            sol_u = Solution(Vu, lift_u(jnp.asarray(uf), u), pu, disp)
            hist = jnp.maximum(hist, tensile_energy(sol_u, {"E": E, "nu": NU}, basis))
            bd, dd = assemble_d(hist)
            d_np, b_np = np.asarray(dd), np.asarray(bd)
            res = np.inf
            if prev is not None:
                A = sp.csr_matrix((d_np, lu_d.cols, lu_d.indptr), shape=lu_d.shape)
                res = float(np.linalg.norm(A @ prev - b_np)
                            / (np.linalg.norm(b_np) or 1.0))
            prev = lu_d.solve(d_np, b_np)
            sol_d = Solution(Vd, ctx_d.dofmap.lift(jnp.asarray(prev)),
                             {"Gc": GC, "l": ELL, "H": PointField(hist)}, damage)
            phi = jnp.clip(sol_d.at(basis=basis)[..., 0], 0.0, 1.0)
            if res < TOL_STAG:
                used = it + 1
                break
        curve.append((u, float(jnp.sum(reactions(sol_u, params=pu)[2 * top + 1]))))
        iters.append(used)
    elapsed = time.perf_counter() - t0

    u_arr = np.array([c[0] for c in curve])
    f_arr = np.array([c[1] for c in curve])
    peak = int(np.argmax(f_arr))
    print(f"    peak {f_arr[peak]:.1f} N at u={u_arr[peak]:.5f}, final {f_arr[-1]:.1f} N, "
          f"{NSTEP} increments, {sum(iters)} staggered sweeps (median "
          f"{int(np.median(iters))}), {elapsed:.0f}s")
    print(f"    factorisations reused: {lu_u.calls - lu_u.refactor} of {lu_u.calls} "
          f"displacement solves, {lu_d.calls - lu_d.refactor} of {lu_d.calls} damage")

    np.savez(os.path.join(FIGURES, "phase_field_curve.npz"), u=u_arr, f=f_arr)

    record = {
        "results": {
            "peak_force": float(f_arr[peak]), "peak_disp": float(u_arr[peak]),
            "final_force": float(f_arr[-1]), "steps": NSTEP,
            "sweeps": int(sum(iters)), "median_sweeps": int(np.median(iters)),
            "l_over_h": ELL / h_min, "band_aspect": aspect,
            "elems": int(Vd.n_elems), "dofs": int(Vu.n_dofs), "levels": LEVELS,
            "base_mesh": BASE, "seconds": elapsed,
            "refactor_u": lu_u.refactor, "solves_u": lu_u.calls,
        },
        "peak": peak,
        "n_elems": int(Vd.n_elems),
        "u": u_arr.tolist(),
        "f": f_arr.tolist(),
    }
    return record, _crack_panel_data(Vd, sol_d)


def phase_field():
    """The tension test on a mesh laid out in advance."""
    print("[6] Phase-field fracture")
    record, arrays = figure_data("phase_field", _phase_field_compute)
    RESULTS["phase_field"] = record["results"]
    _phase_field_figure(record, arrays)


def _phase_field_figure(record, arrays):
    """Load--displacement, the damage field at the final load, and the mesh."""
    from matplotlib.collections import LineCollection

    u_arr = np.asarray(record["u"])
    f_arr = np.asarray(record["f"])
    peak = int(record["peak"])
    x = arrays["x"]
    tri = element_triangulation(int(arrays["n_elems"]), int(arrays["n_samp"]))
    triang = matplotlib.tri.Triangulation(x[:, 0], x[:, 1], tri)
    dmg = arrays["dmg"]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    axes[0].plot(u_arr * 1e3, f_arr, "-", lw=1.3, color="tab:blue")
    axes[0].plot(u_arr[peak] * 1e3, f_arr[peak], "r*", ms=11)
    axes[0].annotate(f"peak {f_arr[peak]:.0f} N",
                     xy=(u_arr[peak] * 1e3, f_arr[peak]), xytext=(-58, -4),
                     textcoords="offset points", fontsize=8, color="tab:red")
    axes[0].set_xlabel(r"prescribed displacement [$\mu$m]")
    axes[0].set_ylabel(r"reaction $F_y$ [N]")
    axes[0].set_title("load--displacement", fontsize=9)
    axes[0].grid(alpha=0.3)

    sc = axes[1].tripcolor(triang, dmg, shading="gouraud", cmap="inferno",
                           vmin=0.0, vmax=1.0)
    axes[1].set_title(r"damage $\phi$ at the final increment", fontsize=9)
    plt.colorbar(sc, ax=axes[1])

    axes[2].add_collection(
        LineCollection(list(arrays["mesh"]), lw=0.12, colors="0.3"))
    axes[2].set_xlim(-0.02, 1.02); axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title(f"adapted mesh ({record['n_elems']:,} elements)", fontsize=9)
    plt.colorbar(sc, ax=axes[2]).ax.set_visible(False)   # keep the axes the same size

    for ax in axes[1:]:
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(os.path.join(FIGURES, "phase_field.pdf"))
    plt.close(fig)
    print(f"  wrote {FIGURES}/phase_field.pdf")


if __name__ == "__main__":
    run_example("phase_field", phase_field)
