"""Independent FEniCSx reference for the plate-with-a-hole benchmark.

Run separately from ``generate_results.py``: FEniCSx and JAXIGA cannot share an
interpreter (DOLFINx is a compiled MPI/PETSc stack pinned to its own NumPy), so
this script writes ``fenicsx_reference.json`` and the generator reads it.
``generate_results.py`` invokes it automatically when a DOLFINx interpreter is
configured; see ``FENICSX_PYTHON`` there.

The boundary-value problem is exactly the one JAXIGA solves: the square
$[-4,4]^2$ minus the unit disk, plane stress, traction-free hole, the exact
Kirsch traction applied on the outer boundary, and the three rigid-body modes
removed by pinning $u_y$ at $(\\pm 1, 0)$ and $u_x$ at $(0, \\pm 1)$ --- values
the analytical solution satisfies, so the constraint is consistent.

Nothing here is shared with JAXIGA. The mesh is built from an analytic
transfinite map between the circle and the square rather than from any NURBS
evaluation, the discretisation is Lagrange finite elements on curved triangles,
and the linear algebra is PETSc.
"""

from __future__ import annotations

import json
import sys
import time

import basix.ufl
import dolfinx
import numpy as np
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

RAD, SIDE, TRAC, EMOD, NU = 1.0, 4.0, 10.0, 1.0e5, 0.3

# Discretisations to run. The last entry is the reference: its own error is
# roughly two orders of magnitude below the best JAXIGA case, so the difference
# between the two codes is dominated by the JAXIGA discretisation and can be
# compared against its error against the analytical solution.
RUNS = [(deg, n) for deg in (1, 2, 3) for n in (4, 8, 16, 32)]
REPEATS = 5
REFERENCE = (3, 48)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def phi(th, t):
    """Transfinite map from the reference annulus to the physical domain.

    ``t = 0`` is the circle of radius ``RAD``, ``t = 1`` the square of
    half-width ``SIDE``; both are hit exactly, so every boundary node lies on
    the true geometry. Only the interior blend is arbitrary.
    """
    direction = np.stack([np.cos(th), np.sin(th)], -1)
    scale = SIDE / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th)))
    return (1 - t)[..., None] * (RAD * direction) + t[..., None] * (scale[..., None] * direction)


def build_mesh(nth, nt):
    """Curved (P2 geometry) triangle mesh of the square minus the disk.

    ``nth`` must be a multiple of eight so that the four corners of the square
    fall on grid lines; otherwise an element would straddle a corner and the
    outer boundary would be cut.
    """
    if nth % 8:
        raise ValueError("nth must be a multiple of 8 so the corners are nodes")
    A, B = 2 * nth, 2 * nt  # doubled lattice: the odd indices are the P2 nodes
    th = 2 * np.pi * np.arange(A) / A
    t = np.arange(B + 1) / B
    node = np.arange(A * (B + 1), dtype=np.int64).reshape(B + 1, A).T
    grid = np.stack(np.meshgrid(th, t, indexing="ij"), -1).reshape(-1, 2)
    x = np.empty((A * (B + 1), 2))
    x[node.reshape(-1)] = phi(grid[:, 0], grid[:, 1])

    nd = lambda i, j: node[i % A, j]  # noqa: E731  -- periodic in theta
    cells = []
    for i in range(nth):
        for j in range(nt):
            i0, i2, j0, j2 = 2 * i, 2 * i + 2, 2 * j, 2 * j + 2
            for v in ([(i0, j0), (i2, j0), (i2, j2)], [(i0, j0), (i2, j2), (i0, j2)]):
                # basix orders the P2 edge nodes opposite each vertex
                edges = [(v[1], v[2]), (v[0], v[2]), (v[0], v[1])]
                cells.append(
                    [nd(*p) for p in v]
                    + [nd((p[0] + q[0]) // 2, (p[1] + q[1]) // 2) for p, q in edges]
                )
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 2, shape=(2,)))
    return dmesh.create_mesh(MPI.COMM_WORLD, np.array(cells, dtype=np.int64), x, domain)


# --------------------------------------------------------------------------
# the Kirsch solution and the constitutive law, written in UFL
# --------------------------------------------------------------------------
def kirsch(x):
    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    th = ufl.atan2(x[1], x[0])
    a2, a4 = RAD**2 / r**2, RAD**4 / r**4
    srr = TRAC / 2 * (1 - a2) + TRAC / 2 * (1 - 4 * a2 + 3 * a4) * ufl.cos(2 * th)
    stt = TRAC / 2 * (1 + a2) - TRAC / 2 * (1 + 3 * a4) * ufl.cos(2 * th)
    srt = -TRAC / 2 * (1 + 2 * a2 - 3 * a4) * ufl.sin(2 * th)
    c, s = ufl.cos(th), ufl.sin(th)
    Q = ufl.as_matrix([[c, -s], [s, c]])
    sigma = Q * ufl.as_matrix([[srr, srt], [srt, stt]]) * Q.T
    ux = (
        (1 + NU) / EMOD * TRAC
        * (r * c / (1 + NU) + 2 * RAD**2 * c / ((1 + NU) * r)
           + RAD**2 * ufl.cos(3 * th) / (2 * r) - RAD**4 * ufl.cos(3 * th) / (2 * r**3))
    )
    uy = (
        (1 + NU) / EMOD * TRAC
        * (-NU * r * s / (1 + NU) - (1 - NU) * RAD**2 * s / ((1 + NU) * r)
           + RAD**2 * ufl.sin(3 * th) / (2 * r) - RAD**4 * ufl.sin(3 * th) / (2 * r**3))
    )
    return ufl.as_vector([ux, uy]), sigma


MU = EMOD / (2 * (1 + NU))
LAM = EMOD * NU / (1 - NU**2)  # plane stress


def stress(u):
    e = ufl.sym(ufl.grad(u))
    return 2 * MU * e + LAM * ufl.tr(e) * ufl.Identity(2)


def compliance(s):
    """``C^{-1} s``, so that ``inner(s, compliance(s))`` is the JAXIGA energy norm."""
    return ufl.as_matrix(
        [[(s[0, 0] - NU * s[1, 1]) / EMOD, (1 + NU) * s[0, 1] / EMOD],
         [(1 + NU) * s[1, 0] / EMOD, (s[1, 1] - NU * s[0, 0]) / EMOD]]
    )


# --------------------------------------------------------------------------
def probe_points(n_r=32, n_th=64, r_lo=1.08, r_hi=3.4):
    """Interior points at which both codes are sampled.

    Held well clear of both boundaries so that neither code is evaluated
    outside its own discrete domain.
    """
    r = np.linspace(r_lo, r_hi, n_r)
    th = 2 * np.pi * (np.arange(n_th) + 0.5) / n_th
    R, T = np.meshgrid(r, th, indexing="ij")
    return np.stack([R * np.cos(T), R * np.sin(T)], -1).reshape(-1, 2)


def solve(deg, nth, nt, points=None):
    msh = build_mesh(nth, nt)
    V = fem.functionspace(msh, ("Lagrange", deg, (2,)))
    x = ufl.SpatialCoordinate(msh)
    u_ex, sig_ex = kirsch(x)

    facets = dmesh.locate_entities_boundary(
        msh, 1, lambda p: np.isclose(np.max(np.abs(p[:2]), axis=0), SIDE, atol=1e-8)
    )
    tags = dmesh.meshtags(msh, 1, np.sort(facets), np.ones(len(facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    meta = {"quadrature_degree": 2 * deg + 4}
    a = ufl.inner(stress(u), ufl.sym(ufl.grad(v))) * ufl.dx(metadata=meta)
    L = ufl.inner(ufl.dot(sig_ex, ufl.FacetNormal(msh)), v) * ds(1, metadata=meta)

    bcs = []
    for pt, comp in (((RAD, 0.0), 1), ((-RAD, 0.0), 1), ((0.0, RAD), 0), ((0.0, -RAD), 0)):
        Vc, _ = V.sub(comp).collapse()
        dofs = fem.locate_dofs_geometrical(
            (V.sub(comp), Vc),
            lambda p, pt=pt: np.isclose(p[0], pt[0], atol=1e-9) & np.isclose(p[1], pt[1], atol=1e-9),
        )
        if dofs[0].size != 1:
            raise RuntimeError(f"expected one dof at {pt}, found {dofs[0].size}")
        zero = fem.Function(Vc)
        bcs.append(fem.dirichletbc(zero, dofs, V.sub(comp)))

    # Timing convention, matched to the JAXIGA side: one-time setup is excluded
    # (mesh, function space, and the FFCx compilation of the forms, which is
    # cached on disk in any case) and the repeated solve is timed after a
    # warm-up call. What remains is assembly, boundary conditions and the
    # factorised solve -- the same work JAXIGA's ``solve`` performs.
    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"},
    )
    uh = problem.solve()  # warm-up: triggers the JIT compilation of the forms
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        uh = problem.solve()
        times.append(time.perf_counter() - t0)
    elapsed = float(np.median(times))

    dxq = ufl.dx(metadata=meta)
    err = uh - u_ex
    l2 = np.sqrt(
        fem.assemble_scalar(fem.form(ufl.inner(err, err) * dxq))
        / fem.assemble_scalar(fem.form(ufl.inner(u_ex, u_ex) * dxq))
    )
    d = stress(uh) - sig_ex
    energy = np.sqrt(
        fem.assemble_scalar(fem.form(ufl.inner(d, compliance(d)) * dxq))
        / fem.assemble_scalar(fem.form(ufl.inner(sig_ex, compliance(sig_ex)) * dxq))
    )

    # total strain energy: a scalar functional both codes can report, so the
    # comparison does not depend on either one's point evaluation
    strain_energy = fem.assemble_scalar(
        fem.form(0.5 * ufl.inner(stress(uh), ufl.sym(ufl.grad(uh))) * dxq)
    )
    exact_energy = fem.assemble_scalar(
        fem.form(0.5 * ufl.inner(sig_ex, compliance(sig_ex)) * dxq)
    )

    out = {
        "degree": deg, "nth": nth, "nt": nt,
        "cells": msh.topology.index_map(2).size_global,
        "dofs": V.dofmap.index_map.size_global * V.dofmap.index_map_bs,
        "l2": float(l2), "energy": float(energy), "time": elapsed,
        "strain_energy": float(strain_energy), "exact_energy": float(exact_energy),
    }
    if points is not None:
        out["u_probe"] = _eval_at(msh, uh, points).tolist()
    return out


def _eval_at(msh, uh, points):
    """Point evaluation, the DOLFINx way: bounding-box tree then collision test."""
    from dolfinx import geometry

    p3 = np.zeros((len(points), 3))
    p3[:, :2] = points
    tree = geometry.bb_tree(msh, msh.topology.dim)
    candidates = geometry.compute_collisions_points(tree, p3)
    colliding = geometry.compute_colliding_cells(msh, candidates, p3)
    cells = np.array([colliding.links(i)[0] for i in range(len(points))], dtype=np.int32)
    return uh.eval(p3, cells)


def main(path):
    points = probe_points()
    runs = [solve(deg, 4 * n, n) for deg, n in RUNS]
    for r in runs:
        print(f"  P{r['degree']} {r['cells']:6d} cells {r['dofs']:7d} dofs  "
              f"L2 {r['l2']:.4e}  energy {r['energy']:.4e}  ({r['time']:.2f}s)", flush=True)

    deg, n = REFERENCE
    ref = solve(deg, 4 * n, n, points=points)
    print(f"  reference P{deg}: {ref['dofs']} dofs, L2 {ref['l2']:.3e}, "
          f"energy {ref['energy']:.3e}, strain energy {ref['strain_energy']:.12e} "
          f"(exact {ref['exact_energy']:.12e})", flush=True)

    data = {
        "runs": runs,
        "reference": ref,
        "probe_points": points.tolist(),
        "problem": {"rad": RAD, "side": SIDE, "traction": TRAC, "E": EMOD, "nu": NU},
        "versions": {
            "dolfinx": dolfinx.__version__,
            "basix": basix.__version__,
            "ufl": ufl.__version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    with open(path, "w") as fh:
        json.dump(data, fh)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fenicsx_reference.json")
