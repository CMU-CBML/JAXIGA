"""Independent JAX-FEM plate-with-a-hole work--precision benchmark.

Run this file with the pinned ``jaxfem-benchmark`` environment described by
``jaxfem-environment.yml``.  It writes plain JSON so JAX-FEM's PETSc/NumPy
stack never shares an interpreter with JAXIGA.

For a controlled comparison, JAX-FEM and the P2 FEniCSx curve use the same
quadratic triangular meshes, plane-stress law, quadrature order and boundary
conditions, while JAX-FEM and JAXIGA use the same SciPy SuperLU backend,
warm-up exclusion and median timing.  Only TRI6 is reported: JAX-FEM's TRI3
element is isoparametric and therefore cannot represent the P2 curved geometry
used by the existing FEniCSx benchmark.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import sys
import time

import basix
import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jax_fem import logger
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver
from petsc4py import PETSc


RAD, SIDE, TRAC, EMOD, NU = 1.0, 4.0, 10.0, 1.0e5, 0.3
MU = EMOD / (2.0 * (1.0 + NU))
LAM = EMOD * NU / (1.0 - NU**2)  # plane stress
RUNS = (4, 8, 16, 32)
REPEATS = 5
JAXFEM_COMMIT = "912632b5993cd09f3776e2aa773547d39f8134a2"

jax.config.update("jax_enable_x64", True)
logger.setLevel(logging.ERROR)


def phi(theta, t):
    """Transfinite map shared mathematically with the FEniCSx benchmark."""
    direction = np.stack([np.cos(theta), np.sin(theta)], -1)
    scale = SIDE / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
    return (1.0 - t)[..., None] * (RAD * direction) + t[..., None] * (
        scale[..., None] * direction
    )


def quadratic_mesh(nth, nt):
    """Return the P2 annular mesh in MeshIO/JAX-FEM TRI6 node order."""
    if nth % 8:
        raise ValueError("nth must be a multiple of eight")

    ntheta, nr = 2 * nth, 2 * nt
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    radial = np.arange(nr + 1) / nr
    node = np.arange(ntheta * (nr + 1), dtype=np.int64).reshape(nr + 1, ntheta).T
    grid = np.stack(np.meshgrid(theta, radial, indexing="ij"), -1).reshape(-1, 2)
    points = np.empty((ntheta * (nr + 1), 2))
    points[node.reshape(-1)] = phi(grid[:, 0], grid[:, 1])

    def nd(i, j):
        return node[i % ntheta, j]

    cells = []
    for i in range(nth):
        for j in range(nt):
            i0, i2, j0, j2 = 2 * i, 2 * i + 2, 2 * j, 2 * j + 2
            triangles = (
                # Counter-clockwise orientation: JAX-FEM uses det(J), not
                # abs(det(J)), in its volume weights.
                ((i0, j0), (i2, j2), (i2, j0)),
                ((i0, j0), (i0, j2), (i2, j2)),
            )
            for v0, v1, v2 in triangles:
                # MeshIO: vertices, then edges (0,1), (1,2), (2,0).
                e01 = ((v0[0] + v1[0]) // 2, (v0[1] + v1[1]) // 2)
                e12 = ((v1[0] + v2[0]) // 2, (v1[1] + v2[1]) // 2)
                e20 = ((v2[0] + v0[0]) // 2, (v2[1] + v0[1]) // 2)
                cells.append([nd(*p) for p in (v0, v1, v2, e01, e12, e20)])

    return Mesh(points, np.asarray(cells, dtype=np.int32), ele_type="TRI6")


def exact_stress(x):
    r = jnp.hypot(x[0], x[1])
    theta = jnp.arctan2(x[1], x[0])
    a2, a4 = RAD**2 / r**2, RAD**4 / r**4
    srr = TRAC / 2 * (1 - a2) + TRAC / 2 * (1 - 4 * a2 + 3 * a4) * jnp.cos(2 * theta)
    stt = TRAC / 2 * (1 + a2) - TRAC / 2 * (1 + 3 * a4) * jnp.cos(2 * theta)
    srt = -TRAC / 2 * (1 + 2 * a2 - 3 * a4) * jnp.sin(2 * theta)
    c, s = jnp.cos(theta), jnp.sin(theta)
    rotation = jnp.array([[c, -s], [s, c]])
    return rotation @ jnp.array([[srr, srt], [srt, stt]]) @ rotation.T


def exact_displacement(x):
    r = jnp.hypot(x[0], x[1])
    theta = jnp.arctan2(x[1], x[0])
    c, s = jnp.cos(theta), jnp.sin(theta)
    ux = (1 + NU) / EMOD * TRAC * (
        r * c / (1 + NU)
        + 2 * RAD**2 * c / ((1 + NU) * r)
        + RAD**2 * jnp.cos(3 * theta) / (2 * r)
        - RAD**4 * jnp.cos(3 * theta) / (2 * r**3)
    )
    uy = (1 + NU) / EMOD * TRAC * (
        -NU * r * s / (1 + NU)
        - (1 - NU) * RAD**2 * s / ((1 + NU) * r)
        + RAD**2 * jnp.sin(3 * theta) / (2 * r)
        - RAD**4 * jnp.sin(3 * theta) / (2 * r**3)
    )
    return jnp.array([ux, uy])


def at_point(x0, x1):
    def predicate(x):
        return jnp.isclose(x[0], x0, atol=1e-9) & jnp.isclose(x[1], x1, atol=1e-9)

    return predicate


def zero(_):
    return 0.0


def outer_boundary(x):
    return jnp.isclose(jnp.max(jnp.abs(x)), SIDE, atol=1e-9)


class PlateWithHole(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            strain = 0.5 * (u_grad + u_grad.T)
            return 2.0 * MU * strain + LAM * jnp.trace(strain) * jnp.eye(2)

        return stress

    def get_surface_maps(self):
        def negative_traction(_u, x):
            # JAX-FEM adds surface maps to its residual, so return -sigma*n.
            use_x = jnp.abs(x[0]) >= jnp.abs(x[1])
            normal = jnp.where(
                use_x,
                jnp.array([jnp.sign(x[0]), 0.0]),
                jnp.array([0.0, jnp.sign(x[1])]),
            )
            return -(exact_stress(x) @ normal)

        return [negative_traction]


def build_problem(n):
    mesh = quadratic_mesh(4 * n, n)
    dirichlet = [
        [at_point(RAD, 0.0), at_point(-RAD, 0.0), at_point(0.0, RAD), at_point(0.0, -RAD)],
        [1, 1, 0, 0],
        [zero, zero, zero, zero],
    ]
    return PlateWithHole(
        mesh,
        vec=2,
        dim=2,
        ele_type="TRI6",
        quadrature_order=8,
        dirichlet_bc_info=dirichlet,
        location_fns=[outer_boundary],
    )


def errors(problem, solution):
    fe = problem.fes[0]
    x = fe.get_physical_quad_points()
    weights = fe.JxW
    uh = fe.convert_from_dof_to_quad(solution)
    grad = fe.sol_to_grad(solution)
    exact_u = jax.vmap(jax.vmap(exact_displacement))(x)
    exact_sigma = jax.vmap(jax.vmap(exact_stress))(x)

    strain = 0.5 * (grad + jnp.swapaxes(grad, -1, -2))
    sigma = 2.0 * MU * strain + LAM * jnp.trace(strain, axis1=-2, axis2=-1)[..., None, None] * jnp.eye(2)
    delta_sigma = sigma - exact_sigma

    def compliance(stress):
        return jnp.stack(
            [
                jnp.stack([(stress[..., 0, 0] - NU * stress[..., 1, 1]) / EMOD,
                           (1 + NU) * stress[..., 0, 1] / EMOD], -1),
                jnp.stack([(1 + NU) * stress[..., 1, 0] / EMOD,
                           (stress[..., 1, 1] - NU * stress[..., 0, 0]) / EMOD], -1),
            ],
            -2,
        )

    l2 = jnp.sqrt(
        jnp.sum(weights * jnp.sum((uh - exact_u) ** 2, -1))
        / jnp.sum(weights * jnp.sum(exact_u**2, -1))
    )
    energy = jnp.sqrt(
        jnp.sum(weights * jnp.sum(delta_sigma * compliance(delta_sigma), (-2, -1)))
        / jnp.sum(weights * jnp.sum(exact_sigma * compliance(exact_sigma), (-2, -1)))
    )
    strain_energy = 0.5 * jnp.sum(weights * jnp.sum(sigma * strain, (-2, -1)))
    exact_energy = 0.5 * jnp.sum(
        weights * jnp.sum(exact_sigma * compliance(exact_sigma), (-2, -1))
    )
    return tuple(float(v) for v in (l2, energy, strain_energy, exact_energy))


# JAX-FEM offers several linear solvers. ``spsolve`` is SciPy SuperLU on the
# host; ``jax`` is the package's own default, Jacobi-preconditioned BiCGSTAB
# in JAX, which runs on whichever device JAX is using. The comparison should
# not hand JAX-FEM the host route while JAXIGA keeps its device one, so the
# route is selectable and both are reported.
LINEAR_OPTIONS = {
    "spsolve": {"spsolve_solver": {}},
    "jax": {"jax_solver": {}},
}
SOLVER = os.environ.get("JAXFEM_LINEAR", "jax")


def run_case(n, repeats):
    problem = build_problem(n)
    options = {
        "newton": {
            "tol": 1e-10,
            "rel_tol": 1e-10,
            "linear": LINEAR_OPTIONS[SOLVER],
        }
    }

    # Exclude construction and compilation, exactly as in the paper benchmark.
    warm = solver(problem, solver_options=options)[0]
    warm.block_until_ready()
    timings = []
    solution = warm
    for _ in range(repeats):
        start = time.perf_counter()
        solution = solver(problem, solver_options=options)[0]
        solution.block_until_ready()
        timings.append(time.perf_counter() - start)

    l2, energy, strain_energy, exact_energy = errors(problem, solution)
    return {
        "degree": 2,
        "element": "TRI6",
        "nth": 4 * n,
        "nt": n,
        "cells": 8 * n * n,
        "dofs": int(problem.num_total_dofs_all_vars),
        "l2": l2,
        "energy": energy,
        "time": float(np.median(timings)),
        "timings": timings,
        "strain_energy": strain_energy,
        "exact_energy": exact_energy,
    }


def main(path, *, quick=False):
    runs = []
    refinements = RUNS[:1] if quick else RUNS
    repeats = 1 if quick else REPEATS
    for n in refinements:
        result = run_case(n, repeats)
        runs.append(result)
        print(
            f"  TRI6 {result['cells']:5d} cells {result['dofs']:6d} dofs  "
            f"L2 {result['l2']:.4e}  energy {result['energy']:.4e}  "
            f"({result['time']:.3f}s)",
            flush=True,
        )

    output = {
        "runs": runs,
        "problem": {"rad": RAD, "side": SIDE, "traction": TRAC, "E": EMOD, "nu": NU},
        "protocol": {
            "mesh": "same P2 transfinite triangle mesh as FEniCSx",
            "quadrature_order": 8,
            "solver": {"spsolve": "scipy.sparse.linalg.spsolve (SuperLU)",
                       "jax": "jax.scipy.sparse.linalg.bicgstab, Jacobi (jax_fem default)"}[SOLVER],
            "warmup_excluded": True,
            "repeats": repeats,
        },
        "versions": {
            "jax_fem": importlib.metadata.version("jax-fem"),
            "jax_fem_commit": JAXFEM_COMMIT,
            "jax": jax.__version__,
            "jaxlib": importlib.metadata.version("jaxlib"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "basix": basix.__version__,
            "petsc": ".".join(map(str, PETSc.Sys.getVersion())),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    with open(path, "w") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(f"  wrote {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: reference_jaxfem.py OUTPUT.json [--quick]")
    main(sys.argv[1], quick="--quick" in sys.argv[2:])
