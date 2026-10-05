"""Plate with a circular hole under far-field tension (the Kirsch problem).

The domain is (-4, 4)^2 minus a unit circular hole, modelled as four NURBS
quadrants. The exact Kirsch traction is applied on the outer boundary, so the
problem is pure Neumann; four control points on the hole are pinned to remove
the rigid-body modes (u_y on the x-axis, u_x on the y-axis).

The setup defines the material law and boundary conditions through the current API.
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx

DEG = 5
REFINEMENTS = 4

E, NU = 1e5, 0.3
RAD, SIDE, TRACTION = 1.0, 4.0, 10.0
TOL = 1e-9


# -- exact Kirsch solution -------------------------------------------------


def exact_displacement(pts):
    x, y = np.asarray(pts)[:, 0], np.asarray(pts)[:, 1]
    r, th = np.hypot(x, y), np.arctan2(y, x)
    ux = ((1 + NU) / E * TRACTION * (
        r * np.cos(th) / (1 + NU) + 2 * RAD**2 * np.cos(th) / ((1 + NU) * r)
        + RAD**2 * np.cos(3 * th) / (2 * r) - RAD**4 * np.cos(3 * th) / (2 * r**3)))
    uy = ((1 + NU) / E * TRACTION * (
        -NU * r * np.sin(th) / (1 + NU) - (1 - NU) * RAD**2 * np.sin(th) / ((1 + NU) * r)
        + RAD**2 * np.sin(3 * th) / (2 * r) - RAD**4 * np.sin(3 * th) / (2 * r**3)))
    return np.stack([ux, uy], axis=1)


def _stress(x, y, lib):
    """Kirsch stress in Cartesian Voigt form; ``lib`` is numpy or jax.numpy."""
    r, th = lib.hypot(x, y), lib.arctan2(y, x)
    srr = (TRACTION / 2 * (1 - RAD**2 / r**2)
           + TRACTION / 2 * (1 - 4 * RAD**2 / r**2 + 3 * RAD**4 / r**4) * lib.cos(2 * th))
    stt = (TRACTION / 2 * (1 + RAD**2 / r**2)
           - TRACTION / 2 * (1 + 3 * RAD**4 / r**4) * lib.cos(2 * th))
    srt = -TRACTION / 2 * (1 + 2 * RAD**2 / r**2 - 3 * RAD**4 / r**4) * lib.sin(2 * th)
    c, s = lib.cos(th), lib.sin(th)
    A = lib.array([[c**2, s**2, 2 * s * c],
                   [s**2, c**2, -2 * s * c],
                   [-s * c, s * c, c**2 - s**2]])
    return lib.linalg.solve(A, lib.stack([srr, stt, srt]) if lib is jnp else [srr, stt, srt])


def exact_stress(pts):
    return np.stack([_stress(x, y, np) for x, y in np.asarray(pts)])


def traction(x, n, p):
    """Outer-boundary traction; evaluated under vmap, so it must be traceable."""
    s = _stress(x[0], x[1], jnp)
    return jnp.stack([n[0] * s[0] + n[1] * s[2], n[0] * s[2] + n[1] * s[1]])


# -- geometry and problem --------------------------------------------------

patches = [
    jx.primitives.plate_with_hole_quadrant(RAD, SIDE, q).elevate(DEG).refine(REFINEMENTS)
    for q in (2, 3, 4, 1)
]
V = jx.FunctionSpace(patches, vec=2)
print(V.summary())

on_hole = lambda x: np.abs(np.hypot(x[:, 0], x[:, 1]) - RAD) < TOL

problem = jx.LinearElasticity(
    V,
    plane="stress",
    dirichlet=[
        # pin u_y where the hole meets the x-axis, u_x where it meets the y-axis
        jx.DirichletBC(0.0, where=lambda x: on_hole(x) & (np.abs(x[:, 1]) < TOL), component=1),
        jx.DirichletBC(0.0, where=lambda x: on_hole(x) & (np.abs(x[:, 0]) < TOL), component=0),
    ],
    neumann=[jx.Neumann(traction, where="outer")],
)

params = {"E": E, "nu": NU}
sol = jx.solve(problem, params=params)

print("relative L2 error:    ", jx.errornorm(sol, exact_displacement, "L2"))
print("relative energy error:", jx.errornorm(sol, exact_displacement, "energy",
                                             exact_grad=exact_stress))

sol.to_vtk("elast2d_platewhole", n=4, fields=["stress", "von_mises"])
print("wrote elast2d_platewhole.vtu")
