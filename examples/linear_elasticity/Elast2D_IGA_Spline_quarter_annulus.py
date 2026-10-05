"""Quarter annulus under internal pressure-like loading.

Symmetry conditions on the two straight edges, exact traction on the inner
arc.
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx

DEG = 5
REFINEMENTS = 4

E, NU = 1e5, 0.3
R_INT, R_EXT, PRESSURE = 1.0, 4.0, 10.0


def _radial_solution(r, lib):
    """Lame solution for a thick-walled cylinder under internal pressure."""
    a2, b2 = R_INT**2, R_EXT**2
    srr = PRESSURE * a2 / (b2 - a2) * (1 - b2 / r**2)
    stt = PRESSURE * a2 / (b2 - a2) * (1 + b2 / r**2)
    return srr, stt


def exact_displacement(pts):
    pts = np.asarray(pts)
    r, th = np.hypot(pts[:, 0], pts[:, 1]), np.arctan2(pts[:, 1], pts[:, 0])
    a2, b2 = R_INT**2, R_EXT**2
    # plane stress radial displacement
    ur = PRESSURE * a2 * r / (E * (b2 - a2)) * ((1 - NU) + (1 + NU) * b2 / r**2)
    return np.stack([ur * np.cos(th), ur * np.sin(th)], axis=1)


def exact_stress(pts):
    pts = np.asarray(pts)
    r, th = np.hypot(pts[:, 0], pts[:, 1]), np.arctan2(pts[:, 1], pts[:, 0])
    srr, stt = _radial_solution(r, np)
    c, s = np.cos(th), np.sin(th)
    return np.stack([srr * c**2 + stt * s**2,
                     srr * s**2 + stt * c**2,
                     (srr - stt) * s * c], axis=1)


def traction(x, n, p):
    """Traction from internal pressure on the inner arc.

    ``n`` is the outward normal of the *solid*, which on the inner arc points
    into the hole (-r_hat). A pressure pushing the wall outwards is therefore
    ``-p n``, giving ``sigma_rr = -p`` there.
    """
    return -PRESSURE * n


patch = jx.primitives.quarter_annulus(R_INT, R_EXT).elevate(DEG).refine(REFINEMENTS)
V = jx.FunctionSpace(patch, vec=2)
print(V.summary())

problem = jx.LinearElasticity(
    V,
    plane="stress",
    dirichlet=[
        jx.DirichletBC(0.0, where="sym_y", component=1),  # u_y = 0 on y = 0
        jx.DirichletBC(0.0, where="sym_x", component=0),  # u_x = 0 on x = 0
    ],
    neumann=[jx.Neumann(traction, where="inner")],
)

sol = jx.solve(problem, params={"E": E, "nu": NU})

print("relative L2 error:    ", jx.errornorm(sol, exact_displacement, "L2"))
print("relative energy error:", jx.errornorm(sol, exact_displacement, "energy",
                                             exact_grad=exact_stress))

sol.to_vtk("elast2d_quarter_annulus", n=4, fields=["stress", "von_mises"])
print("wrote elast2d_quarter_annulus.vtu")
