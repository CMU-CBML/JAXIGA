"""Unit square stretched vertically by a prescribed displacement.

Fixed at the bottom, pulled to u_y = 0.1 at the top.
"""

import numpy as np

import jaxiga as jx

DEG = 4
REFINEMENTS = 4

E, NU = 1e5, 0.3
LENGTH, WIDTH, PULL = 1.0, 1.0, 0.1

patch = jx.primitives.rectangle(0.0, 0.0, LENGTH, WIDTH).elevate(DEG).refine(REFINEMENTS)
V = jx.FunctionSpace(patch, vec=2)
print(V.summary())

problem = jx.LinearElasticity(
    V,
    plane="stress",
    dirichlet=[
        jx.DirichletBC([0.0, 0.0], where="bottom"),
        jx.DirichletBC([0.0, PULL], where="top"),
    ],
)

sol = jx.solve(problem, params={"E": E, "nu": NU})
print(sol.summary())

# Uniaxial stretch: u_y should vary linearly from 0 to PULL.
pts, vals = sol.sample(n=4)
uy_at_top = vals[np.abs(np.asarray(pts)[:, 1] - WIDTH) < 1e-9][:, 1]
print(f"u_y on the top edge: min={float(uy_at_top.min()):.6f} "
      f"max={float(uy_at_top.max()):.6f} (prescribed {PULL})")

probe = np.array([[0.5, 0.25], [0.5, 0.5], [0.5, 0.75]])
print("u_y along the centreline:", np.asarray(sol.probe(probe))[:, 1].round(6))

sol.to_vtk("elast2d_tension_plate", n=4, fields=["stress", "von_mises"])
print("wrote elast2d_tension_plate.vtu")
