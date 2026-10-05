"""3D linear elasticity on a cube.

Replaces Elast3D_Cube_IGA_Spline.py. The 3D constitutive matrix is selected
automatically from the space dimension, so the problem definition is identical
to the 2D one.
"""

import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.methods._common import reactions

DEG = 2
REFINEMENTS = 2

E, NU = 1e5, 0.3
STRETCH = 0.01

patch = jx.primitives.cuboid([0, 0, 0], [1, 1, 1]).elevate(DEG).refine(REFINEMENTS)
V = jx.FunctionSpace(patch, vec=3)
print(V.summary())

problem = jx.LinearElasticity(
    V,
    dirichlet=[
        jx.DirichletBC([0.0, 0.0, 0.0], where="bottom"),
        jx.DirichletBC([0.0, STRETCH, 0.0], where="top"),
    ],
)
params = {"E": E, "nu": NU}
sol = jx.solve(problem, params=params)
print(sol.summary())

# The loaded faces are clamped laterally but the sides are free, so the
# apparent stiffness sits between the two textbook limits: E (free lateral
# contraction, uniaxial stress) and the constrained modulus
# E(1-nu)/((1+nu)(1-2nu)) (fully confined, uniaxial strain).
top = np.asarray(V.boundaries["top"].dofs)
force = float(jnp.sum(reactions(sol, params=params)[3 * top + 1]))
uniaxial_stress = E * STRETCH
uniaxial_strain = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU)) * STRETCH
print(f"\nreaction F_y            = {force:.4f}")
print(f"uniaxial stress  E*u    = {uniaxial_stress:.4f}  (lower bound)")
print(f"uniaxial strain  M*u    = {uniaxial_strain:.4f}  (upper bound)")
assert uniaxial_stress < force < uniaxial_strain, "reaction outside the physical bounds"
print("reaction lies between the two limits, as end constraint requires")

pts, vals = sol.sample(n=3)
pts, vals = np.asarray(pts), np.asarray(vals)
mid = np.abs(pts[:, 1] - 0.5) < 1e-9
print(f"u_y at mid-height: [{vals[mid, 1].min():.6f}, {vals[mid, 1].max():.6f}] "
      f"(expected {STRETCH / 2:.6f})")

sol.to_vtk("elast3d_cube", n=3, fields=["von_mises"])
print("wrote elast3d_cube.vtu")
