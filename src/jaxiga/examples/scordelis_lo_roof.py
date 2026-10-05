"""Scordelis-Lo roof: a Kirchhoff-Love shell benchmark of the shell obstacle course.

A cylindrical roof (radius 25, length 50, opening angle 80 degrees, thickness
0.25) is supported by rigid diaphragms at its curved ends and loaded by its
self-weight, 90 per unit area. The longitudinal edges are free. The quantity of
interest is the vertical displacement at the midpoint of a free edge; 0.3024
is the usual reference value (MacNeal and Harder, 1985).

The midsurface is an exact NURBS cylinder; the displacement uses the same
rational basis after degree elevation and knot insertion.

Usage:
    python -m jaxiga.examples.scordelis_lo_roof
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxiga as jx
from jaxiga.geometry.nurbs import Patch
from jaxiga.space.pointset import PointSet

RADIUS, LENGTH, HALF_ANGLE, THICKNESS = 25.0, 50.0, np.deg2rad(40.0), 0.25
YOUNG, POISSON, WEIGHT = 4.32e8, 0.0, 90.0
REFERENCE = 0.3024


def roof_patch():
    """Quadratic x linear NURBS surface; u runs around the arc, v along the axis (y)."""
    c, s = np.cos(HALF_ANGLE), np.sin(HALF_ANGLE)
    arc = [(-RADIUS * s, RADIUS * c), (0.0, RADIUS / c), (RADIUS * s, RADIUS * c)]
    points = [[x, y, z] for y in (0.0, LENGTH) for x, z in arc]
    weights = [1.0, c, 1.0] * 2
    return Patch.create(knots=([0, 0, 0, 1, 1, 1], [0, 0, 1, 1]), degree=(2, 1),
                        ctrl_pts=np.array(points), weights=np.array(weights),
                        labels={"u0": "free_left", "u1": "free_right",
                                "v0": "diaphragm_front", "v1": "diaphragm_back"})


def solve_roof(degree, elements):
    """Solve on ``elements x elements`` elements of degree ``degree``."""
    knots = np.linspace(0, 1, elements + 1)[1:-1]
    patch = roof_patch().elevate(degree).insert_knots(knots, knots)
    space = jx.FunctionSpace(patch, vec=3)
    corner = np.array([-RADIUS * np.sin(HALF_ANGLE), 0.0, RADIUS * np.cos(HALF_ANGLE)])
    dirichlet = [jx.DirichletBC(0.0, where=side, component=c)
                 for side in ("diaphragm_front", "diaphragm_back") for c in (0, 2)]
    # The diaphragms leave rigid axial translation free; fix u_y at one corner.
    dirichlet.append(jx.DirichletBC(
        0.0, where=lambda x: np.linalg.norm(np.asarray(x) - corner, axis=-1) < 1e-8, component=1))
    problem = jx.KirchhoffLoveShell(space, dirichlet=dirichlet,
                                    source=lambda x, p: jnp.array([0.0, 0.0, -WEIGHT]))
    solution = jx.solve(problem, params=problem.parameters(YOUNG, POISSON, THICKNESS))
    return space, solution


def displacement_at(space, solution, u, v):
    """Displacement at parametric point (u, v)."""
    boxes = np.asarray(space.elem_vertex)
    inside = np.flatnonzero((boxes[:, 0] <= u) & (u <= boxes[:, 2]) & (boxes[:, 1] <= v) & (v <= boxes[:, 3]))
    e = int(inside[0]); lo, hi = boxes[e, :2], boxes[e, 2:]
    ref = 2.0 * (np.array([u, v]) - lo) / (hi - lo) - 1.0
    return np.asarray(solution.at(PointSet(elems=np.array([e]), ref=ref[None, :], tag="grid")))[0, 0]


def main():
    print(" p  elements   dofs    u_z(A)   error vs 0.3024")
    for degree in (2, 3, 4):
        for elements in (4, 8, 16, 32):
            space, solution = solve_roof(degree, elements)
            uz = -displacement_at(space, solution, 0.0, 0.5)[2]
            print(f" {degree}  {elements:4d}  {space.n_dofs:7d}   {uz:.5f}   {100 * (uz / REFERENCE - 1):+.2f}%")

    # Deformed shape (displacements magnified) coloured by vertical displacement.
    space, solution = solve_roof(3, 16)
    from jaxiga.space import pointset as P
    from jaxiga.space.evaluation import evaluate
    ps = P.grid(space, 6)
    x = np.asarray(evaluate(space, ps).x).reshape(-1, 3)
    u = np.asarray(solution.at(ps)).reshape(-1, 3)
    deformed = x + 10.0 * u
    ax = plt.figure(figsize=(7, 5)).add_subplot(projection="3d")
    art = ax.scatter(deformed[:, 0], deformed[:, 1], deformed[:, 2], c=u[:, 2], s=2, cmap="viridis")
    plt.colorbar(art, label="vertical displacement $u_z$", shrink=0.7)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Scordelis-Lo roof, deformed (x10)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
