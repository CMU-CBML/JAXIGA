"""Poisson on a perforated plate using curved Gmsh triangles directly.

Run: python -m jaxiga.examples.gmsh_poisson_triangles
Requires: pip install -e '.[gmsh,post]' matplotlib
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import jaxiga as jx


def main():
    geometry = Path(__file__).with_name('geometries') / 'perforated_plate.geo'
    mesh = jx.read_gmsh(geometry, cell_type='simplex', mesh_size=2., geometry_order=2)
    space = jx.FunctionSpace(mesh)
    problem = jx.Poisson(
        space, source=lambda x, p: 1.,
        dirichlet=[jx.DirichletBC(0., where=name) for name in space.boundaries],
    )
    solution = jx.solve(problem, params={'a0': 1.})
    output = solution.to_vtk("gmsh_poisson_triangles.vtu", n=4)
    print(f"ParaView: {output}")
    print(f'{len(mesh)} quadratic triangles, {space.n_dofs} DOFs; '
          f'CAD boundary error: {mesh.geometry_error:.3g}')

    # Triangulate each reference cell separately, so the holes stay empty.
    samples = jx.grid(space, 7)
    basis = jx.evaluate(space, samples)
    local = mtri.Triangulation(samples.ref[:, 0], samples.ref[:, 1]).triangles
    triangles = np.concatenate([local + e * basis.n_q for e in range(space.n_elems)])
    xy = np.asarray(basis.x).reshape(-1, 2)
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    values = np.asarray(solution.at(basis=basis)).ravel()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    plot = ax.tripcolor(triangulation, values, shading='gouraud')
    fig.colorbar(plot, ax=ax, label='u')
    ax.set(title='Poisson on curved triangles: −Δu = 1, u = 0 on all boundaries',
           xlabel='x', ylabel='y', aspect='equal')
    plt.show()


if __name__ == '__main__':
    main()
