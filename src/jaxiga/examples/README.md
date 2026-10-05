# Small JAXIGA examples

Each script sets up one small problem, solves it and displays a Matplotlib
plot. All meshes and parameters are fixed in the file so you can edit them
directly. The scripts are independent: each contains its own imports, geometry,
boundary conditions and plotting code, and needs no paper code. The Gmsh
example additionally reads a small `.geo` geometry file shipped beside it.

Install the library and Matplotlib from a source checkout:

```bash
python -m pip install -e . matplotlib
```

Then start with the two-dimensional Poisson example:

```bash
python -m jaxiga.examples.poisson_2d
```

The examples are included in the installed package. You can also run a file
directly, or copy it to another directory and run it there with JAXIGA installed:

```bash
python src/jaxiga/examples/poisson_2d.py
```

| File | Problem and main API |
| --- | --- |
| [`poisson_2d.py`](poisson_2d.py) | Unit-square Poisson problem; geometry, `FunctionSpace`, boundary conditions, `solve`, error and field plot |
| [`poisson_methods.py`](poisson_methods.py) | One 1D Poisson problem solved with Galerkin, energy minimisation and collocation |
| [`linear_elasticity.py`](linear_elasticity.py) | A plate with a circular hole; four joined NURBS patches, plane stress and a von Mises stress plot |
| [`gmsh_poisson_triangles.py`](gmsh_poisson_triangles.py) | Poisson on the perforated plate using quadratic Gmsh triangles directly, with a field plot |
| [`gmsh_plate.py`](gmsh_plate.py) | Gmsh CAD input: a plate with unequal holes and an offset rounded slot; named boundaries, quad mesh and stress plot |
| [`gmsh_rational_plate.py`](gmsh_rational_plate.py) | Exact conic boundaries, mixed quad/triangle elasticity, mesh/stress plots and ParaView export |
| [`gmsh_cylinder_3d.py`](gmsh_cylinder_3d.py) | Internally pressurized quarter cylinder with exact cylindrical walls, mixed hex/tet coupling, 3D mesh/stress plots and ParaView export |
| [`blade_3d.py`](blade_3d.py) | Actual IGAPack NURBS turbine blade, collapsed-edge constraints, bending, mesh/stress plots and ParaView export |
| [`connecting_rod_3d.py`](connecting_rod_3d.py) | Actual IGAPack 25-patch connecting rod, optional mixed hex/tet shaft, refinement, mesh/stress plots and ParaView export |
| [`gmsh_plate_triangles.py`](gmsh_plate_triangles.py) | The same plane-stress tension problem using curved triangles; mesh and von Mises stress plotted side by side |
| [`differentiable_poisson.py`](differentiable_poisson.py) | `jax.value_and_grad` through a diffusion solve, checked against finite differences |
| [`darcy_inverse.py`](darcy_inverse.py) | A custom `Problem` energy and two permeability parameters recovered from synthetic pressures |
| [`material_identification.py`](material_identification.py) | Inclusion identification from full-field displacements (Mechanical MNIST): measured edges as Dirichlet data, a 2,601-parameter `PointField` stiffness and L-BFGS-B with adjoint gradients |
| [`adaptive_poisson.py`](adaptive_poisson.py) | Three solve–estimate–refine cycles using `adapt`, with the final solution and mesh |
| [`kirchhoff_plate.py`](kirchhoff_plate.py) | A clamped plate under uniform loading, using `KirchhoffPlate` and `ClampedBC` |
| [`scordelis_lo_roof.py`](scordelis_lo_roof.py) | Scordelis–Lo roof with `KirchhoffLoveShell` on an exact NURBS cylinder: diaphragm supports, self-weight, convergence table and deformed shape |

Read them in roughly this order. The Poisson example uses cubic splines on
8 × 8 elements; the method comparison uses eight 1D elements. These are
learning examples with small discretizations, rather than convergence studies.
The elasticity example uses four cubic NURBS patches with 8 × 8 elements each.
The bottom edge is fixed and the top edge is displaced vertically; the hole and
lateral edges are traction-free. Its plot draws elements individually to retain
the hole, and shows stress on the undeformed geometry.
The Darcy observations are noiseless and generated with the same model used
for recovery. Its purpose is to demonstrate the inverse-problem workflow.
The material-identification example runs on a synthetic specimen (a ring-shaped
stiff inclusion with noisy displacements) unless you pass a specimen file from
the experimental Mechanical MNIST data set, for example
`python -m jaxiga.examples.material_identification 016.npz`. For real specimens
it uses the stiffness ratio of the material pair and prints the Dice score of the
thresholded field against the released label.

Each script prints an error, solver summary or other numerical result and opens
one figure. Close the figure to finish. To save it, add
`plt.savefig("result.png", dpi=150)` before `plt.show()`. Matplotlib is the only
extra plotting dependency; these examples do not require PyVista or VTK.

Importing an example does not run it. All execution is in `main()`, which you
can also call from a notebook or another script.

For the Gmsh example, install the optional mesher and run:

```bash
python -m pip install -e '.[gmsh,post]' matplotlib
python -m jaxiga.examples.gmsh_plate
```

Edit `geometries/perforated_plate.geo` to change the dimensions, holes or slot.
The example requests `mesh_size="auto"` and cubic geometry, then uses
`mesh.refine()` to insert knots and refit new boundary samples to CAD. For a
fixed size, use `mesh_size=2.0`; larger values request fewer patches.
Meshing tries Quasi-structured Quad first, then Blossom-Quad, then
standard unstructured meshing if generation or quality checks fail.
The script prints the method, minimum scaled Jacobian and sampled CAD error.
Curved Gmsh geometry is preserved by a Lagrange-to-Bernstein transformation.
The holes remain polynomial approximations; unlike `linear_elasticity.py`, they
are not exact rational circles. Ordinary `patch.refine()` preserves the current
polynomial, while `mesh.refine()` resamples CAD. The [Gmsh guide](../../../docs/gmsh.md)
describes automatic sizing, curved conversion, CAD refitting and quality checks.

For direct triangle support (without conversion to quads), run:

```bash
python -m jaxiga.examples.gmsh_poisson_triangles
python -m jaxiga.examples.gmsh_plate_triangles
```

Both use `cell_type="simplex"`, a fixed coarse mesh size and quadratic curved
geometry. The elasticity example matches the quad example's loading and
material: plane stress, a fixed left edge, horizontal traction of 10 on the
right edge, `E=1e5` and `nu=0.3`. Its left panel draws the actual curved cell
edges; its right panel shows von Mises stress on the undeformed plate.
In 3D the same import option produces tetrahedra from `.geo` or STEP
solids. See [the Gmsh guide](../../../docs/gmsh.md) for scope and limitations.

The Gmsh scripts also write `.vtu` files in the working directory. Open these
VTK XML files directly in ParaView. Elasticity exports include a `displacement`
vector for **Warp By Vector**, component fields, `von_mises` and `element_id`.
The VTK files contain the full mesh, including volume cells in 3D.

To try rational CAD reconstruction and mixed-element coupling:

```bash
python -m jaxiga.examples.gmsh_rational_plate
python -m jaxiga.examples.gmsh_cylinder_3d
```

The 2D script joins quads to rational triangles while retaining exact circular
holes. The 3D script starts from the supplied `quarter_cylinder.geo`, reconstructs
exact cylindrical walls, and joins one hex to six rational tetrahedra under
internal pressure. Both plot the actual cell edges and the stress field. These
small meshes demonstrate the APIs; they are not mesh-converged stress studies.
The [Gmsh guide](../../../docs/gmsh.md) explains supported CAD families and
mixed-interface restrictions.

Add `--refine 1` to either command for one uniform subdivision level (or use
`--refine 2` for two). Each 2D cell becomes four children and each 3D cell eight,
with the same degrees, exact rational geometry and boundary labels. The scripts
use `patches = jx.refine_cells(patches, n=refinements)` before building the space;
both the plots and `.vtu` files then show the refined mesh. The 3D mesh grows
from one hex and six tets to eight hexes and 48 tets after one level.

The cylinder defaults to Jacobi-preconditioned CG and chunked assembly, and
prints setup/solve progress and its relative residual. Use `--refine 2` for
448 cells, `--no-plot` to write VTK without a plot window, or `--solver scipy`
to compare with direct sparse LU. Failed convergence stops the example before
export. The [Gmsh guide](../../../docs/gmsh.md) includes the solver configuration.

For the more complex IGAPack geometries:

```bash
python -m jaxiga.examples.blade_3d
python -m jaxiga.examples.connecting_rod_3d --refine 1
python -m jaxiga.examples.connecting_rod_3d --mixed
```

Both include their NURBS data and need no MATLAB or Gmsh. They use CG by default,
accept `--solver scipy` and `--no-plot`, and save displacement plus cell-average
von Mises stress. The [IGAPack geometry guide](../../../docs/igapack.md) explains
the original geometry sources, loading, collapsed blade edges and refinement.
The connecting rod also accepts `--maxiter` and `--tol` to control CG on finer
meshes; its default iteration budget is 50,000.
