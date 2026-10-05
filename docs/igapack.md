# IGAPack blade and connecting-rod examples

These examples use the actual volumetric NURBS geometry stored with
`IGAPack/elasticity3D/Blade.m` and `ConnectingRodMP.m`. The two MAT geometry files
are bundled inside `jaxiga.examples`, so an installed package can run them
without MATLAB, Gmsh, the paper, or the IGAPack checkout. Their provenance is
recorded in [geometries/README.md](../src/jaxiga/examples/geometries/README.md).

```bash
python -m pip install -e '.[post]' 'matplotlib>=3.6'
python -m jaxiga.examples.blade_3d
python -m jaxiga.examples.connecting_rod_3d --refine 1
python -m jaxiga.examples.connecting_rod_3d --mixed
```

Both scripts plot the exact undeformed mesh beside cell-average von Mises
stress and save displacement vectors and stress to ParaView-readable VTK XML
files in the current directory. `--no-plot` suppresses the plot window but
keeps VTK export. `--solver scipy` selects direct sparse LU for comparison;
the default is Jacobi-preconditioned CG with chunked assembly. Each script
checks the actual residual and stops before exporting an unconverged result.

## Blade

[`blade_3d.py`](../src/jaxiga/examples/blade_3d.py) imports the `blade` struct
from `blade.mat`: one NURBS volume of degree `(2,2,1)` with 200 knot spans.
It reverses the first parameter to obtain positive volume orientation and
elevates to degree two in all directions, preserving the physical geometry.
The default has 200 elements and 2,112 displacement DOFs.

The root at `v=0` is fixed in all components, and the tip at `v=1` has traction
`(0,-0.001,0)`. The material is `E=1e5`, `nu=0.3`, using consistent units with
the supplied coordinates. This is a representative bending load; it does not
reproduce the original MATLAB script's unfinished/commented loading setup or
its adaptive PHT/GIFT analysis.

The leading and trailing parametric faces collapse to physical edges. Ordinary
independent coefficients on those faces can assign multiple displacement
values to the same edge. The example therefore uses:

```python
space = jx.FunctionSpace(patch, vec=3, collapse_degenerate=True)
```

This opt-in setting identifies a whole tensor boundary face that is constant
in a tangential direction and ties its repeated, equal-weight coefficients.
It preserves the geometry and makes the displacement single-valued on the
collapsed edge. Arbitrary coincident control points remain independent. This
is not a repair for folded geometry or a general treatment of singular CAD.
Simplex/mixed and hierarchical refinement with this setting are unsupported.

The Jacobian remains singular exactly on the collapsed edges. Integration uses
interior Gauss points; stresses are displayed/exported as volume-weighted cell
averages computed there. No arbitrary epsilon offset or geometry trimming is
used. Pointwise stress export on a space with collapsed boundaries raises an
error directing the caller to `cell_fields`.

`--refine 1` inserts knots uniformly into the NURBS patch, producing 1,600
elements and retaining the patch's spline continuity. It does not split the
blade into tetrahedra. The relative solver tolerance is `1e-6`, suitable for
this slender, poorly conditioned demonstration model; the CG solution is
checked against a direct solve in the tests. Output: **`blade_3d.vtu`**.

## Connecting rod

[`connecting_rod_3d.py`](../src/jaxiga/examples/connecting_rod_3d.py) imports
all 25 solids from `ConnRod.mat`. The 39 matching patch interfaces are coupled
strongly. Known negative parameter orientations are reversed and degrees are
elevated to a common quadratic tensor space. These operations do not alter
the geometry. The stored MAT file already contains the corrected shaft
interfaces; Python does not rerun the geometry-construction MATLAB scripts.

The model follows the MATLAB example's material (`E=4e5`, `nu=0.3`) and support
and load regions: both big-end cut faces are fixed, and traction `(0,0,-1)`
acts on the small-end bore. The original bore radii, curved shaft and stepped
thickness are retained. This uses NURBS/C0 mixed analysis rather than the
original PHT/GIFT discretization, so it is not presented as an IGAPack result
reproduction or a mesh-converged stress benchmark.

By default the coarse model has 25 elements and 1,098 displacement DOFs.
`--refine 1` uses knot insertion within each NURBS patch: 200 elements and
3,096 DOFs. Output: **`connecting_rod_3d.vtu`**.

With `--mixed`, the shaft's native degree-`(2,2,1)` rational map is restricted
to six degree-five tetrahedra; the other 24 patches remain quadratic hexes.
This gives 30 cells and 1,503 DOFs. Its quad/triangle face interfaces use the
same C0 coupling as the cylinder example. `--mixed --refine 1` applies uniform
`jx.refine_cells` subdivision to the whole mixed mesh, preserving the rational
geometry and matching face partitions. Mixed spaces can be much more expensive
than the native tensor space because of their higher simplex degree.
The native and mixed solver tolerances are `1e-8` and `1e-6`, respectively;
tests compare the mixed CG solution with direct sparse LU. Mixed output:
**`connecting_rod_mixed_3d.vtu`**.

The connecting-rod script accepts `--maxiter` (default 50,000) and `--tol`
to control CG. For example:

```bash
python -m jaxiga.examples.connecting_rod_3d --mixed --refine 2 --maxiter 50000 --tol 1e-6
```

The higher-degree tetrahedra and slender geometry can require many iterations
with Jacobi preconditioning. Increasing `--maxiter` allows more work without
changing the requested accuracy; increasing `--tol` relaxes the relative
residual target. The script still checks the actual residual before exporting.
An iteration limit is a budget, not a guarantee of convergence.
On CPU, assembled CG also recomputes the residual and can perform up to three
correction solves when finite-precision drift makes the recursive CG residual
too optimistic. These corrections share the same total `maxiter` budget.

## Reusing the import and output APIs

```python
patches = jx.read_matlab_nurbs("geometry.mat")
selected = jx.read_matlab_nurbs("geometry.mat", ["solid4", "solid1A"])
```

The reader understands MATLAB NURBS-toolbox structs containing `coefs`, `knots`
and `order`. It unweights homogeneous control points and preserves knot vectors,
weights and parameter orientation; optional variable names set the return
order. MATLAB v7.3/HDF5 is not supported: save these geometry structs with
`save -v7`. Geometry files are read as data, never executed as MATLAB code.
This is direct NURBS-volume import, separate from the Gmsh analytic CAD
reconstruction path; it does not add arbitrary trimmed STEP reconstruction.

`patch.reverse(axis)` reverses one parameter and swaps its side labels.
`jx.bezier_cells(patches)` extracts exact rational geometry on each knot span.
The plotting helper uses those cells for drawing multi-span tensor meshes,
leaving the analysis space unchanged. Building a *new* space on the extracted
cells instead gives C0 continuity between them.

```python
averages = solution.cell_average("von_mises")
solution.to_vtk("result.vtu", n=3, cell_fields=("von_mises",))
plot_mesh_field(solution, "von_mises", field_location="cell")
```

`cell_average` averages the named field itself with physical integration
weights. VTK stores these as **Cell Data**, along with `element_id`; displacement
remains Point Data. ParaView can color by cell-average `von_mises`, show Surface
With Edges, or apply Warp By Vector using `displacement`. Increasing the VTK
sampling `n` smooths displayed geometry but does not refine the analysis or the
cell-average stress field.
