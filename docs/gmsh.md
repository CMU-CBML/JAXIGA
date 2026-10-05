# Gmsh geometry and mesh input

JAXIGA can mesh `.geo` and STEP (`.step`/`.stp`) geometry, or read an existing
`.msh` file, and convert its cells into analysis patches. Quad/hex conversion
is the default. Use `cell_type="simplex"` for direct triangles/tetrahedra
([details and example](#direct-triangles-and-tetrahedra)). Gmsh is an optional
dependency; importing JAXIGA does not import or initialize Gmsh.

The following quad/hex sections describe the default tensor-product mode.

```bash
python -m pip install -e '.[gmsh]'
```

On the tested macOS Gmsh 4.15.2 / Python 3.14 environment, high-order 3D tests
can encounter a native shutdown fault with multiple OpenMP threads. Starting
Python with `OMP_NUM_THREADS=1` avoids the observed fault:

```bash
OMP_NUM_THREADS=1 python your_script.py
```

```python
import jaxiga as jx

mesh = jx.read_gmsh("part.geo", mesh_size="auto", geometry_order=3)
space = jx.FunctionSpace(mesh, vec=2)
print(len(mesh), mesh.min_scaled_jacobian, mesh.geometry_error)
```

`jx.io.read("part.geo", ...)` dispatches to the same reader. The returned
`GmshMesh` is a sequence, so `jx.FunctionSpace(mesh)` uses the imported patches
directly. The reader owns a temporary Gmsh session and opens no GUI. Finalize
any active Gmsh session before calling it; an active session is rejected without
modifying its models or options.

## A geometry worth trying

Run `python -m jaxiga.examples.gmsh_plate`. Its
[`perforated_plate.geo`](../src/jaxiga/examples/geometries/perforated_plate.geo)
defines a 12 × 6 plate with two unequal circular holes and an offset rounded
slot. Boolean operations construct the domain; physical groups name the fixed,
loaded and traction-free boundaries. The example displays the generated quad
mesh and the plane-stress solution under horizontal tension.

This exercises multiple holes, curved boundaries, unstructured connectivity
and boundary-name transfer without requiring a manually designed patch layout.
The example requests automatic coarse sizing and cubic geometry, then performs
one CAD-aware knot-refinement step. Its mesh plot samples curved element edges.
Geometry is editable in the `.geo` file independently of the analysis script.
The file's fixed-size setting (2.0) is used if `mesh_size` is omitted.

Increase the target size for fewer patches, or decrease it for a finer mesh:

```python
mesh = jx.read_gmsh("perforated_plate.geo", mesh_size=2.0, geometry_order=3)
fine = mesh.refine(1)
space = jx.FunctionSpace(fine, vec=2)
```

Here the coarse mesh defines the patches. `mesh.refine(1)` adds knots inside
each patch and refits boundary interpolation points to the original CAD.

## Curved polynomial geometry

For CAD input, `geometry_order=2` or `3` requests quadratic or cubic geometry.
The reader first creates a linear all-quad/all-hex topology, then uses Gmsh to
place high-order nodes on CAD. In 2D it also runs Gmsh's high-order optimizer to
repair distortion introduced by curving a coarse mesh. The default order is
still one when `geometry_order` is omitted.

For `.msh` input, the existing polynomial geometry is preserved. The reader
evaluates Gmsh's Lagrange basis and transforms its nodal coordinates into a full
tensor-product Bernstein control net with unit weights. This preserves the
Gmsh polynomial mapping to numerical precision; it does not replace curved
cells with straight ones. Complete quad/hex elements of degrees 1 through 8
and serendipity cells such as QUAD8/HEX20 are supported. CAD-only options cannot
be used to change the order of an imported `.msh`.

The resulting polynomial is an approximation to CAD. Exact rational circle or
NURBS reconstruction is a separate capability.

## Automatic coarse sizing

```python
mesh = jx.read_gmsh("part.step", mesh_size="auto", geometry_order=3,
                    geometry_tolerance=1e-3, max_sizing_steps=6)
print(mesh.mesh_size, mesh.sizing_history)
```

Automatic sizing starts at one quarter of the CAD bounding box's longest
extent and halves the target until the quality checks and a sampled
CAD-distance tolerance pass. The tolerance is an absolute distance in model
units; by default it is 0.001 times that longest extent. Failure after the
attempt limit raises an error with the sizing history. The method is a bounded
heuristic, not a search for the globally smallest mesh. Gmsh can still introduce
smaller cells near geometric features.

`geometry_error` measures the maximum sampled distance of polynomial boundary
curves/faces to their classified CAD entities, including off-node samples on
every knot span. It is not a certified Hausdorff bound, a PDE error estimate,
or a guarantee of accurate stress concentrations.

## CAD-aware refinement

```python
fine = mesh.refine(2)                       # Uniform knot insertion, CAD refit.
fine = mesh.elevate(4)                      # Raise degree, CAD refit.
fine = mesh.insert_knots([0.5], [0.5])       # Explicit knots for a 2D mesh.
unchanged = mesh.refine(1, to_cad=False)     # Preserve the current polynomial.
```

These mesh-level operations keep the patch layout and labels. They sample the
enriched spline at Greville points, project boundary samples onto their source
CAD curves/faces, and solve for new spline control points. The boundary
interpolation points lie on CAD to kernel precision; **control points are not
projected**, and the spline between interpolation points still approximates CAD.
Original mesh vertices are retained. In 3D, CAD curves take precedence at face
intersections. Interior samples retain the previous geometry mapping.

Refitted geometry must pass Jacobian, distortion and shared-interface checks.
An incompatible knot pattern or an invalid refit raises an error and leaves the
original mesh unchanged. Maximum boundary error need not decrease after every
small interpolation-space change; inspect the updated `geometry_error`.

Ordinary `Patch.refine`, `Patch.insert_knots` and `Patch.elevate` retain their
shape-preserving behavior and need no CAD. Mesh-level refitting changes the
geometry, so rebuild the `FunctionSpace` and solve again; an old solution is
not automatically transferred. Refitting is a NumPy/Gmsh setup operation, not
a differentiable CAD operation.

CAD-aware operations require the original `.geo`/STEP file and its dependencies
to remain available and unchanged. Changes to the main source file are detected
by a content hash; edits to files included by a `.geo` require a fresh import.
An existing Gmsh session is never overwritten. A standalone `.msh` has no source
CAD and supports only `to_cad=False`.

## Meshing and subdivision

For 2D `.geo` and STEP input, the reader tries these methods in order:

1. **Quasi-structured Quad** (`Mesh.Algorithm=11`).
2. **Frontal-Delaunay for Quads** (`Mesh.Algorithm=8`) with **Blossom-Quad
   recombination** (`Mesh.RecombinationAlgorithm=1`).
3. Standard **Frontal-Delaunay** (`Mesh.Algorithm=6`) with simple recombination
   (`Mesh.RecombinationAlgorithm=0`).

If generation, subdivision or cell-quality validation fails, it reopens the
geometry in a fresh Gmsh session and tries the next method.
Remaining triangles are subdivided into quads when `subdivide=True`. The same
quality threshold applies to every attempt. Invalid boundary labels and
unsupported input are reported directly; existing `.msh` files are never
remeshed on failure.

For 3D geometry, the file/Gmsh meshing settings apply, with unstructured Delaunay
tetrahedral meshing as the default. Gmsh's standard unstructured 3D algorithms
produce tetrahedra, not a general all-hex mesh: the hex fallback is tetrahedral
meshing followed by subdivision. Blossom applies to 2D quadrilaterals.

Use `quad_algorithm="blossom"` to start with the second method, or
`quad_algorithm="frontal_delaunay"` to use only the last method. This option
applies to 2D CAD input. Quasi-structured meshing can adjust prescribed
transfinite counts as part of its remeshing and refinement; select `"blossom"`
when those counts must be retained:

```python
mesh = jx.read_gmsh("structured.geo", quad_algorithm="blossom", subdivide=False)
```

The 2D algorithm choices above override
file-defined algorithm selections. 3D transfinite/extrusion settings are retained.
`mesh_size` optionally supplies a uniform target size; otherwise
the geometry file and Gmsh defaults control it. The generated mesh is held in
memory; no `.msh` file is written automatically.

With the default `subdivide=True`, a remaining mixed linear mesh is subdivided
into all quads or all hexes using Gmsh. A triangle becomes three quads and a
tetrahedron becomes four hexes. Existing quads/hexes can also be refined during
this operation. For a standalone `.msh`, new vertices are linearly interpolated
because the original CAD parametrization is unavailable.

```python
mesh = jx.read_gmsh("solid.step", mesh_size=1.0)  # Tet-to-hex fallback if needed.
mesh = jx.read_gmsh("quads.msh", subdivide=False)  # Require quad/hex input.
```

High-order mixed/simplex `.msh` cells are rejected rather than silently dropping curved nodes.
`mesh_size` is not accepted for an existing mesh. Surface meshes must be planar
and parallel to XY; a constant z coordinate is dropped. Embedded shells are not
supported. Solid meshes are three-dimensional.

## Physical groups and boundary conditions

In 2D, name curves; in 3D, name surfaces. For example, in a `.geo` file:

```text
Physical Curve("fixed") = {left_curve};
Physical Curve("loaded") = {right_curve};
Physical Surface("material") = {domain_surface};
```

Then use `jx.DirichletBC(..., where="fixed")` or
`jx.Neumann(..., where="loaded")`. Untagged external sides retain JAXIGA's
default `patch<index>/<side>` names. A physical boundary spanning several
entities becomes one named boundary. Two different physical names on the same
side are rejected because a patch side currently supports one label.

The returned object also exposes:

| Attribute | Meaning |
| --- | --- |
| `patches` | List of patches, in import order |
| `element_tags` | Gmsh cell tags after any subdivision, one per patch |
| `cell_groups` | Physical domain names mapped to patch indices |
| `boundary_groups` | Physical boundary names mapped to `(patch_index, side)` pairs |
| `physical_names` | Original `(dimension, physical_tag)` to name mapping |
| `scaled_jacobians` | Minimum sampled scaled Jacobian for each cell |
| `jacobian_lower_bounds` | Minimum determinant checks for each cell |
| `source`, `gmsh_version` | Input path and mesher version |
| `meshing_method` | `quasi_structured`, `blossom`, `frontal_delaunay`, `gmsh_3d`, or `existing_mesh` |
| `subdivided` | Whether mixed cells were subdivided into quads/hexes |
| `fallback_reason` | Named failures from earlier 2D attempts, or `None` |
| `mesh_size` | Explicit or automatically selected target size, or `None` |
| `geometry_error` | Maximum sampled CAD distance, or `None` for `.msh` |
| `sizing_history` | Automatic sizing attempts with size, error and quality diagnostics |

Unnamed physical groups receive `physical_<dimension>_<tag>` names. Tagged
internal interfaces remain in `boundary_groups`, but are glued by the function
space and are not exposed as external boundary conditions. Physical domain
groups are metadata; they do not automatically assign different materials.

A STEP file ordinarily needs a small `.geo` wrapper to assign analysis-specific
physical groups after importing the CAD entities. Direct STEP import is also
supported, but the importer does not infer support/load names from CAD faces.

## Geometry quality and scope

The importer normalizes reversed cell orientation and rejects repeated
vertices, duplicate cells, non-manifold face connectivity, non-finite
coordinates, non-positive Jacobians and excessive distortion. The default
minimum scaled Jacobian is `0.05`, sampled with at least `max(5, 2*p+3)` points
per direction, including corners. High-order and CAD-refitted polynomial
determinants are converted to Bernstein form on each knot span. Positive
coefficient bounds, tightened by subdivision when needed, validate the cells;
an inconclusive bound is rejected. The same check covers linear 3D cells.
Two-dimensional bilinear-cell determinants
attain their extrema at corners.

```python
mesh = jx.read_gmsh("part.geo", min_scaled_jacobian=0.1)
```

Setting the threshold to zero disables only the distortion threshold; invalid
Jacobian checks remain active. Mesh-level CAD refinement updates the quality
diagnostics. They do not track later manual edits to individual patches. These are cell checks, not a proof
that arbitrary disconnected mesh regions never overlap. `FunctionSpace` also
checks interface compatibility before joining patches.

By default, each mesh cell becomes a polynomial patch with unit weights. The resulting
space is C0 across patch interfaces. Keep neighbouring refinements and degrees
compatible. The sections below describe direct simplex support, rational CAD
reconstruction and mixed C0 coupling. C1 coupling remains unsupported.

The reader uses Gmsh's documented [meshing and subdivision
options](https://gmsh.info/doc/texinfo/gmsh.html#Mesh-options).


## Direct triangles and tetrahedra

Use `cell_type="simplex"` to keep a good simplex mesh as triangles/tetrahedra:

```python
import jaxiga as jx

mesh = jx.read_gmsh("plate.geo", cell_type="simplex",
                    mesh_size=2.0, geometry_order=2)
space = jx.FunctionSpace(mesh)
problem = jx.Poisson(
    space,
    source=lambda x, p: 1.0,
    dirichlet=[jx.DirichletBC(0.0, where=name) for name in space.boundaries],
)
solution = jx.solve(problem, params={"a0": 1.0})
```

The runnable [triangle example](../src/jaxiga/examples/gmsh_poisson_triangles.py)
uses the plate with two circular holes and a rounded slot. It plots the solution
without connecting triangles across the holes. For linear elasticity, run
`python -m jaxiga.examples.gmsh_plate_triangles`. This
[triangle elasticity example](../src/jaxiga/examples/gmsh_plate_triangles.py)
uses the same plane-stress material and tension loading as the quad example,
and plots the curved triangle mesh beside the von Mises stress field.
For tetrahedra, pass a solid
`.geo`/`.step` file or an existing tetrahedral `.msh` with the same option.
For an existing `.msh`, omit `mesh_size` and `geometry_order`.

- CAD uses Frontal-Delaunay triangles or Delaunay tetrahedra. Surface/volume
  transfinite and recombination constraints are removed for this mode. There
  is no quad/hex subdivision, and `subdivide` is ignored.
- Complete polynomial simplex geometries of degrees 1–4 are supported. A
  Lagrange-to-Bernstein transformation preserves their polynomial map, not just
  their corner vertices. Incomplete simplices, mixed degrees, and mixed
  simplex/tensor meshes are rejected. The default `cell_type="tensor"` retains
  the previous quad/hex behavior.
- The returned `SimplexPatch` cells use a total-degree Bernstein basis, with
  C0 continuity across matching edges/faces. Physical boundary names, domain
  groups, reversed cell orientations and high-order shared faces are handled.
  Imported Jacobians are independently bounded in Bernstein form, with
  subdivision when necessary; the scaled-Jacobian threshold is sampled on a
  barycentric lattice. These checks do not certify global non-overlap.
- Galerkin assembly, Dirichlet/Neumann data, physical first/second derivatives,
  solution sampling/probing, VTK output and JAX geometry/weight differentiation
  use the usual APIs. Simplex Gauss rules use a Duffy transform; `grid(V, n)`
  returns a barycentric lattice, not an `n` by `n` tensor grid.
- `mesh_size="auto"` uses the same coarse-first quality/CAD-distance heuristic.
  To subdivide the existing geometry, use `jx.refine_cells(mesh, n=1)`.
  To remesh from CAD instead, reimport with a smaller `mesh_size`. To raise degree while
  preserving the imported geometry, use `mesh.elevate(p, to_cad=False)` and
  rebuild the space. To obtain a higher-order CAD approximation, reimport with
  a higher `geometry_order`.
- Simplex CAD refitting, knot insertion, hierarchical/local h-refinement,
  Greville collocation, flux-jump error estimation and C1 coupling are not
  implemented. Tensor-specific operations reject simplex input explicitly.
  Fourth-order forms such as `KirchhoffPlate` need C1 coupling and are rejected.

Simplex patches can also be built without Gmsh:

```python
triangle = jx.primitives.triangle([[0, 0], [1, 0], [0, 1]]).elevate(3)
tetrahedron = jx.primitives.tetrahedron(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
).elevate(2)
```

Use positively oriented vertices. `SimplexPatch.create(degree, ctrl_pts,
weights=..., labels=...)` supports rational Bezier geometry with positive
weights; default polynomial Gmsh imports have unit weights. Control points are indexed by
`patch.multi_indices`, ordered lexicographically with descending barycentric
exponents. Face `fi` is opposite vertex `i` (`lambda_i=0`). The parameter domain
is `x >= 0, sum(x) <= 1`, with `lambda_0=1-sum(x)`.

For nonconstant essential boundary data, the existing `DirichletBC` behavior
still evaluates values at control points. This is exact for affine data, but
is not a general Bernstein interpolation/projection of nonlinear data.

## Rational CAD reconstruction

```python
mesh = jx.read_gmsh("part.geo", mesh_size="auto", geometry_order=2,
                    rational_geometry=True)
print(mesh.geometry_representation)  # rational_cad
print(mesh.reconstruction_report)
```

This opt-in path reconstructs **circular and elliptical arcs** as rational
quadratic Bezier curves, and **aligned cylindrical quad faces** as rational
arc × line surfaces. Straight curves and planar faces remain planar. Degrees
above two use exact homogeneous degree elevation. The import still starts from
Gmsh topology; it replaces boundary control points and weights, checks shared
traces, and certifies positive Jacobians using the homogeneous determinant
numerator. Distorted reconstructions trigger the usual quad algorithm retries
or automatic size reduction.

This is exact conic/cylinder reconstruction to floating-point and CAD-kernel
precision, rather than polynomial boundary projection. A CAD closest-point
query can itself have an endpoint error of a few `1e-8`; `geometry_error` reports
that sampled distance and is not an exactness certificate. The tests also check
circle/ellipse/cylinder equations directly, independently of CAD projection.

**Scope:** CAD curves must be lines, circles or ellipses; CAD faces must be
planes or cylinders. Cylindrical faces need an aligned quad layout following
circular arcs and straight axial generators, such as a transfinite/extruded hex
mesh. Arbitrary trimmed NURBS surfaces, spheres, cones, tori and unaligned
triangular cylinder reconstruction are not implemented. Unsupported entities
raise an error. Merely importing an arbitrary STEP file does not guarantee a
reconstructible layout. `.msh` input has no retained CAD and cannot use this
option. Polynomial import remains available for these other geometries.

Once reconstructed, ordinary degree elevation and knot insertion preserve the
exact rational geometry. On a rational `GmshMesh`, `refine`/`elevate` do this
without reopening or projecting onto CAD. Simplex knot insertion remains
unsupported. To obtain exact rational simplices from a reconstructed Bezier
quad/hex, use `jx.split_to_simplices(patch)`; it restricts the homogeneous map to
two triangles or six tetrahedra. The total simplex degree is the sum of the
tensor degrees, and can exceed the direct Gmsh simplex import limit of four.

## Mixed-element spaces

```python
mesh = jx.read_gmsh("mixed.msh", cell_type="mixed")
space = jx.FunctionSpace(mesh, vec=3)
```

`cell_type="mixed"` retains triangle/quad or tetrahedron/hex blocks, with no
conversion of the entire mesh to one family. CAD input respects per-surface
recombination/transfinite settings instead of forcing recombination everywhere.
Pyramids and prisms are rejected; in particular, Gmsh's automatic hex-to-tet
transition can introduce pyramids, which this implementation does not handle.

Coupling is **strong C0 coupling** through homogeneous Bezier trace constraints,
eliminated into the extraction operators before assembly. It supports matching
edges/faces and, in 3D, a quadrilateral face covered by two triangular faces
along a common diagonal. The solution trace agrees everywhere along the
interface, including with rational weights. There are no penalty parameters or
interface multipliers. C1 continuity is not imposed.

A tensor face of degrees `(p, q)` restricts to a triangle of total degree
`p+q`. Adjacent tetrahedra therefore need at least that degree. The mixed Gmsh
importer elevates cells when necessary and reconstructs slave-face homogeneous
controls to match the master trace, then validates the changed geometry. This
can change the original tetrahedral parametrization. `FunctionSpace` itself
never changes input geometry: incompatible supplied traces or insufficient
trace degree raise an error.

Mixed spaces currently require single-span Bezier cells and matching facet
partitions. General hanging-node meshes, arbitrary nonmatching mortar coupling,
and mixed hierarchical refinement are not implemented. Splitting neighboring
hexes must use compatible face diagonals; the importer rejects a partial or
overlapping hex/tet face partition. The space's `degree` is the common degree
used for basis evaluation; individual cell degrees and families remain in
`space.patches` and `space.element_types` (0 = tensor, 1 = simplex).

## Uniform refinement of cell meshes

Use `refine_cells` for mixed meshes and for pure triangle, quad, tetrahedron or
hex Bezier cell meshes:

```python
# patches is the full coarse mesh, including both element families.
fine_patches = jx.refine_cells(patches, n=1)
fine_space = jx.FunctionSpace(fine_patches, vec=3)  # vec=2 for 2D elasticity
# Recreate the problem on fine_space and solve again.
```

Each level splits a quad or triangle into four cells, and a hex or tetrahedron
into eight. The element families and degrees stay unchanged. Refinement acts on
homogeneous control points `(w*x, w)` using de Casteljau subdivision, preserving
the entire rational map to floating-point precision, including exact conic and
cylindrical boundaries. No source CAD file or projection is needed. With
polynomial input, it preserves the existing CAD approximation.

The subdivisions match along shared edges/faces, including a hex face covered
by two triangles. Tetrahedron boundary faces receive the same four-triangle
midpoint subdivision regardless of vertex ordering. Rebuilding `FunctionSpace`
reconstructs the strong C0 constraints. Parent face labels pass only to child
faces on that face, so named supports and loads can be reused.
Within each tetrahedron, the central octahedron is split along its shortest
physical diagonal to limit elongation under repeated subdivision. This choice
does not affect the face partitions and is not a general mesh-quality guarantee.

`n` is the number of subdivision levels; `n=0` returns a list of the original
cells. A single patch, a patch sequence or a `GmshMesh` is accepted. The return
value is a **flat list of patches**, not a `GmshMesh`: source element tags,
physical volume groups and import-quality diagnostics are not carried forward.
Face labels are retained. Existing solutions are not automatically transferred;
rebuild the problem and solve on the refined space.

This operation requires single-span Bezier cells and refines the whole supplied
mesh. Local marked refinement, hanging-interface constraints and error-driven
adaptivity are not implemented. Do not refine only one side of an interface.
`Patch.refine()` and the tensor `GmshMesh.refine()` still perform knot insertion
inside patches; use **`jx.refine_cells()`** to obtain separate child cells for a
mixed space. Cell counts grow by `4**n` in 2D or `8**n` in 3D.

Both rational examples accept a refinement level:

```bash
python -m jaxiga.examples.gmsh_rational_plate --refine 1
python -m jaxiga.examples.gmsh_cylinder_3d --refine 1
```

Without the flag, they use their coarse demonstration meshes. With one level,
the cylinder has eight hexes and 48 tetrahedra. Matplotlib and `.vtu` output use
the refined analysis mesh; the VTK `n` sampling parameter remains independent
of analysis refinement.

The cylinder example defaults to **Jacobi-preconditioned conjugate gradients**
with a relative residual tolerance of `1e-8`, at most 5000 iterations, and
assembly chunks of 16 cells. This avoids the expensive sparse LU factorization
at refinement level 2 (448 cells, 47,187 displacement DOFs). It prints progress
and the independently checked residual, and does not export an unconverged
solution. Use `--solver scipy` to explicitly select direct sparse LU, or
`--no-plot` to export VTK without opening a Matplotlib window:

```bash
python -m jaxiga.examples.gmsh_cylinder_3d --refine 2 --no-plot
```

The corresponding library call is:

```python
solution = jx.solve(
    problem, params={"E": 1e5, "nu": 0.3}, chunk=16,
    linear=jx.LinearOptions(method="cg", preconditioner="jacobi",
                            tol=1e-8, maxiter=5000),
)
print(solution.stats["converged"], solution.stats["relative_residual"])
```

CG requires a symmetric positive-definite system, as in this elasticity example
with its rigid-body modes constrained. Assembled CPU CG uses SciPy's native CSR
matrix-vector kernel; accelerator and matrix-free CG remain in JAX. The CPU
callback is covered by the existing implicit derivative rule, including the
adjoint solve. General library CPU solves still default to direct sparse LU
unless CG is requested. Chunking bounds basis/element assembly memory, but the
assembled sparse matrix and its indexing still consume memory.

## 2D and 3D examples, Matplotlib and ParaView

Install the optional input/output dependencies:

```bash
python -m pip install -e '.[gmsh,post]' matplotlib
python -m jaxiga.examples.gmsh_rational_plate
python -m jaxiga.examples.gmsh_cylinder_3d
```

- [`gmsh_rational_plate.py`](../src/jaxiga/examples/gmsh_rational_plate.py):
  exact circular hole/slot boundaries, quads on the left and rational triangles
  on the right, plane-stress tension. Writes `gmsh_rational_plate.vtu`.
- [`gmsh_cylinder_3d.py`](../src/jaxiga/examples/gmsh_cylinder_3d.py):
  a thick-walled quarter cylinder with radii 1 and 2 and length 2, loaded by
  internal pressure 10. Symmetry conditions and zero axial displacement at
  both ends give a plane-strain Lamé comparison. The lower region is a rational
  hex and the upper region is six rational tets, restricted from a second hex.
  Writes `gmsh_cylinder_3d.vtu`. Edit the bundled
  [`quarter_cylinder.geo`](../src/jaxiga/examples/geometries/quarter_cylinder.geo)
  to inspect the CAD and structured starting mesh.

Both solve linear elasticity, plot the mesh beside von Mises stress, and save
ParaView-readable VTK XML unstructured grids in the current directory. The
existing quad, triangle elasticity and triangle Poisson examples also save
`.vtu` files. These are sampled visualizations: `n` controls geometric sampling,
not analysis refinement. Curved patches are displayed/exported as small linear
cells, preserving holes and element-wise derivative jumps. In 3D the Matplotlib
plot shows exterior faces; VTK contains the full volume.

For any solution:

```python
path = solution.to_vtk("result.vtu", n=5, fields=("von_mises",))
```

The file includes displacement components, a three-component `displacement`
vector (zero third component in 2D), the requested fields, and `element_id`.
Mixed output retains triangle/quad or tet/hex VTK cell types. Open it directly
in ParaView; use **Surface With Edges**, color by **von_mises**, or apply
**Warp By Vector** with **displacement**. Scalar Poisson output is named
`solution`. A `.vtu` file uses the VTK XML format, rather than legacy `.vtk`.
