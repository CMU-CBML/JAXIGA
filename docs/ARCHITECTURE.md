# JAXIGA architecture

JAXIGA separates geometry and discretization setup from differentiable JAX
computations. The public API is exported by `src/jaxiga/__init__.py`.

## Geometry and spaces

`geometry/` implements NURBS patches, exact primitives, degree elevation, knot
insertion, multipatch topology, and geometry import. It supports tensor-product,
simplex, rational and mixed cells. Gmsh is an optional dependency; geomdl JSON
files can be read without installing geomdl.

`space/` constructs `FunctionSpace` connectivity, Bézier extraction operators,
quadrature and sampling point sets, and basis evaluations. THB-spline spaces
support local refinement. Transfer operators map fields between nested spaces.
Control points and weights are differentiable leaves; connectivity and changes
in mesh topology are setup operations.

## Physics and methods

`forms/` defines `Problem`, boundary conditions, material laws, and built-in
Poisson, elasticity, Darcy, plate and shell problems. A custom problem defines
its pointwise energy using JAX operations.

`methods/` provides Galerkin, energy minimization and collocation through the
same `solve()` interface. It shares boundary-condition handling, parameter
context, assembly infrastructure and implicit differentiation helpers.
Collocation requires a tensor-product space. Fourth-order plate and shell
problems require the smoothness documented by their individual examples.

## Solvers and outputs

`solvers/` supplies sparse and dense linear solvers, iterative methods, Newton
iterations, optimizers, implicit sensitivities, dynamics, and staggered
phase-field fracture. Diagnostics report convergence and residuals. JAXIGA
uses 64-bit floating-point arithmetic by default.

`post/` provides `Solution` evaluation, interpolation and probing, error norms,
residual estimators, adaptive refinement, plots, and VTK exports. Plotting and
VTK packages are optional dependencies.

## Examples and validation

`src/jaxiga/examples/` contains small learning examples bundled with the
package. `examples/` contains larger applications using the current API.
`paper/experiments/` contains the paper studies, with shared helpers and the
`paper/generate_results.py` driver. The manuscript and saved simulation outputs
are excluded; see [the study guide](../paper/README.md) for recomputation and
external benchmark environments.

Tests check exact geometry, invariance under refinement, SciPy B-spline
references, analytical solutions, convergence, partition-of-unity identities,
solver consistency, gradients, and file-format interoperability. Optional
integration tests skip when their external dependencies are unavailable.
The retired `utils/` and `utils_iga/` implementations are not included.
