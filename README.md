# JAXIGA

Differentiable, GPU-capable isogeometric analysis (IGA) on JAX.

One NURBS discretisation, one pointwise physics definition, three solution
methods — Galerkin, energy minimisation and collocation — and `jax.grad`
straight through the solve. Adaptive local refinement with THB-splines, C¹
fourth-order problems, matrix-free assembly, batched solves and differentiable
time integration.

```python
import jax.numpy as jnp
import jaxiga as jx

patch = jx.primitives.quadrilateral(
    [[0, 0], [1, 0], [1, 1], [0, 1]]
).elevate(3).refine(5)

V = jx.FunctionSpace(patch)

problem = jx.Poisson(
    V,
    dirichlet=[jx.DirichletBC(0.0, where=lambda x: jnp.full(x.shape[0], True))],
    source=lambda x, p: 8 * jnp.pi**2 * jnp.sin(2*jnp.pi*x[0]) * jnp.sin(2*jnp.pi*x[1]),
)

sol = jx.solve(problem, params={"a0": 1.0})
print(jx.errornorm(sol, lambda x: jnp.sin(2*jnp.pi*x[0]) * jnp.sin(2*jnp.pi*x[1]), "L2"))
sol.to_vtk("poisson.vtu")
```

## Installation

```bash
pip install -e .              # core
pip install -e ".[post,plot]" # VTK output and matplotlib
pip install -e ".[test]"      # to run the test suite
```

JAXIGA needs Python 3.10+ and depends on JAX, NumPy, SciPy and optax (which
backs the energy method's optimisers, so it is a core requirement). JAXIGA
enables 64-bit floats on import; FEM-grade conditioning requires them. For GPU,
install a matching `jax[cuda...]` wheel first.

`requirements-lock.txt` provides a pinned dependency snapshot. The independent
JAX-FEM comparison has its own reproducible environment in `paper/jaxfem-environment.yml`. Licence: MIT (see `LICENSE`).

## Learning examples

Short, self-contained scripts are included in
[`src/jaxiga/examples/`](src/jaxiga/examples/README.md) and shipped with the
library. Each runs a small representative problem and displays a Matplotlib
plot. Start with Poisson, then explore solution methods, a plate with a hole,
differentiation, inverse problems, adaptivity and plate bending.

```bash
python -m pip install matplotlib
python -m jaxiga.examples.poisson_2d
```

You can also run or copy any individual `.py` file. These examples use only the
library and its dependencies; they do not need the `paper/` directory.

For geometry defined in Gmsh `.geo`, STEP or `.msh` files, install the optional
mesher and try the plate with two holes and a rounded slot:

```bash
python -m pip install -e '.[gmsh]' matplotlib
python -m jaxiga.examples.gmsh_plate
```

The [Gmsh input guide](docs/gmsh.md) explains quad/hex conversion, named
boundaries and quality checks. Copy the accompanying `geometries/` directory
along with this example if running it outside the package.

## Concepts

A **Patch** is a NURBS geometry with named sides. A **FunctionSpace** turns one
or more conforming patches into a global basis, handling Bézier extraction,
connectivity and patch gluing. A **Problem** is a function space plus a
pointwise energy density `psi(grad_u, u, x, params)` plus boundary conditions —
JAXIGA differentiates that one definition into whatever each method needs.
`solve` runs one of three **methods** and returns a **Solution** you can
evaluate anywhere, write to VTK, or differentiate through.

If you know FEniCS/Firedrake, `FunctionSpace`, `DirichletBC`, `solve` and
`errornorm` mean what you expect; instead of UFL you write a plain JAX
function. If you know JAX-FEM, `vec`, location functions, pytree `params` and
the implicitly-differentiated solver all work the same way.

## Package structure

```
src/jaxiga/
├── config.py          # enables x64 on import
├── geometry/          # Patch, primitives, multipatch topology, geomdl/G+Smo IO
├── space/             # Bézier extraction, FunctionSpace, point sets, evaluation,
│                      #   hierarchical (THB) local refinement
├── forms/             # Problem, boundary conditions, materials, problem library
├── methods/           # galerkin, energy, collocation + the solve() entry point
├── solvers/           # linear (implicit diff), Newton, optimizers, time integration
├── post/              # Solution, VTK, error norms, residual estimator + adapt()
└── examples/          # small, standalone learning examples with plots
```

## Differentiability

`solve` supports **first-order reverse-mode sensitivities** of converged,
locally nonsingular discrete solutions with respect to material parameters,
sources, boundary values, control points and weights. Primal and adjoint
accuracy both matter. The number of adjoint solves is independent of parameter
count; total parameter-derivative work and storage can still grow.

| Operation | Current contract |
|---|---|
| `jax.grad`, `jax.jit(jax.grad(...))` | Supported for all three methods at converged regular roots |
| `jax.vmap` | Tested for linear solves; host sparse callbacks batch sequentially |
| `jax.jvp`, `jax.jacfwd` through assembled solves | Unsupported by the custom VJP |
| Higher derivatives through solves | Not supported; the detached-root correction is first order |
| Basis/energy gradients and Hessians | Supported inside the physics kernels |
| Remeshing and stateful staggered fracture | Outside the solver differentiation contract |

Newton, energy minimisation and least-squares collocation attach a derivative
at a root (`r=0`, `grad(Pi)=0`, or `J.T @ F=0`). They pay one extra forward
tangent solve for the correction and one adjoint solve. Collocation uses the
exact stationarity Hessian `J.T @ J + sum(F_i * Hessian(F_i))`, not just the
Gauss-Newton approximation. Closed-over geometry data remain differentiable.

Check `sol.stats["converged"]` and residual diagnostics before using a result.
Failed iterative roots are returned without the correction and do not carry
valid implicit sensitivities. Eager calls warn; compiled callers must inspect
the array diagnostics. Fracture steps have `converged` and `residual` fields;
`solver.run(loads, on_failure="raise")` stops at an unconverged step.

Automatic accelerator CG selection requires `is_positive_definite=True` on a
linear problem, appropriate coefficients, and constraints removing nullspaces.
Overriding a built-in energy invalidates its inherited SPD declaration. An
explicit `LinearOptions` always takes precedence. Variational methods require
`energy()`; flux/mass-only problems are supported by collocation.

## Repeated-solve performance

Construct the space and problem once, then compile the parameter-to-solution
function once outside the sweep:

```python
import jax
solve_parameters = jax.jit(lambda a: jx.solve(problem, params={"a0": a}).u)
solve_parameters(1.0).block_until_ready()  # compile and warm up
u = solve_parameters(2.0).block_until_ready()
```

`python paper/benchmark_execution.py --output tmp/benchmarks/execution.json` compares
synchronized eager and fully jitted solves, recording cold costs, raw warm
samples and environment information. It refuses to overwrite an existing
record.

## Adaptive local refinement

Tensor-product meshes cannot refine locally: one extra knot line crosses the
whole patch. Truncated hierarchical B-splines (THB) can, and they slot in
without touching the compute path — everything downstream of the
`FunctionSpace` constructor consumes only `(elem_dofs, extraction, cpts,
wgts)`, and a hierarchical basis is expressible in exactly that form.

```python
V = jx.FunctionSpace(patches)
for _ in range(n_cycles):
    sol = jx.solve(make_problem(V), params=params)
    eta = jx.residual_indicator(sol, params)
    V = jx.refine_elements(V, jx.dorfler_mark(eta, frac=0.3))

# or the driver, which does the same loop
sol, V, history = jx.adapt(make_problem, params, space=V0, n_cycles=8)
```

`adapt` takes a *factory* `FunctionSpace -> Problem`, because a `Problem`
resolves its `where` clauses against a fixed space when it is built. Marking
changes array shapes, so the loop is ordinary Python and `jax.grad` does not
flow across cycles — but every adapted space differentiates exactly as a
tensor-product one does.

## Beyond the basics

| Capability | API |
|---|---|
| Fourth-order problems (C¹) | `Problem.needs_hessian`, `jx.KirchhoffPlate`, `jx.ClampedBC` |
| Matrix-free tangent | `LinearOptions(method="cg", matrix_free=True)` |
| Chunked assembly (large 3D) | `jx.solve(..., chunk=n_elements)` |
| Batched parameter sweeps | `jax.vmap(lambda p: jx.solve(problem, params=p).u)` |
| Transient dynamics | `jx.mass_matrix`, `jx.dynamics.integrate`, `jx.newmark`, `jx.generalized_alpha` |
| Adaptive phase-field fracture (2D and 3D) | `jx.phase_field.StaggeredSolver` |
| Fields across a refinement | `jx.parent_map`, `jx.project`, `jx.carry_points` |
| Geometry import/export | `jx.io.read`, `jx.io.write` (geomdl JSON, G+Smo XML) |
| VTK output, single field or many | `sol.to_vtk`, `jx.post.vtk.write_vtu` |
| Time series for ParaView | `jx.post.vtk.write_series` (`.pvd`) |

## Examples

| Script | Shows |
|---|---|
| `poisson/Poisson1D_IGA.py`, `Poisson2D_IGA.py`, `Poisson3D_Cube.py` | Galerkin in 1D/2D/3D, convergence rates |
| `poisson/Poisson2D_Collocation.py` | collocation, multipatch flux continuity |
| `linear_elasticity/Elast2D_IGA_Spline_plate_w_hole.py` | multipatch Kirsch problem |
| `linear_elasticity/Elast2D_IGA_Spline_quarter_annulus.py` | curved NURBS, Lamé solution |
| `linear_elasticity/Elast3D_Cube.py` | 3D elasticity |
| `energy_minimization.py` | energy vs Galerkin; nonlinear neo-Hookean |
| `shape_derivative.py` | geometry gradients |
| `darcy/Darcy2D_inverse.py` | differentiable inverse problem |
| `phase_field/PF2D_tension_plate.py` | staggered phase-field fracture |
| `phase_field/adaptive_fracture.py` | crack-following adaptivity: tension, shear, 3D cube; renders a movie |
| `adaptive/Poisson2D_LShape_adaptive.py` | THB adaptivity on a corner singularity |
| `plate/Plate_Kirchhoff.py` | C¹ fourth-order plate vs Navier and Timoshenko |
| `linear_elasticity/Cooks_membrane.py` | neo-Hookean; Newton vs energy minimisation |
| `optimization/Topology_SIMP.py` | SIMP compliance minimisation |
| `optimization/Hole_shape.py` | shape optimisation through the control points |
| `optimization/Parametric_sweep.py` | batched solves and forward UQ |
| `dynamics/FWI1D_inclusion.py` | inversion through a whole time loop |

The examples use the current API; older PINN/DEM scripts and legacy utilities
are excluded from this public source tree.

## Multipatch coupling

Patches are glued by verifying candidate interfaces, not by merging coincident
control points. A candidate side pair must match in control-point count, point
positions, orientation (a flip or transpose, not an arbitrary permutation),
tangential knot vectors and weights — and its outward normals must be
antiparallel, which is what distinguishes an interface from two patches that
overlap. Corner contact (2D) and edge contact (3D) are recognised separately;
anything else coincident raises rather than being silently glued or silently
torn.

## Paper examples

The paper's studies are available as separate runnable modules in
[`paper/experiments/`](paper/experiments/). Start with the Poisson setup and
method comparison; the [paper example guide](paper/README.md) lists all 17
studies, a small first solve, output locations and cache controls.

```bash
python -m paper.experiments.method_comparison
# Or select a study through the original driver:
python paper/generate_results.py --example darcy_inverse
```

Running `python paper/generate_results.py` runs all studies. The manuscript and
generated outputs are not included in this repository. The plate comparison
requires separately generated reference results; see the [example guide](paper/README.md).

## Testing

```bash
pytest
```

The suite checks analytical solutions, convergence, geometry invariance,
SciPy B-spline references, derivative consistency, and solver agreement.

Independent end-to-end verification is also available through FEniCSx.
`paper/reference_fenicsx.py` solves the plate-with-hole benchmark with Lagrange
elements on curved triangles, sharing no code with JAXIGA — not even an
interpreter — and writes `paper/fenicsx_reference.json`. Both codes converge to
the analytical Kirsch solution. To generate reference results for an independent
comparison (not bundled in this repository):

```bash
/path/to/dolfinx/python paper/reference_fenicsx.py paper/fenicsx_reference.json
```

The computational comparison also includes JAX-FEM on the same curved P2
triangle meshes. Its exact environment, including the JAX-FEM git revision, is
created and the reference results generated with:

```bash
mamba env create -f paper/jaxfem-environment.yml
conda run -n jaxfem-benchmark \
  python paper/reference_jaxfem.py paper/jaxfem_reference.json
```

`paper/generate_results.py` consumes locally generated reference results. Set
`JAXIGA_JAXFEM_PYTHON=/path/to/jaxfem-benchmark/bin/python` to regenerate it
as part of the complete paper workflow.

## Status

The new API covers Galerkin, energy minimisation and collocation in 1D/2D/3D,
plus adaptive refinement, fourth-order problems, matrix-free and chunked
assembly, batching and time integration.

Known limits:

* collocation is tensor-product only — this implementation has no THB point-selection/weighting scheme, and `method="collocation"` raises on a hierarchical space;
* multipatch coupling is C⁰, so a `KirchhoffPlate` interface behaves as a
  hinge; use a single patch unless that is what you mean;
* the residual estimator sees the PDE residual but not the error in
  interpolated Dirichlet data, which can put a floor under an adaptive run
  with non-constant boundary data (see the L-shape example);
* `Solution.probe` locates points on the host, so it cannot run inside `jit`
  or `vmap` unless `targets` is precomputed with `jx.locate_points`, and it
  does not scale to thousands of points;
* collocation forms its Gauss–Newton system densely in the number of free dofs;
  the structural sparsity is real but not yet exploited;
* mixed degrees across patches, trimmed patches and non-conforming coupling are
  not supported, and are detected and refused rather than approximated.

