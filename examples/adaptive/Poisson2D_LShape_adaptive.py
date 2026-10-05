"""Adaptive local refinement on the L-shaped domain (THB-splines).

The classical benchmark for adaptivity: on

    Omega = (-1,1)^2 \\ ([0,1] x [-1,0])

the harmonic function ``u = r^(2/3) sin(2 theta / 3)`` has an unbounded
gradient at the re-entrant corner. It lies in ``H^(1+2/3)`` and no more, so
*uniform* refinement converges at ``N^(-1/3)`` in the energy norm no matter how
high the degree -- the regularity, not the approximation power, is the
bottleneck. Grading the mesh into the corner restores the optimal ``N^(-p/2)``.

The loop is solve -> estimate -> mark -> refine:

* ``residual_indicator`` reuses the collocation strong-residual machinery, so
  no new mathematics is involved;
* ``dorfler_mark`` selects the fewest elements carrying a given share of the
  estimated error;
* ``refine_elements`` bisects them, mirrors the marking across patch
  interfaces, and closes the mesh for class-2 admissibility.

Refinement changes array shapes, so this is an ordinary Python loop --
``jax.grad`` does not flow across cycles, though every individual adapted space
differentiates exactly as a tensor-product one does.

A caveat this example is deliberately built to expose: the residual estimator
sees the PDE residual, *not* the error in the Dirichlet data, which this
package interpolates at boundary control points. On a coarse outer boundary
that interpolation error is a floor the loop cannot descend below, and at
``p = 3`` it is reached almost immediately. Starting from a mesh that already
resolves the boundary data removes it -- see the two starting meshes below.
"""

import time

import numpy as np
import jax.numpy as jnp

import jaxiga as jx

ALPHA = 2.0 / 3.0


def _theta(x, y):
    """Polar angle measured on ``[0, 3 pi / 2]`` across the L-shape."""
    t = jnp.arctan2(y, x)
    return jnp.where(t < -1e-12, t + 2 * jnp.pi, t)


def exact(x):
    r = jnp.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-300
    return r**ALPHA * jnp.sin(ALPHA * _theta(x[0], x[1]))


def exact_grad(x):
    r = jnp.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-300
    t = _theta(x[0], x[1])
    c = ALPHA * r ** (ALPHA - 1.0)
    return jnp.array([c * jnp.sin((ALPHA - 1.0) * t), c * jnp.cos((ALPHA - 1.0) * t)])


def lshape(degree, refine):
    """Three unit squares filling every quadrant but the fourth."""
    quadrants = [
        jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0),
        jx.primitives.rectangle(-1.0, 0.0, 0.0, 1.0),
        jx.primitives.rectangle(-1.0, -1.0, 0.0, 0.0),
    ]
    return [p.elevate(degree).refine(refine) for p in quadrants]


def make_problem(V):
    # u is not zero on the outer boundary, so the data is genuinely non-constant
    return jx.Poisson(
        V,
        dirichlet=[
            jx.DirichletBC(lambda x, p: exact(x), where=lambda X: np.ones(len(X), bool))
        ],
    )


def energy_norm(sol):
    return jx.errornorm(sol, exact, "H1", exact_grad=exact_grad)


def slope(hist, key="H1", window=8):
    """Least-squares rate over the last few cycles.

    Dorfler marking refines in bursts, so the error falls in steps rather than
    smoothly; a short window lands on whichever part of a step it happens to
    span. Fitting the tail over several cycles averages that out. The early
    cycles are pre-asymptotic -- the corner is not yet resolved at all -- and
    are excluded on purpose.
    """
    n = np.array([h["n_dofs"] for h in hist], dtype=float)
    e = np.array([h[key] for h in hist], dtype=float)
    k = max(0, len(n) - window)
    return float(np.polyfit(np.log(n[k:]), np.log(e[k:]), 1)[0])


def uniform_study(degree, levels):
    out = []
    for n in levels:
        V = jx.FunctionSpace(lshape(degree, n))
        sol = jx.solve(make_problem(V), params={"a0": 1.0})
        out.append({"n_dofs": V.n_dofs, "n_elems": V.n_elems, "H1": energy_norm(sol)})
    return out


PARAMS = {"a0": 1.0}

for degree, start, n_cycles in ((2, 1, 16), (3, 3, 12)):
    print(f"\n{'=' * 74}\ndegree {degree}, optimal energy-norm slope {-degree / 2:.2f}")
    print("=" * 74)

    V0 = jx.FunctionSpace(lshape(degree, start))
    t0 = time.perf_counter()
    sol, V, hist = jx.adapt(
        make_problem,
        PARAMS,
        space=V0,
        n_cycles=n_cycles,
        frac=0.5,
        exact=exact,
        exact_grad=exact_grad,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  adaptive (start: uniform refine({start}))")
    print("     dofs   elems      eta         energy-norm error")
    for h in hist:
        print(f"  {h['n_dofs']:7d} {h['n_elems']:7d}   {h['eta']:.4e}   {h['H1']:.4e}")
    s = slope(hist)
    print(
        f"  slope {s:+.3f} vs optimal {-degree / 2:+.3f} "
        f"({100 * abs(s + degree / 2) / (degree / 2):.1f}% off), "
        f"{V.hierarchy.n_levels} levels, "
        f"n_local_max {V.n_local} (tensor-product {V.n_bernstein}), {elapsed:.0f}s"
    )

    uni = uniform_study(degree, range(start, start + 4))
    print("\n  uniform")
    print("     dofs   elems   energy-norm error")
    for h in uni:
        print(f"  {h['n_dofs']:7d} {h['n_elems']:7d}   {h['H1']:.4e}")
    # The uniform rate is fitted over the last three meshes only: the first is
    # pre-asymptotic, and including it biases the estimate away from -1/3.
    print(f"  slope {slope(uni, window=3):+.3f}  "
          f"(regularity-limited at -1/3; degree cannot help)")

    best_uniform = uni[-1]
    print(
        f"\n  at the end: adaptive {hist[-1]['H1']:.3e} with {hist[-1]['n_dofs']} dofs "
        f"vs uniform {best_uniform['H1']:.3e} with {best_uniform['n_dofs']} dofs "
        f"-> {best_uniform['H1'] / hist[-1]['H1']:.1f}x less error on "
        f"{best_uniform['n_dofs'] / hist[-1]['n_dofs']:.1f}x fewer dofs"
    )

    sol.to_vtk(f"lshape_adaptive_p{degree}", n=4)
    print(f"  wrote lshape_adaptive_p{degree}.vtu")
