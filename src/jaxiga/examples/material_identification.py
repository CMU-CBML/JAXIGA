"""Identify a stiff inclusion from full-field displacements (Mechanical MNIST).

A plate in uniaxial tension contains an inclusion of unknown shape. Given the
displacement field on a 51 x 51 grid, the edge displacements are prescribed as
Dirichlet data, and a nodal inclusion field phi in (0, 1) sets the stiffness
E = exp(phi log r). The field is found by minimising the interior displacement
misfit plus a total-variation penalty with L-BFGS-B. Gradients come from one
adjoint solve per evaluation (implicit differentiation through the solve).

Usage:
    python -m jaxiga.examples.material_identification            # synthetic specimen
    python -m jaxiga.examples.material_identification SPECIMEN.npz
The second form reads a specimen of the experimental Mechanical MNIST data set
(keys DIC_disp, instron_disp and, if present, label). It uses the stiffness ratio
of the specimen's material pair.
"""

import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

import jaxiga as jx
from jaxiga.space.evaluation import evaluate
from jaxiga.space.pointset import PointSet, gauss

N, SIDE, NU, TV = 51, 40.0, 0.4, 0.1
# Inclusion-to-matrix stiffness ratios of the four material pairs (inclusion, matrix).
CONTRAST = {(0, 8): 1.78, (3, 9): 3.84, (4, 8): 10.7, (7, 9): 0.9}


def pixel_space():
    """Quadratic B-splines with a knot on every interior pixel line of the N x N grid."""
    s = np.linspace(0, 1, N)[1:-1]
    patch = jx.primitives.rectangle(0.0, 0.0, SIDE, SIDE).elevate(2).insert_knots(s, s)
    V = jx.FunctionSpace(patch, vec=2)
    # Grid node (i, j): row i from the top, column j from the left.
    ev = np.asarray(V.elem_vertex)
    cell = -np.ones((N - 1, N - 1), int)
    cell[N - 2 - np.rint(ev[:, 1] * (N - 1)).astype(int), np.rint(ev[:, 0] * (N - 1)).astype(int)] = np.arange(len(ev))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    ci, cj = np.clip(ii - 1, 0, N - 2), np.clip(jj, 0, N - 2)
    ref = np.stack([2.0 * (jj - cj) - 1.0, 2.0 * ((ci + 1) - ii) - 1.0], -1).reshape(-1, 1, 2)
    nodes = evaluate(V, PointSet(elems=cell[ci, cj].ravel(), ref=ref.astype(float), tag="grid"))
    # Bilinear interpolation of nodal fields to the Gauss points.
    qx = np.asarray(evaluate(V, gauss(V)).x)
    fx, fy = qx[..., 0] / SIDE * (N - 1), (SIDE - qx[..., 1]) / SIDE * (N - 1)
    j0, i0 = np.clip(np.floor(fx).astype(int), 0, N - 2), np.clip(np.floor(fy).astype(int), 0, N - 2)
    ty, tx = fy - i0, fx - j0

    def to_quad(f):
        return ((1 - ty) * ((1 - tx) * f[i0, j0] + tx * f[i0, j0 + 1])
                + ty * ((1 - tx) * f[i0 + 1, j0] + tx * f[i0 + 1, j0 + 1]))
    return V, nodes, to_quad


def measured_edges(x, params):
    """Dirichlet value at a boundary control point, from the measured edge displacements."""
    e, grid, tol = params["edges"], jnp.linspace(0.0, SIDE, N), 1e-9 * SIDE
    at = lambda t, v: jnp.stack([jnp.interp(t, grid, v[:, c]) for c in range(2)])
    return jnp.where(x[1] > SIDE - tol, at(x[0], e["top"]),
           jnp.where(x[1] < tol, at(x[0], e["bottom"]),
           jnp.where(x[0] < tol, at(x[1], e["left"][::-1]), at(x[1], e["right"][::-1]))))


def edges_of(u):
    return {"top": u[0], "bottom": u[-1], "left": u[:, 0], "right": u[:, -1]}


def load_specimen(path):
    """Displacement per mm of crosshead travel (least squares over the upper half of the load)."""
    with np.load(path) as z:
        disp, d = z["DIC_disp"], z["instron_disp"]
        label = z["label"] if "label" in z.files else None
    high = d >= 0.5 * d.max()
    unit = np.einsum("t,thwc->hwc", d[high], disp[high]) / (d[high] @ d[high])
    yy, xx = np.meshgrid(np.linspace(0, unit.shape[0] - 1, N), np.linspace(0, unit.shape[1] - 1, N), indexing="ij")
    unit = np.stack([map_coordinates(unit[..., c], [yy, xx], order=1) for c in range(2)], -1)
    truth = contrast = None
    if label is not None:
        label = map_coordinates(label.astype(float), [yy, xx], order=0).astype(int)
        pair = tuple(int(v) for v in np.unique(label))
        contrast = CONTRAST.get(pair)
        truth = label == pair[0]
    return unit, truth, contrast or 3.84


def synthetic_specimen(solve):
    """A ring-shaped stiff inclusion under a stretch with slightly uneven grips, plus noise."""
    y, x = np.meshgrid(np.linspace(1, -1, N), np.linspace(-1, 1, N), indexing="ij")
    r = np.hypot(x / 0.8, (y - 0.1) / 1.1)
    truth = (r > 0.35) & (r < 0.6)
    s = np.linspace(0, 1, N)
    boundary = np.zeros((N, N, 2))
    boundary[..., 1] = 0.02 + 0.97 * s[::-1, None]                     # stretch with bottom slip
    boundary[..., 0] = -0.3 * (s[None, :] - 0.5) * (1 - 0.6 * np.exp(-((s[:, None] - 0.5) / 0.35) ** 2 * 4))
    u = np.asarray(solve(jnp.asarray(truth, float), np.log(3.84), edges_of(jnp.asarray(boundary))))
    u = u + 2e-4 * np.random.default_rng(0).normal(size=u.shape)    # DIC-like noise
    return u, truth, 3.84


def main():
    V, nodes, to_quad = pixel_space()
    bcs = [jx.DirichletBC(measured_edges, where=w) for w in ("top", "bottom", "left", "right")]
    problem = jx.LinearElasticity(V, plane="stress", dirichlet=bcs)

    def solve(phi, log_contrast, edges):
        params = {"E": jx.PointField(jnp.exp(log_contrast * to_quad(phi))), "nu": NU, "edges": edges}
        sol = jx.solve(problem, params=params, linear=jx.LinearOptions(method="scipy"))
        return sol.at(basis=nodes).reshape(N, N, 2)

    if len(sys.argv) > 1:
        target, truth, contrast = load_specimen(sys.argv[1])
    else:
        target, truth, contrast = synthetic_specimen(solve)

    # Per-component weights: the non-affine part carries the geometry.
    yy, xx = np.meshgrid(np.linspace(1, 0, N), np.linspace(0, 1, N), indexing="ij")
    A = np.stack([np.ones(N * N), xx.ravel(), yy.ravel()], 1)
    scale = [np.std(target[..., c].ravel() - A @ np.linalg.lstsq(A, target[..., c].ravel(), rcond=None)[0]) for c in range(2)]
    interior = np.zeros((N, N, 1)); interior[1:-1, 1:-1] = 1
    w = jnp.asarray(interior / np.square(scale) / interior.sum())
    u_obs, edges = jnp.asarray(target), edges_of(jnp.asarray(target))

    def loss(z):
        phi = jax.nn.sigmoid(z.reshape(N, N))
        misfit = jnp.sum(w * (solve(phi, np.log(contrast), edges) - u_obs) ** 2)
        dy, dx = jnp.diff(phi, axis=0), jnp.diff(phi, axis=1)
        return misfit + TV * (jnp.sum(jnp.sqrt(dy**2 + 1e-4)) + jnp.sum(jnp.sqrt(dx**2 + 1e-4))) / N**2

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    fun = lambda z: tuple(np.asarray(v, float) for v in value_and_grad(jnp.asarray(z)))
    fit = minimize(fun, np.zeros(N * N), jac=True, method="L-BFGS-B",
                   bounds=[(-8.0, 8.0)] * N * N, options={"maxiter": 80, "maxls": 20})
    phi = 1 / (1 + np.exp(-fit.x.reshape(N, N)))
    print(f"{V.summary()}\nStiffness ratio {contrast}; objective {fun(np.zeros(N * N))[0]:.4f} -> {fit.fun:.4f} "
          f"in {fit.nit} L-BFGS-B iterations")
    if truth is not None:
        pred = phi > 0.5
        dice = np.mean([2 * np.sum((truth == c) & (pred == c)) / (np.sum(truth == c) + np.sum(pred == c)) for c in (0, 1)])
        print(f"Dice of the thresholded field against the true inclusion: {dice:.3f}")

    strain = -np.gradient(target[..., 1], axis=0) * (N - 1) / SIDE
    panels = [(strain, "measured axial strain"), (phi, r"recovered $\phi$")]
    if truth is not None:
        panels.append((truth.astype(float), "true inclusion"))
    _, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2))
    for ax, (a, title) in zip(axes, panels):
        ax.imshow(a, cmap="viridis"); ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
