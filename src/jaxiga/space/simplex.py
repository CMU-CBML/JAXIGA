"""Total-degree Bernstein polynomials on the unit triangle/tetrahedron.

Coordinates are ``(lambda_1, ..., lambda_d)`` with ``lambda_0=1-sum(x)``.
Multi-indices are ordered lexicographically, descending in each exponent.
"""

from functools import lru_cache
from math import factorial

import jax.numpy as jnp
import numpy as np

from jaxiga.space.bernstein import voigt_pairs


@lru_cache(None)
def indices(dim, degree):
    if dim == 0:
        return ((degree,),)
    return tuple((i, *tail) for i in range(degree, -1, -1)
                 for tail in indices(dim - 1, degree - i))


def lattice(dim, degree):
    if degree == 0:
        return np.full((1, dim), 1 / (dim + 1))
    return np.asarray(indices(dim, degree), dtype=float)[:, 1:] / degree


def _values(bary, degree, xp):
    out = []
    for alpha in indices(bary.shape[-1] - 1, degree):
        value = xp.ones(bary.shape[:-1]) * factorial(degree)
        for d, a in enumerate(alpha):
            if a:
                value = value * bary[..., d] ** a / factorial(a)
        out.append(value)
    return xp.stack(out, axis=-1)


def bernstein_jnp(ref, degree):
    return _values(jnp.concatenate([jnp.atleast_1d(1 - ref.sum()), ref]), degree, jnp)


def bernstein_simplex(points, degree, order=1):
    """Values and first/second derivatives, in the evaluator's standard layout."""
    points = np.asarray(points)
    dim = points.shape[-1]
    p = int(degree[0] if isinstance(degree, tuple) else degree)
    bary = np.column_stack([1 - points.sum(axis=1), points])
    alpha = np.asarray(indices(dim, p))
    B = _values(bary, p, np)

    def lower(k, shifts):
        if p < k:
            return np.zeros_like(B)
        lookup = {a: i for i, a in enumerate(indices(dim, p - k))}
        values = _values(bary, p - k, np)
        values = np.column_stack([values, np.zeros(len(points))])
        beta = alpha.copy()
        for d in shifts:
            beta[:, d] -= 1
        cols = [lookup.get(tuple(a), values.shape[1] - 1) for a in beta]
        return values[:, cols]

    dB = np.stack([p * (lower(1, [d + 1]) - lower(1, [0]))
                   for d in range(dim)], axis=1)
    d2B = None
    if order >= 2:
        d2B = np.stack([p * (p - 1) * (
            lower(2, [a + 1, b + 1]) - lower(2, [a + 1, 0])
            - lower(2, [0, b + 1]) + lower(2, [0, 0]))
            for a, b in voigt_pairs(dim)], axis=1)
    return B, dB, d2B


def gauss_rule(dim, n):
    """Duffy Gauss-Jacobi rule, including the transformation Jacobian.

    Jacobi weights integrate the Duffy factors exactly even when n=1; a
    one-point tensor Legendre rule would give the wrong tetrahedron volume.
    """
    from scipy.special import roots_jacobi

    if dim == 0:
        return np.zeros((1, 0)), np.ones(1)
    rules = [roots_jacobi(n, dim - d - 1, 0) for d in range(dim)]
    grids = np.meshgrid(*[(x + 1) / 2 for x, _ in rules], indexing="ij")
    wg = np.meshgrid(*[w / 2**(dim - d) for d, (_, w) in enumerate(rules)], indexing="ij")
    cube = np.stack([g.ravel() for g in grids], axis=1)
    weights = np.prod([v.ravel() for v in wg], axis=0)
    pts = np.empty_like(cube)
    remaining = np.ones(len(cube))
    for d in range(dim):
        pts[:, d] = remaining * cube[:, d]
        remaining *= 1 - cube[:, d]
    return pts, weights


def face_points(dim, side, points):
    """Embed a lower-dimensional simplex in the face lambda_side=0."""
    bary = np.column_stack([1 - points.sum(axis=1), points])
    full = np.zeros((len(points), dim + 1))
    full[:, np.arange(dim + 1) != side] = bary
    return full[:, 1:]


@lru_cache(None)
def sample_cells(dim, subdivisions):
    """Linear simplex connectivity on the sampling lattice (plotting only)."""
    from scipy.spatial import Delaunay
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    points = lattice(dim, subdivisions)
    cells = Delaunay(points).simplices.copy()
    determinant = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]])
    cells = cells[np.abs(determinant) > 1e-12 / subdivisions**dim]
    determinant = determinant[np.abs(determinant) > 1e-12 / subdivisions**dim]
    cells[determinant < 0] = cells[determinant < 0][:, [1, 0, *range(2, dim + 1)]]
    return cells
