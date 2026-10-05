"""Polynomial geometry transforms and CAD-aware setup operations.

Gmsh itself is passed in by the reader; importing this module opens no session.
"""

from functools import lru_cache
from itertools import product

import numpy as np
from scipy.interpolate import BSpline

from jaxiga.space.bernstein import bernstein_basis, bernstein_tensor


def tensor_grid(values):
    """Cartesian points, first parameter fastest."""
    return np.stack(np.meshgrid(*values, indexing="ij"), axis=-1).reshape(
        -1, len(values), order="F")


def tensor_solve(matrices, values):
    """Solve a tensor collocation system without forming a large Kronecker matrix."""
    shape = tuple(len(a) for a in matrices)
    result = np.asarray(values).reshape((*shape, -1), order="F")
    for d, matrix in enumerate(matrices):
        moved = np.moveaxis(result, d, 0)
        solved = np.linalg.solve(matrix, moved.reshape(len(matrix), -1))
        result = np.moveaxis(solved.reshape(moved.shape), 0, d)
    return result.reshape(len(values), -1, order="F")


def lagrange_to_bernstein(gmsh, kind):
    """Map Gmsh nodal coordinates to the equivalent full tensor Bernstein net.

    Evaluating Gmsh's own Lagrange basis also handles serendipity QUAD8/HEX20,
    whose polynomial spaces are contained in the full tensor-product space.
    """
    name, dim, degree, count, ref, primary = gmsh.model.mesh.getElementProperties(kind)
    family = "Quadrilateral" if dim == 2 else "Hexahedron"
    if dim not in (2, 3) or not name.startswith(family) or primary != 2 ** dim:
        raise ValueError(f"expected quadrilateral/hexahedral cells; found {name}")
    if not 1 <= degree <= 8:
        raise ValueError(f"geometry degree {degree} is unsupported (supported: 1 through 8)")
    grid = tensor_grid([np.linspace(-1, 1, degree + 1)] * dim)
    coordinates = np.zeros((len(grid), 3))
    coordinates[:, :dim] = grid
    _, basis, _ = gmsh.model.mesh.getBasisFunctions(kind, coordinates.ravel(), "Lagrange")
    matrix = bernstein_basis(np.linspace(-1, 1, degree + 1), degree)
    transform = tensor_solve([matrix] * dim, np.asarray(basis).reshape(len(grid), count))
    reference = np.asarray(ref).reshape(count, dim)
    corner_order = np.lexsort(tuple(reference[:primary, d] for d in range(dim)))
    return degree, transform, corner_order, reference


@lru_cache(maxsize=32)
def _bezier_derivatives(dim, degree):
    points = tensor_grid([np.linspace(-1, 1, max(5, 2 * degree + 3))] * dim)
    return bernstein_tensor(points, (degree,) * dim)[1]


def bezier_jacobians(points, degree):
    dim = points.shape[-1]
    derivative = _bezier_derivatives(dim, degree)
    jac = np.einsum("enx,qrn->eqxr", points, derivative, optimize=True)
    det = np.linalg.det(jac)
    scale = np.linalg.norm(jac, axis=2).prod(axis=-1)
    scaled = np.divide(det, scale, out=np.zeros_like(det), where=scale > 0)
    return det, scaled


def spline_basis(patch, parameters, derivative=None):
    factors = []
    for d, (knots, degree, count) in enumerate(zip(patch.knots, patch.degree, patch.n_cp_per_dir)):
        spline = BSpline(knots, np.eye(count), degree, extrapolate=False)
        factors.append(spline(parameters[:, d], nu=int(derivative == d)))
    result = factors[-1]
    for factor in factors[-2::-1]:
        result = np.einsum("qi,qj->qij", result.reshape(len(parameters), -1), factor)
    return result.reshape(len(parameters), -1)


def sample_parameters(patch, count=None):
    values = []
    for knots, degree in zip(patch.knots, patch.degree):
        knots = np.unique(knots)
        n = count or max(5, 2 * degree + 3)
        values.append(np.unique(np.concatenate([
            np.linspace(a, b, n) for a, b in zip(knots[:-1], knots[1:])
        ])))
    return values


def _split_coefficients(coefficients, axis):
    work = np.moveaxis(coefficients, axis, 0).copy()
    left, right = [work[0].copy()], [work[-1].copy()]
    while len(work) > 1:
        work = (work[:-1] + work[1:]) / 2
        left.append(work[0].copy())
        right.append(work[-1].copy())
    return (np.moveaxis(np.asarray(left), 0, axis),
            np.moveaxis(np.asarray(right[::-1]), 0, axis))


def _positive_bound(coefficients, depth=0):
    """Conservative Bernstein convex-hull bound, tightened by subdivision."""
    lo = float(coefficients.min())
    tolerance = 1e-11 * float(np.max(np.abs(coefficients)))
    if lo > tolerance:
        return lo
    corners = coefficients[np.ix_(*[[0, size - 1] for size in coefficients.shape])]
    if corners.min() <= tolerance or depth == 12:
        raise ValueError("cannot certify a positive geometry Jacobian")
    children = _split_coefficients(coefficients, depth % coefficients.ndim)
    return min(_positive_bound(child, depth + 1) for child in children)


def spline_quality(patch):
    """Sample scaled Jacobians and bound determinants on each knot span.

    Used after CAD refitting, when Gmsh no longer owns the spline geometry.
    Rational patches use bounds on the homogeneous determinant numerator.
    """
    if not np.allclose(patch.weights, 1, rtol=0, atol=1e-14):
        from jaxiga.geometry.gmsh_rational import rational_quality
        return rational_quality(patch)
    points = tensor_grid(sample_parameters(patch))
    control = np.asarray(patch.ctrl_pts)
    jac = np.stack([spline_basis(patch, points, d) @ control for d in range(patch.dim)], axis=-1)
    determinants = np.linalg.det(jac)
    scale = np.linalg.norm(jac, axis=1).prod(axis=-1)
    quality = np.divide(determinants, scale, out=np.zeros_like(scale), where=scale > 0).min()
    if not np.isfinite(quality) or determinants.min() <= 0:
        raise ValueError("inverted or degenerate geometry after CAD refitting")
    bounds = []
    spans = [list(zip(np.unique(k)[:-1], np.unique(k)[1:])) for k in patch.knots]
    det_degrees = [patch.dim * p - 1 for p in patch.degree]
    matrices = [bernstein_basis(np.linspace(-1, 1, p + 1), p) for p in det_degrees]
    for box in product(*spans):
        xi = tensor_grid([np.linspace(a, b, p + 1) for (a, b), p in zip(box, det_degrees)])
        jac = np.stack([spline_basis(patch, xi, d) @ control for d in range(patch.dim)], axis=-1)
        coefficients = tensor_solve(matrices, np.linalg.det(jac)[:, None])
        coefficients = coefficients.reshape(tuple(p + 1 for p in det_degrees), order="F")
        bounds.append(_positive_bound(coefficients) / 2 ** patch.dim)
    return float(quality), min(bounds)
