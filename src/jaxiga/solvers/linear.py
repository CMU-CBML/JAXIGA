"""Sparse and dense linear solves with implicit differentiation.

The backward pass solves the adjoint system once. Solver iterations are never
unrolled or differentiated through: the cotangent of a solve ``A u = b`` is
obtained from ``A^T lam = u_bar``, giving

    dL/db     = lam
    dL/dA_ij  = -lam_i u_j

so the number of adjoint solves is independent of the iteration count.
Gradient accuracy depends on convergence of both primal and adjoint systems.
The assembled custom VJP supports first-order reverse mode, not forward mode
or a general higher-order differentiation contract.

Galerkin matrices use a hashable :class:`CSRPattern` as the second static
argument.  Its sparsity and element scatter map are built once; only unique
CSR values cross the traced solve or host-callback boundary.  The legacy COO
entry point remains available for collocation and external callers.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from jax.experimental import sparse as jsparse

# Counts forward and adjoint solves so tests can assert that a VJP costs
# exactly one adjoint solve.
SOLVE_COUNTER = {"forward": 0, "adjoint": 0}


@dataclasses.dataclass(frozen=True, eq=False)
class CSRPattern:
    """Static CSR structure shared by assembly and all linear solves.

    ``scatter`` maps flattened element-matrix entries to the unique CSR value
    slot.  Its last (sentinel) slot collects prescribed/padded entries and is
    discarded after assembly.  The remaining arrays are concrete setup data;
    only the CSR values enter traced solver callbacks.

    Identity hashing is intentional.  JAX treats a pattern as a static
    ``custom_vjp`` argument, and two independently built patterns need not be
    compared element by element merely because their arrays happen to agree.
    """

    indptr: np.ndarray
    column_indices: np.ndarray
    row_indices: np.ndarray
    scatter: np.ndarray
    shape: tuple[int, int]
    symmetric: bool = False

    __hash__ = object.__hash__

    @property
    def nnz(self) -> int:
        return len(self.column_indices)

    @property
    def coo_indices(self) -> np.ndarray:
        """Unique COO view, in CSR order, for backwards compatibility."""
        return np.stack([self.row_indices, self.column_indices], axis=1)


def reset_solve_counter():
    SOLVE_COUNTER["forward"] = 0
    SOLVE_COUNTER["adjoint"] = 0


@dataclasses.dataclass(frozen=True)
class LinearOptions:
    """Linear-solver configuration.

    ``method`` is one of ``"cg"``, ``"bicgstab"``, ``"gmres"``, ``"dense"`` or
    ``"scipy"``. ``"scipy"`` routes a sparse LU through ``pure_callback``: it
    is the most robust choice for ill-conditioned systems and still
    differentiates, since the adjoint solve goes through the same path.
    Assembled CSR ``"cg"`` uses SciPy on CPUs and JAX on accelerators;
    matrix-free solves remain in JAX. CPU CG retains implicit differentiation.
    CPU CG checks the true residual and allows up to three correction solves,
    sharing the total ``maxiter`` budget, to counter recursive-residual drift.
    """

    method: str = "scipy"
    # Matrix-free applies the tangent as the JVP of the residual instead of
    # assembling it. Requires an iterative ``method``.
    matrix_free: bool = False
    preconditioner: str = "jacobi"
    tol: float = 1e-10
    maxiter: int | None = None
    # ``auto`` recognises Galerkin's symmetric CSR pattern and asks SuperLU
    # for a minimum-degree ordering of A^T+A in symmetric mode.  An explicit
    # value is useful for benchmarking or for non-Galerkin COO callers.
    superlu_ordering: str = "auto"

    def __post_init__(self):
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be finite and positive")
        if self.maxiter is not None and (
            isinstance(self.maxiter, (bool, np.bool_))
            or not isinstance(self.maxiter, (int, np.integer)) or self.maxiter <= 0
        ):
            raise ValueError("maxiter must be a positive integer or None")
        allowed = {"cg", "bicgstab", "gmres", "dense", "scipy"}
        if self.method not in allowed:
            raise ValueError(f"unknown linear method {self.method!r}; expected one of {allowed}")
        if self.preconditioner not in {"none", "jacobi"}:
            raise ValueError(
                f"unknown preconditioner {self.preconditioner!r}; expected 'none' or 'jacobi'"
            )
        orderings = {"auto", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}
        if self.superlu_ordering not in orderings:
            raise ValueError(
                f"unknown SuperLU ordering {self.superlu_ordering!r}; "
                f"expected one of {orderings}"
            )
        if self.matrix_free and self.method in {"dense", "scipy"}:
            raise ValueError(
                f"matrix_free=True needs an iterative method, not {self.method!r}; "
                f"there is no matrix for a direct solver to factor"
            )


def _dense_from_triplets(indices, data, n):
    return jnp.zeros((n, n)).at[indices[:, 0], indices[:, 1]].add(data)


def _scipy_spsolve(opts, indices, data, b):
    """CPU sparse LU through a host callback."""
    n = b.shape[0]

    def host(idx, dat, rhs):
        A = scipy.sparse.coo_matrix(
            (np.asarray(dat), (np.asarray(idx)[:, 0], np.asarray(idx)[:, 1])), shape=(n, n)
        ).tocsr()
        ordering = "COLAMD" if opts.superlu_ordering == "auto" else opts.superlu_ordering
        return scipy.sparse.linalg.spsolve(
            A, np.asarray(rhs), permc_spec=ordering
        ).astype(np.float64)

    return jax.pure_callback(
        host, jax.ShapeDtypeStruct((n,), jnp.float64), indices, data, b, vmap_method="sequential"
    )


def _iterative(opts, indices, data, b):
    n = b.shape[0]
    A = jsparse.BCOO((data, indices), shape=(n, n))

    matvec = lambda v: A @ v  # noqa: E731

    precond = None
    if opts.preconditioner == "jacobi":
        diag = jnp.zeros(n).at[indices[:, 0]].add(jnp.where(indices[:, 0] == indices[:, 1], data, 0.0))
        inv_diag = jnp.where(jnp.abs(diag) > 0, 1.0 / jnp.where(diag == 0, 1.0, diag), 1.0)
        precond = lambda v: inv_diag * v  # noqa: E731

    solver = {
        "cg": jax.scipy.sparse.linalg.cg,
        "bicgstab": jax.scipy.sparse.linalg.bicgstab,
        "gmres": jax.scipy.sparse.linalg.gmres,
    }[opts.method]

    u, _ = solver(matvec, b, tol=opts.tol, maxiter=opts.maxiter, M=precond)
    return u


def _solve_impl(opts, indices, data, b):
    if opts.method == "dense":
        return jnp.linalg.solve(_dense_from_triplets(indices, data, b.shape[0]), b)
    if opts.method == "scipy":
        return _scipy_spsolve(opts, indices, data, b)
    return _iterative(opts, indices, data, b)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _coo_sparse_solve(opts: LinearOptions, indices, data, b):
    """Solve ``A u = b`` where ``A`` is given by COO ``indices``/``data``.

    Duplicate entries are summed, so assembly can emit raw element triplets.
    """
    SOLVE_COUNTER["forward"] += 1
    return _solve_impl(opts, indices, data, b)


def _sparse_solve_fwd(opts, indices, data, b):
    u = _solve_impl(opts, indices, data, b)
    SOLVE_COUNTER["forward"] += 1
    return u, (indices, data, u)


def _sparse_solve_bwd(opts, res, u_bar):
    indices, data, u = res
    SOLVE_COUNTER["adjoint"] += 1

    # A^T lam = u_bar: transposing a COO matrix is a swap of the index columns.
    lam = _solve_impl(opts, indices[:, ::-1], data, u_bar)

    # entry k sits at (i, j) = indices[k], so dL/d data[k] = -lam_i u_j
    data_bar = -lam[indices[:, 0]] * u[indices[:, 1]]
    return (None, data_bar, lam)


_coo_sparse_solve.defvjp(_sparse_solve_fwd, _sparse_solve_bwd)


def _dense_from_csr(pattern, data):
    return (
        jnp.zeros(pattern.shape, dtype=data.dtype)
        .at[pattern.row_indices, pattern.column_indices]
        .set(data)
    )


def _scipy_csr_solve(opts, pattern, data, b):
    """CPU sparse LU with the already unique, sorted CSR structure closed over."""
    n = pattern.shape[0]
    indptr = pattern.indptr
    columns = pattern.column_indices

    def host(dat, rhs):
        A = scipy.sparse.csr_matrix(
            (np.asarray(dat), columns, indptr), shape=pattern.shape
        )
        if opts.superlu_ordering == "auto":
            ordering = "MMD_AT_PLUS_A" if pattern.symmetric else "COLAMD"
        else:
            ordering = opts.superlu_ordering
        symmetric_mode = pattern.symmetric and ordering == "MMD_AT_PLUS_A"
        if symmetric_mode:
            # SuperLU remains a general LU factorisation, but this mode keeps
            # the symmetric ordering and reduces row permutations.  Retaining
            # its default diagonal-pivot threshold is important for robustness
            # on symmetric indefinite Newton tangents.
            factor = scipy.sparse.linalg.splu(
                A.tocsc(),
                permc_spec="MMD_AT_PLUS_A",
                options={"SymmetricMode": True},
            )
            return factor.solve(np.asarray(rhs)).astype(np.float64)

        return scipy.sparse.linalg.spsolve(
            A, np.asarray(rhs), permc_spec=ordering
        ).astype(np.float64)

    return jax.pure_callback(
        host,
        jax.ShapeDtypeStruct((n,), jnp.float64),
        data,
        b,
        vmap_method="sequential",
    )


def _scipy_csr_cg(opts, pattern, data, b):
    """CPU CG using native CSR matvecs, inside the existing implicit VJP.

    A gather/scatter over every nonzero at every iteration is much more costly
    on the CPU than SciPy's CSR matrix-vector kernel. The callback returns the
    iterate even when the iteration limit is reached; the independent residual
    diagnostics in the methods layer then report non-convergence.
    """
    def host(dat, rhs):
        A = scipy.sparse.csr_matrix(
            (np.asarray(dat), pattern.column_indices, pattern.indptr),
            shape=pattern.shape,
        )
        precond = None
        if opts.preconditioner == "jacobi":
            diagonal = A.diagonal()
            inverse = np.ones_like(diagonal)
            np.divide(1.0, diagonal, out=inverse, where=diagonal != 0)
            precond = scipy.sparse.linalg.LinearOperator(
                A.shape, matvec=lambda v: inverse * v, dtype=A.dtype
            )
        rhs = np.asarray(rhs)
        u = np.zeros_like(rhs)
        residual = rhs.copy()
        norm = np.linalg.norm(residual)
        # Leave headroom for the independently evaluated JAX residual. On
        # ill-conditioned systems CG's recursive residual can drift from b-Au.
        target = 0.5 * opts.tol * norm
        remaining = opts.maxiter if opts.maxiter is not None else 10 * len(rhs)
        # Up to three correction solves, all sharing the requested budget.
        # Stop if refinement stagnates; never hide a failed true residual.
        for attempt in range(4):
            if norm <= target or remaining <= 0:
                break
            used = 0

            def count_iteration(_):
                nonlocal used
                used += 1

            correction, info = scipy.sparse.linalg.cg(
                A, residual, M=precond, rtol=min(0.5, 0.5 * target / norm),
                atol=0.0, maxiter=remaining, callback=count_iteration,
            )
            remaining -= used
            candidate = u + correction
            actual = rhs - A @ candidate
            actual_norm = np.linalg.norm(actual)
            # Preserve the first CG iterate even if truncated before its
            # Euclidean residual decreases (CG does not minimize that norm).
            if attempt > 0 and actual_norm >= norm:
                break
            u, residual, norm = candidate, actual, actual_norm
            if info != 0 or used == 0 or not np.isfinite(norm):
                break
        return np.asarray(u, dtype=np.asarray(rhs).dtype)

    return jax.pure_callback(
        host, jax.ShapeDtypeStruct(b.shape, b.dtype), data, b,
        vmap_method="sequential",
    )


def _iterative_csr(opts, pattern, data, b):
    if opts.method == "cg" and jax.default_backend() == "cpu":
        return _scipy_csr_cg(opts, pattern, data, b)
    rows = pattern.row_indices
    columns = pattern.column_indices
    n = pattern.shape[0]

    def matvec(v):
        return jnp.zeros(n, dtype=data.dtype).at[rows].add(data * v[columns])

    precond = None
    if opts.preconditioner == "jacobi":
        diagonal = (
            jnp.zeros(n, dtype=data.dtype)
            .at[rows]
            .add(jnp.where(rows == columns, data, 0.0))
        )
        safe = jnp.where(jnp.abs(diagonal) > 0, diagonal, 1.0)
        inv_diagonal = jnp.where(jnp.abs(diagonal) > 0, 1.0 / safe, 1.0)
        precond = lambda v: inv_diagonal * v  # noqa: E731

    solver = {
        "cg": jax.scipy.sparse.linalg.cg,
        "bicgstab": jax.scipy.sparse.linalg.bicgstab,
        "gmres": jax.scipy.sparse.linalg.gmres,
    }[opts.method]
    u, _ = solver(matvec, b, tol=opts.tol, maxiter=opts.maxiter, M=precond)
    return u


def _csr_solve_impl(opts, pattern, data, b):
    if opts.method == "dense":
        return jnp.linalg.solve(_dense_from_csr(pattern, data), b)
    if opts.method == "scipy":
        return _scipy_csr_solve(opts, pattern, data, b)
    return _iterative_csr(opts, pattern, data, b)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _csr_sparse_solve(opts: LinearOptions, pattern: CSRPattern, data, b):
    SOLVE_COUNTER["forward"] += 1
    return _csr_solve_impl(opts, pattern, data, b)


def _csr_sparse_solve_fwd(opts, pattern, data, b):
    u = _csr_solve_impl(opts, pattern, data, b)
    SOLVE_COUNTER["forward"] += 1
    return u, (data, u)


def _csr_sparse_solve_bwd(opts, pattern, res, u_bar):
    data, u = res
    SOLVE_COUNTER["adjoint"] += 1
    if not pattern.symmetric:
        raise NotImplementedError(
            "the static CSR adjoint currently requires a symmetric pattern"
        )

    # Galerkin energy Hessians are symmetric, so the adjoint reuses exactly
    # the same CSR structure and values; no transposed sparse matrix is built.
    lam = _csr_solve_impl(opts, pattern, data, u_bar)
    data_bar = -lam[pattern.row_indices] * u[pattern.column_indices]
    return data_bar, lam


_csr_sparse_solve.defvjp(_csr_sparse_solve_fwd, _csr_sparse_solve_bwd)


def sparse_solve(opts: LinearOptions, indices, data, b):
    """Solve from a static CSR pattern or backwards-compatible COO indices."""
    if isinstance(indices, CSRPattern):
        return _csr_sparse_solve(opts, indices, data, b)
    return _coo_sparse_solve(opts, indices, data, b)


def _krylov(opts, matvec, b, precond):
    solver = {
        "cg": jax.scipy.sparse.linalg.cg,
        "bicgstab": jax.scipy.sparse.linalg.bicgstab,
        "gmres": jax.scipy.sparse.linalg.gmres,
    }.get(opts.method)
    if solver is None:
        raise ValueError(
            f"matrix_free needs an iterative method, not {opts.method!r}; "
            f"use 'cg', 'bicgstab' or 'gmres'"
        )
    u, _ = solver(matvec, b, tol=opts.tol, maxiter=opts.maxiter, M=precond)
    return u


def matrix_free_solve(opts: LinearOptions, matvec, b, *, diagonal=None):
    """Solve ``A u = b`` with ``A`` given only by its action.

    ``matvec`` is the JVP of the residual, so no matrix is ever formed and
    memory is ``O(n_dofs)`` rather than ``O(nnz)``.

    Differentiation goes through :func:`jax.lax.custom_linear_solve`, which
    supplies the transposed operator for the adjoint and therefore gives the
    same one-adjoint-solve cost as the assembled path -- including with respect
    to everything ``matvec`` closes over, which is what makes shape and
    material derivatives work here too.
    """
    precond = None
    if diagonal is not None:
        safe = jnp.where(jnp.abs(diagonal) > 0, diagonal, 1.0)
        inv = jnp.where(jnp.abs(diagonal) > 0, 1.0 / safe, 1.0)
        precond = lambda v: inv * v  # noqa: E731

    def solve_fn(mv, rhs):
        SOLVE_COUNTER["forward"] += 1
        return _krylov(opts, mv, rhs, precond)

    def transpose_fn(mv, rhs):
        SOLVE_COUNTER["adjoint"] += 1
        return _krylov(opts, mv, rhs, precond)

    return jax.lax.custom_linear_solve(
        matvec, b, solve_fn, transpose_solve=transpose_fn, symmetric=False
    )


def solve_linear(opts: LinearOptions, indices, data, b):
    """Public entry point; see :func:`sparse_solve`."""
    return sparse_solve(opts, indices, data, b)
