"""Galerkin method: solve ``grad Pi = 0``.

For linear problems the residual is affine in the dofs, so the tangent is
constant. Built-in PDEs evaluate their element stiffness directly; user-defined
energies fall back to Hessians of the element energy. For nonlinear problems
the same automatic tangent drives Newton.

The right-hand side is obtained as ``-restrict(r(0))``. That single expression
already carries the source term, the Neumann work, and the coupling to
inhomogeneous Dirichlet data (the ``-K_fp @ values`` term), so none of them
needs separate assembly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.methods._common import (
    MethodContext,
    _chunk_pointsets,
    _slice_point_fields,
    build_context,
    chunk_body,
    element_energy,
    expand_dofs_traced,
    residual,
    split_point_fields,
    total_energy,
)
from jaxiga.space.function_space import pad_coeffs
from jaxiga.solvers.linear import LinearOptions, matrix_free_solve, sparse_solve
from jaxiga.solvers.diagnostics import linear_diagnostics, warn_if_failed


def _defining_class(cls, name):
    return next((base for base in cls.__mro__ if name in base.__dict__), None)


def _has_matching_element_kernel(problem):
    """Use a kernel only when it implements the active energy definition.

    In particular, subclassing a built-in PDE and overriding ``energy`` must
    fall back to automatic Hessians instead of silently inheriting a kernel
    for the parent class's different energy.
    """
    cls = type(problem)
    kernel_owner = _defining_class(cls, "element_tangent")
    return kernel_owner is not None and kernel_owner is _defining_class(cls, "energy")


def _tangent_block(basis, u_local, problem, values, elem_axes, point_axes, n_local, vec):
    """Element stiffness matrices for one group of elements."""

    d2R = basis.d2R if problem.needs_hessian else None
    if _has_matching_element_kernel(problem):

        def optimized(R, dR, x, w, p, d2):
            return problem.element_tangent(R, dR, x, w, p, point_axes, d2)

        return jax.vmap(
            optimized,
            in_axes=(0, 0, 0, 0, elem_axes, None if d2R is None else 0),
        )(basis.R, basis.dR, basis.x, basis.w, values, d2R)

    def one(ul, R, dR, x, w, p, d2):
        def energy_of(flat):
            return element_energy(
                flat.reshape(n_local, vec), R, dR, x, w, problem, p, point_axes, d2
            )

        # The source term is affine in u and drops out of the Hessian; for a
        # linear problem the result is independent of the expansion point.
        return jax.hessian(energy_of)(ul)

    return jax.vmap(one, in_axes=(0, 0, 0, 0, 0, elem_axes, None if d2R is None else 0))(
        u_local, basis.R, basis.dR, basis.x, basis.w, values, d2R
    )


def element_tangents(ctx: MethodContext, params, u_free=None):
    """Element stiffness matrices and their dof maps, chunk by chunk.

    Yields ``(dofs, K_e)`` with ``K_e`` of shape
    ``(n_e, n_local*vec, n_local*vec)``. Eager evaluation yields one pair;
    chunked evaluation yields one per chunk, each recomputing its own basis
    under ``jax.checkpoint``.
    """
    problem = ctx.problem
    space = problem.space
    vec = space.vec
    n_local = space.n_local

    values, elem_axes, point_axes = split_point_fields(params)
    u = None if u_free is None else pad_coeffs(ctx.dofmap.lift(u_free), vec)

    def locals_of(dofs):
        n = len(dofs)
        if u is None:
            return jnp.zeros((n, n_local * vec))
        return u[expand_dofs_traced(dofs, vec)].reshape(n, n_local * vec)

    if ctx.chunked:
        for (lo, hi), ps_chunk in _chunk_pointsets(ctx.interior):
            dofs = np.asarray(space.elem_dofs)[np.asarray(ps_chunk.elems)]
            vals = _slice_point_fields(params, values, lo, hi)
            body = chunk_body(ctx, ps_chunk)(
                lambda basis, ul, vv: _tangent_block(
                    basis, ul, problem, vv, elem_axes, point_axes, n_local, vec
                )
            )
            yield dofs, body(locals_of(dofs), vals)
    else:
        interior = ctx.interior
        dofs = np.asarray(interior.dofs)
        yield dofs, _tangent_block(
            interior, locals_of(dofs), problem, values, elem_axes, point_axes, n_local, vec
        )


def _assemble_csr_values(pattern, element_blocks):
    """Scatter element blocks directly into the pattern's unique CSR slots."""
    # One extra slot is the sink for prescribed and hierarchical padding dofs.
    values = None
    elem = 0
    for block in element_blocks:
        n_elems = block.shape[0]
        if values is None:
            values = jnp.zeros(pattern.nnz + 1, dtype=block.dtype)
        scatter = pattern.scatter[elem : elem + n_elems].reshape(-1)
        values = values.at[scatter].add(block.reshape(-1))
        elem += n_elems

    if elem != pattern.scatter.shape[0]:
        raise ValueError(
            f"element assembly covered {elem} elements, but the CSR map has "
            f"{pattern.scatter.shape[0]}"
        )
    if values is None:
        values = jnp.zeros(pattern.nnz + 1)
    return values[: pattern.nnz]


def assemble_csr_values(ctx: MethodContext, params, u_free=None):
    """Tangent values on the context's precomputed unique CSR pattern."""
    blocks = (K_e for _, K_e in element_tangents(ctx, params, u_free))
    return _assemble_csr_values(ctx.assembly, blocks)


def assemble_triplets(ctx: MethodContext, params, u_free=None):
    """Unique COO view of the tangent, retained for API compatibility.

    Unlike the former implementation, the returned entries contain no
    duplicates: their order is the CSR order stored in ``ctx.assembly``.
    Internal solves consume the static CSR pattern and only these values.
    """
    return ctx.assembly.coo_indices, assemble_csr_values(ctx, params, u_free)


def mass_triplets(ctx: MethodContext, rho=1.0, params=None):
    """COO triplets of ``M_ij = int rho phi_i phi_j``, on the free dofs.

    Emitted on **exactly the same pattern** as :func:`assemble_triplets`, since
    both scatter the outer product of the same element dof list. Time
    integration relies on that: it combines mass and stiffness by adding their
    values, with no index bookkeeping.

    ``rho`` may be a scalar or a callable ``rho(x, params)``.
    """
    space = ctx.problem.space
    vec = space.vec
    n_local = space.n_local

    def block(basis):
        if callable(rho):
            rho_q = jax.vmap(jax.vmap(rho, in_axes=(0, None)), in_axes=(0, None))(
                basis.x, params
            )
        else:
            rho_q = jnp.asarray(rho)
        scalar = jnp.einsum("eq,eqa,eqb->eab", basis.w * rho_q, basis.R, basis.R)
        # Components are uncoupled: expand each scalar entry into a vec-by-vec
        # identity block, interleaved to match expand_dofs.
        eye = jnp.eye(vec)
        return jnp.einsum("eab,cd->eacbd", scalar, eye).reshape(
            -1, n_local * vec, n_local * vec
        )

    blocks = (block(basis) for basis, _ in ctx.interior_chunks())
    return ctx.assembly.coo_indices, _assemble_csr_values(ctx.assembly, blocks)


def mass_matrix(problem, rho=1.0, params=None, quadrature=None):
    """Convenience wrapper: ``(indices, m_data, k_data, ctx)`` for a problem.

    Returns the mass and stiffness triplets on the shared pattern, together
    with the context that defines the free-dof numbering.
    """
    ctx = build_context(problem, params, quadrature)
    indices, k_data = assemble_triplets(ctx, params)
    m_indices, m_data = mass_triplets(ctx, rho, params)
    assert np.array_equal(indices, m_indices), "mass and stiffness patterns diverged"
    return indices, m_data, k_data, ctx


def tangent_matvec(ctx: MethodContext, params, u_free=None):
    """The tangent's action on a vector, with no matrix formed.

    This is the JVP of the residual the Newton path already builds, so it needs
    no separate derivation and is exact for nonlinear problems too.
    """
    base = ctx.dofmap.zeros() if u_free is None else u_free

    def matvec(v):
        return jax.jvp(lambda uf: residual(uf, params, ctx), (base,), (v,))[1]

    return matvec


def tangent_diagonal(ctx: MethodContext, params, u_free=None):
    """The tangent's diagonal, for Jacobi preconditioning of the matrix-free path.

    The accumulated diagonal uses ``O(n_dofs)`` storage, but each batch still
    forms element tangents; use chunking to bound that temporary memory.
    Using the same
    preconditioner as the assembled path is what makes the two take the same
    number of iterations.
    """
    space = ctx.problem.space
    vec = space.vec
    accum = jnp.zeros(space.n_dofs + vec)
    for dofs, K_e in element_tangents(ctx, params, u_free):
        gdofs = np.asarray(expand_dofs_traced(dofs, vec))
        d = jnp.diagonal(K_e, axis1=1, axis2=2)  # (n_e, n_dof_e)
        accum = accum.at[gdofs.reshape(-1)].add(d.reshape(-1))
    return accum[ctx.dofmap.free]


def rhs(problem, params, ctx: MethodContext | None = None):
    """Load vector on the free dofs, including the Dirichlet lifting term."""
    if ctx is None:
        ctx = build_context(problem, params)
    zero = ctx.dofmap.zeros()
    return -residual(zero, params, ctx)


# Default free-dof threshold for selecting CG on accelerators. Small systems
# can favor host factorization because of iterative kernel-launch overhead.
# The crossover depends on device, degree and conditioning; callers can choose
# a solver explicitly with LinearOptions.
CG_CROSSOVER = 3000


def default_linear(problem, n_free: int | None = None) -> LinearOptions:
    """Choose CG only for an explicitly declared SPD linear energy.

    The declaration assumes admissible coefficients and boundary conditions
    removing nullspaces. An inherited declaration is invalidated by an energy
    override unless the subclass explicitly opts in again. The 3000-free-dof
    accelerator threshold is a heuristic; callers can
    override it with LinearOptions. CPUs and unclassified operators use LU.
    """
    big_enough = n_free is None or n_free >= CG_CROSSOVER
    cls = type(problem)
    spd = problem.is_positive_definite and (
        "is_positive_definite" in cls.__dict__
        or _defining_class(cls, "is_positive_definite") is _defining_class(cls, "energy")
    )
    if problem.is_linear and spd and jax.default_backend() != "cpu" and big_enough:
        return LinearOptions(method="cg", preconditioner="jacobi", tol=1e-12)
    return LinearOptions()


def solve_galerkin(
    problem,
    params=None,
    linear: LinearOptions | None = None,
    initial_guess=None,
    quadrature=None,
    chunk=None,
    **kwargs,
):
    """Solve a problem by the Galerkin method.

    Returns the full (lifted) dof vector.
    """
    from jaxiga.solvers.newton import newton
    from jaxiga.forms.problem import Problem

    if type(problem).energy is Problem.energy:
        raise TypeError(
            "Galerkin requires an energy density; define energy(), or use "
            "method='collocation' for a flux/mass-only problem"
        )

    # The context comes first: the default solver depends on how big the system
    # turns out to be, and building it does not depend on the solver.
    ctx = build_context(problem, params, quadrature, chunk=chunk)
    if linear is None:
        linear = default_linear(problem, len(ctx.dofmap.free))

    if problem.is_linear:
        b = rhs(problem, params, ctx)
        if linear.matrix_free:
            u_free = matrix_free_solve(
                linear,
                tangent_matvec(ctx, params),
                b,
                diagonal=(
                    tangent_diagonal(ctx, params)
                    if linear.preconditioner == "jacobi"
                    else None
                ),
            )
        else:
            data = assemble_csr_values(ctx, params)
            u_free = sparse_solve(linear, ctx.assembly, data, b)
        stats = {
            "method": "galerkin",
            "linear": linear.method,
            "matrix_free": linear.matrix_free,
            "n_free": len(ctx.dofmap.free),
            # A diagnostic must not build the full element-to-CSR scatter map
            # for a matrix-free solve. None denotes an unassembled operator.
            "nnz": None if linear.matrix_free else ctx.assembly.nnz,
        }
        if linear.matrix_free:
            matvec = tangent_matvec(ctx, params)
        else:
            pattern = ctx.assembly
            matvec = lambda v: jnp.zeros_like(b).at[pattern.row_indices].add(
                data * v[pattern.column_indices]
            )
        stats.update(linear_diagnostics(matvec, u_free, b, linear.tol))
        warn_if_failed("linear Galerkin solve", stats["converged"], stats["residual"])
    else:
        u0 = ctx.dofmap.zeros() if initial_guess is None else initial_guess
        u_free, stats = newton(
            lambda uf, p: residual(uf, p, ctx),
            u0,
            params,
            tangent_fn=lambda uf, p: (ctx.assembly, assemble_csr_values(ctx, p, uf)),
            matvec_fn=(
                (lambda uf, p: tangent_matvec(ctx, p, uf)) if linear.matrix_free else None
            ),
            diagonal_fn=(
                (lambda uf, p: tangent_diagonal(ctx, p, uf))
                if linear.matrix_free and linear.preconditioner == "jacobi"
                else None
            ),
            lin_opts=linear,
            **kwargs,
        )
        stats = {"method": "galerkin", **stats}

    return ctx.dofmap.lift(u_free), ctx, stats


def assemble_triplets_at(ctx: MethodContext, params, u_free):
    """Tangent triplets at a given state; kept as the Newton-facing spelling."""
    return assemble_triplets(ctx, params, u_free)
