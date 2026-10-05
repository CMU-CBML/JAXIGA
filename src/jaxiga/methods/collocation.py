"""Collocation: enforce the strong form at Greville abscissae.

The strong residual is derived from the same energy density the other two
methods use, so no separate physics definition is needed. Two derivations are
available and must agree:

* ``"d2R"`` (default) expands ``div(flux)`` by the chain rule using the second
  physical derivatives that :func:`~jaxiga.space.evaluation.evaluate` already
  provides at ``order=2``;
* ``"autodiff"`` builds the field as a function of the parametric coordinate
  and takes nested ``jacfwd`` through it. Slower, but it makes no assumption
  about what the basis evaluator supplies, so it is the generic fallback and
  the cross-check.

A multipatch space adds flux-continuity equations at interface points, and
Neumann conditions add their own, so the system is generally overdetermined and
is solved in the least-squares sense by Gauss-Newton on the normal equations.
The Jacobian stays sparse: each collocation point couples only the basis
functions of its own element, so rows are assembled from per-point Jacobians
exactly as element stiffness matrices are.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.methods._common import build_context, expand_dofs_traced
from jaxiga.space import pointset as P
from jaxiga.space.bernstein import voigt_pairs
from jaxiga.space.evaluation import evaluate, tensor_bernstein_jnp
from jaxiga.solvers.implicit import implicit_correction
from jaxiga.solvers.linear import LinearOptions, sparse_solve


@dataclasses.dataclass
class Block:
    """One group of collocation equations."""

    basis: object  # BasisData
    kind: str  # "pde" | "neumann" | "interface"
    scale: float = 1.0
    bc: object = None
    partner: object = None  # BasisData on the far side, interface blocks only
    comp_mask: np.ndarray = None  # (n_pts, vec) bool: which components are equations
    ps: object = None  # the PointSet, needed by the autodiff residual variant


# --------------------------------------------------------------------------
# strong residual
# --------------------------------------------------------------------------


def _voigt_to_full(d2, dim):
    """(n_d2,) Voigt second derivatives -> (dim, dim) symmetric matrix."""
    out = jnp.zeros((dim, dim))
    for k, (i, j) in enumerate(voigt_pairs(dim)):
        out = out.at[i, j].set(d2[k])
        if i != j:
            out = out.at[j, i].set(d2[k])
    return out


def strong_residual_d2R(problem, u, grad_u, d2u, x, params):
    """``-div(flux) + mass - source`` from second physical derivatives.

    ``d2u`` is ``(vec, dim, dim)``. The divergence expands by the chain rule::

        d sigma_cd / dx_d = dsigma/dgrad_u : d2u  +  dsigma/du . grad_u
                            + dsigma/dx (explicit)

    which keeps the full generality of an arbitrary energy density.
    """
    flux = lambda g, uu, xx: problem.flux(g, uu, xx, params)  # noqa: E731

    J_grad = jax.jacfwd(flux, argnums=0)(grad_u, u, x)  # (vec,dim,vec,dim)
    J_u = jax.jacfwd(flux, argnums=1)(grad_u, u, x)  # (vec,dim,vec)
    J_x = jax.jacfwd(flux, argnums=2)(grad_u, u, x)  # (vec,dim,dim)

    div = (
        jnp.einsum("cdab,abd->c", J_grad, d2u)
        + jnp.einsum("cda,ad->c", J_u, grad_u)
        + jnp.einsum("cdd->c", J_x)
    )

    res = -div + problem.mass(grad_u, u, x, params)
    if problem.source is not None:
        res = res - jnp.atleast_1d(jnp.asarray(problem.source(x, params))).reshape(res.shape)
    return res


def strong_residual_autodiff(problem, field_fn, ref, params):
    """``-div(flux) + mass - source`` by nested ``jacfwd`` in parametric space.

    ``field_fn(ref) -> (u, x, dx_dref)``; derivatives are pulled back with
    ``dref/dx = inv(dx/dref)``.
    """

    def state(r):
        u, x, dx_dref = field_fn(r)
        dref_dx = jnp.linalg.inv(dx_dref)
        du_dref = jax.jacfwd(lambda rr: field_fn(rr)[0])(r)  # (vec, dim)
        grad_u = du_dref @ dref_dx
        return u, x, grad_u, dref_dx

    def sigma_of_ref(r):
        u, x, grad_u, _ = state(r)
        return problem.flux(grad_u, u, x, params)

    u, x, grad_u, dref_dx = state(ref)
    dsigma_dref = jax.jacfwd(sigma_of_ref)(ref)  # (vec, dim, dim_ref)
    div = jnp.einsum("cda,ad->c", dsigma_dref, dref_dx)

    res = -div + problem.mass(grad_u, u, x, params)
    if problem.source is not None:
        res = res - jnp.atleast_1d(jnp.asarray(problem.source(x, params))).reshape(res.shape)
    return res


def _make_field_fn_from_extraction(space, C, cpts_local, wgts_local, u_local):
    """Field, position and geometry Jacobian as functions of the reference point."""
    degree = space.degree
    dim = space.dim

    def rational(ref):
        M = (C @ tensor_bernstein_jnp(ref, degree)) * wgts_local
        return M / jnp.sum(M)

    def field_fn(ref):
        R = rational(ref)
        dx_dref = jax.jacfwd(lambda r: rational(r) @ cpts_local)(ref)
        return R @ u_local, R @ cpts_local, dx_dref

    return field_fn


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


def build_blocks(problem, params, ctx, block_scales=None):
    """Assemble the residual blocks for a collocation solve."""
    space = problem.space
    vec = space.vec
    scales = dict(block_scales or {})

    prescribed = set(int(d) for d in np.asarray(ctx.dofmap.prescribed))

    # --- interior: one equation per free dof ---
    interior_ps = P.greville(space)
    interior = evaluate(space, interior_ps, order=2)

    point_dofs = np.asarray(interior_ps.point_dofs)
    comp_mask = np.ones((len(point_dofs), vec), dtype=bool)
    for q, sd in enumerate(point_dofs):
        for c in range(vec):
            if vec * int(sd) + c in prescribed:
                comp_mask[q, c] = False

    blocks = [Block(interior, "pde", scales.get("pde", 1.0), comp_mask=comp_mask, ps=interior_ps)]

    # --- Neumann ---
    for index, (bc, bset) in enumerate(problem.resolve_neumann()):
        label = bc.where if isinstance(bc.where, str) else f"neumann{index}"
        ps = P.boundary_greville(space, bset)
        basis = evaluate(space, ps, order=1)
        mask = np.ones((len(np.asarray(ps.point_dofs)), vec), dtype=bool)
        for q, sd in enumerate(np.asarray(ps.point_dofs)):
            for c in range(vec):
                if vec * int(sd) + c in prescribed:
                    mask[q, c] = False
        blocks.append(
            Block(basis, "neumann", scales.get(f"boundary:{label}", 1.0), bc=bc, comp_mask=mask)
        )

    # --- interfaces ---
    for iface in space.topology.interfaces:
        ps_a, ps_b = P.interface_greville(space, iface)
        basis_a = evaluate(space, ps_a, order=1)
        basis_b = evaluate(space, ps_b, order=1)
        mask = np.ones((len(np.asarray(ps_a.point_dofs)), vec), dtype=bool)
        for q, sd in enumerate(np.asarray(ps_a.point_dofs)):
            for c in range(vec):
                if vec * int(sd) + c in prescribed:
                    mask[q, c] = False
        blocks.append(
            Block(basis_a, "interface", scales.get("interface", 1.0),
                  partner=basis_b, comp_mask=mask)
        )

    return blocks


def _block_residual(problem, block, u_full, params, variant="d2R"):
    """Residual values of one block, shape ``(n_pts, vec)``."""
    space = problem.space
    vec = space.vec
    basis = block.basis

    idx = expand_dofs_traced(basis.dofs, vec)
    u_local = u_full[idx].reshape(basis.n_elems, basis.n_local, vec)

    u_q = jnp.einsum("eqn,enc->eqc", basis.R, u_local)[:, 0, :]
    grad_q = jnp.einsum("eqdn,enc->eqcd", basis.dR, u_local)[:, 0, :, :]
    x_q = basis.x[:, 0, :]

    if block.kind == "pde":
        d2_q = jnp.einsum("eqkn,enc->eqck", basis.d2R, u_local)[:, 0, :, :]
        dim = space.dim
        if variant == "d2R":
            d2_full = jax.vmap(jax.vmap(_voigt_to_full, in_axes=(0, None)), in_axes=(0, None))(
                d2_q, dim
            )
            return jax.vmap(strong_residual_d2R, in_axes=(None, 0, 0, 0, 0, None))(
                problem, u_q, grad_q, d2_full, x_q, params
            )
        return _autodiff_block_residual(problem, block, u_local, params)

    if block.kind == "neumann":
        n_q = basis.normal[:, 0, :]
        flux_q = jax.vmap(problem.flux, in_axes=(0, 0, 0, None))(grad_q, u_q, x_q, params)
        traction = jax.vmap(block.bc.value, in_axes=(0, 0, None))(x_q, n_q, params)
        return jnp.einsum("ecd,ed->ec", flux_q, n_q) - traction.reshape(u_q.shape)

    # interface: outward fluxes from both sides must cancel
    other = block.partner
    idx_b = expand_dofs_traced(other.dofs, vec)
    u_local_b = u_full[idx_b].reshape(other.n_elems, other.n_local, vec)
    u_b = jnp.einsum("eqn,enc->eqc", other.R, u_local_b)[:, 0, :]
    grad_b = jnp.einsum("eqdn,enc->eqcd", other.dR, u_local_b)[:, 0, :, :]
    x_b = other.x[:, 0, :]

    n_a = basis.normal[:, 0, :]
    n_b = other.normal[:, 0, :]
    flux_a = jax.vmap(problem.flux, in_axes=(0, 0, 0, None))(grad_q, u_q, x_q, params)
    flux_b = jax.vmap(problem.flux, in_axes=(0, 0, 0, None))(grad_b, u_b, x_b, params)
    return jnp.einsum("ecd,ed->ec", flux_a, n_a) + jnp.einsum("ecd,ed->ec", flux_b, n_b)


def _autodiff_block_residual(problem, block, u_local, params):
    """PDE residual via nested jacfwd, the generic fallback derivation.

    Derivatives are taken with respect to the *reference* coordinate, so the
    chain-rule factor to patch parameters cancels against the pull-back and
    never has to be applied explicitly.
    """
    space = problem.space
    basis = block.basis

    cpts_local = space.cpts[np.asarray(basis.dofs)]
    wgts_local = space.wgts[np.asarray(basis.dofs)]
    refs = jnp.asarray(block.ps.ref)[:, 0, :]
    elems = np.asarray(block.ps.elems)

    def one(C, ul, cl, wl, ref):
        field_fn = _make_field_fn_from_extraction(space, C, cl, wl, ul)
        return strong_residual_autodiff(problem, field_fn, ref, params)

    C_all = space.extraction[elems]
    return jax.vmap(one)(C_all, u_local, cpts_local, wgts_local, refs)


# --------------------------------------------------------------------------
# assembly and solve
# --------------------------------------------------------------------------


def _rows_and_columns(problem, blocks, ctx):
    """Equation numbering and the free-dof columns each equation touches."""
    space = problem.space
    vec = space.vec
    full_to_free = np.full(space.n_dofs, -1, dtype=int)
    full_to_free[np.asarray(ctx.dofmap.free)] = np.arange(len(ctx.dofmap.free))

    layout = []
    n_eq = 0
    for block in blocks:
        mask = block.comp_mask
        n_pts, _ = mask.shape
        rows = np.full((n_pts, vec), -1, dtype=int)
        active = np.flatnonzero(mask.reshape(-1))
        rows.reshape(-1)[active] = np.arange(n_eq, n_eq + len(active))
        n_eq += len(active)

        cols_scalar = np.asarray(block.basis.dofs)  # (n_pts, n_local)
        cols = full_to_free[
            np.asarray(expand_dofs_traced(cols_scalar, vec))
        ]
        partner_cols = None
        if block.partner is not None:
            partner_cols = full_to_free[
                np.asarray(expand_dofs_traced(np.asarray(block.partner.dofs), vec))
            ]
        layout.append((rows, cols, partner_cols))

    return layout, n_eq


def solve_collocation(
    problem,
    params=None,
    linear: LinearOptions | None = None,
    initial_guess=None,
    system: str = "least_squares",
    block_scales=None,
    variant: str = "d2R",
    tol: float = 1e-10,
    max_steps: int = 20,
    implicit: bool = True,
    **kwargs,
):
    """Solve by collocation at Greville points.

    Parameters
    ----------
    system : {"least_squares", "square"}
        ``"square"`` requires exactly as many equations as free dofs.
    block_scales : dict, optional
        Per-block weights, e.g. ``{"boundary:outer": 10.0}``.
    variant : {"d2R", "autodiff"}
        Which strong-residual derivation to use.
    """
    if problem.space.is_hierarchical:
        raise NotImplementedError(
            "collocation on a hierarchical (THB) space is not implemented; "
            "this driver constructs tensor-product Greville points. Use method='galerkin' "
            "or method='energy' on an adapted space."
        )
    from jaxiga.solvers.diagnostics import warn_if_failed

    if system not in {"least_squares", "square"}:
        raise ValueError("system must be 'least_squares' or 'square'")
    linear = linear or LinearOptions()
    ctx = build_context(problem, params)
    blocks = build_blocks(problem, params, ctx, block_scales)
    layout, n_eq = _rows_and_columns(problem, blocks, ctx)

    n_free = len(ctx.dofmap.free)
    if system == "square" and n_eq != n_free:
        raise ValueError(
            f"system='square' needs one equation per free dof, but there are "
            f"{n_eq} equations and {n_free} free dofs; use system='least_squares'"
        )

    def residual_vector(u_free):
        u = ctx.dofmap.lift(u_free)
        out = []
        for block, (rows, _, _) in zip(blocks, layout):
            r = _block_residual(problem, block, u, params, variant) * block.scale
            out.append(r.reshape(-1)[np.flatnonzero(block.comp_mask.reshape(-1))])
        return jnp.concatenate(out)

    def least_squares(u_free):
        """``1/2 |F|^2``, the objective the solver actually minimises.

        Writing the method this way rather than in terms of ``F`` alone is what
        makes the root, and hence its derivative, unambiguous: once Neumann and
        interface blocks are added the system is overdetermined, and no ``u``
        makes ``F`` vanish.
        """
        F = residual_vector(u_free)
        return 0.5 * jnp.sum(F**2)

    # The root the solver reaches is a stationary point of that objective, so
    # the residual whose implicit derivative is wanted is its gradient, and the
    # Jacobian of *that* residual is the objective's Hessian.
    normal_residual = jax.grad(least_squares)

    # The normal matrix is dense (J is dense in the free dofs), so it is passed
    # to the shared linear solver as a complete triplet pattern. That keeps the
    # user's LinearOptions meaningful, and routes the solve through the adjoint
    # rule that makes a gradient cost one solve.
    dense_pattern = np.stack(
        np.meshgrid(np.arange(n_free), np.arange(n_free), indexing="ij"), axis=-1
    ).reshape(-1, 2)

    def normal_solve(u_free, rhs):
        """Solve ``H x = rhs`` with the *exact* Jacobian of ``normal_residual``.

        Deliberately not the Gauss-Newton matrix ``J^T J`` that drives the
        iteration. The true Jacobian of ``J^T F`` is

            H = J^T J + sum_i F_i grad^2 F_i,

        and the second term vanishes only when ``F`` is affine in ``u`` (a
        linear problem) or when the residual is zero at the solution. For a
        nonlinear problem solved in the least-squares sense neither holds, and
        using ``J^T J`` here would give a gradient that is merely close.
        Gauss-Newton remains the right choice for the *iteration*, where an
        approximate Jacobian costs convergence rate and nothing else.
        """
        H = jax.hessian(least_squares)(u_free)
        return sparse_solve(linear, dense_pattern, H.reshape(-1), rhs)

    def gauss_newton_step(u_free):
        F = residual_vector(u_free)
        J = jax.jacfwd(residual_vector)(u_free)
        return sparse_solve(linear, dense_pattern, (J.T @ J).reshape(-1), -(J.T @ F))

    u_free = jnp.zeros(n_free) if initial_guess is None else initial_guess

    if problem.is_linear:
        # The residual is affine, so one Gauss-Newton step is exact.
        u_free = u_free + gauss_newton_step(u_free)
        steps = jnp.asarray(1)
    else:

        def cond(state):
            _, grad_norm, step = state
            return jnp.isfinite(grad_norm) & (grad_norm > tol) & (step < max_steps)

        def body(state):
            u, _, step = state
            u = u + gauss_newton_step(u)
            return u, jnp.linalg.norm(normal_residual(u)), step + 1

        u_free, _, steps = jax.lax.while_loop(
            cond,
            body,
            (u_free, jnp.linalg.norm(normal_residual(u_free)), jnp.asarray(0)),
        )

    if implicit:
        norm = jnp.linalg.norm(normal_residual(u_free))
        u_free = implicit_correction(
            u_free, normal_residual, normal_solve,
            converged=jnp.isfinite(norm) & (norm <= tol),
        )

    norm = jnp.linalg.norm(normal_residual(u_free))
    converged = jnp.isfinite(norm) & (norm <= tol)
    warn_if_failed("collocation stationarity", converged, norm)

    stats = {
        "method": "collocation",
        "system": system,
        "variant": variant,
        "n_equations": n_eq,
        "n_free": n_free,
        "residual": jnp.linalg.norm(residual_vector(u_free)),
        "grad_norm": norm,
        "converged": converged,
        "finite": jnp.isfinite(norm),
        "steps": steps,
    }
    return ctx.dofmap.lift(u_free), ctx, stats


# Note on the Gauss-Newton system
# -------------------------------
# Each collocation row couples only its own element's basis functions, so the
# Jacobian is structurally sparse. It is nevertheless built with ``jacfwd`` over
# the free dofs and contracted densely above, because the sparse path this
# replaced read the pattern from the *values* of ``J`` at the current iterate:
# a coefficient that happened to vanish there was dropped from the pattern for
# good, and the conversion to NumPy that found it made the whole solve
# untraceable, so neither ``jit`` nor implicit differentiation could be applied.
# Dense contraction is correct unconditionally and adequate at the sizes
# collocation targets; recovering the structural pattern from connectivity would
# restore sparsity without either defect, and is noted as future work.
