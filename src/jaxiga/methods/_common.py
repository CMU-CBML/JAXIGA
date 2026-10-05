"""Machinery shared by the solution methods.

The centrepiece is :func:`total_energy`, the discrete potential

    Pi(d) = sum_q w_q psi_q(d) - source work - Neumann work

Every method is a different way of finding its stationary point: energy
minimization minimizes it directly, Galerkin solves ``grad Pi = 0``, and
collocation enforces the strong form the same energy implies. That shared
definition is what makes the methods provably consistent.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.space import pointset as P
from jaxiga.space.evaluation import ChunkedBasis, evaluate
from jaxiga.space.function_space import expand_dofs, make_dofmap, pad_coeffs
from jaxiga.solvers.linear import CSRPattern


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PointField:
    """Per-quadrature-point data to be passed into a kernel through ``params``.

    Kernels are pointwise -- they see ``(grad_u, u, x, params)`` and nothing
    about which point they are at -- so field-valued coefficients cannot simply
    be arrays. Wrapping one in ``PointField`` tells the driver to slice it
    alongside the quadrature points, so the kernel receives the scalar (or
    small array) belonging to its own point.

    Used for the phase-field history and damage fields, and for any spatially
    varying material data.

    ``values`` has shape ``(n_elems, n_q, ...)`` matching the interior
    quadrature set.
    """

    values: jnp.ndarray

    @property
    def shape(self):
        return self.values.shape


def _is_point_field(x):
    return isinstance(x, PointField)


def split_point_fields(params):
    """Replace PointFields by their arrays, and build matching vmap axes.

    Returns ``(values, elem_axes, point_axes)``: the params pytree with fields
    unwrapped, plus ``in_axes`` specs mapping over elements and then over
    quadrature points.
    """
    from jax.tree_util import tree_map

    if params is None:
        return None, None, None

    values = tree_map(lambda p: p.values if _is_point_field(p) else p,
                      params, is_leaf=_is_point_field)
    elem_axes = tree_map(lambda p: 0 if _is_point_field(p) else None,
                         params, is_leaf=_is_point_field)
    point_axes = tree_map(lambda p: 0 if _is_point_field(p) else None,
                          params, is_leaf=_is_point_field)
    return values, elem_axes, point_axes


def has_point_fields(params):
    from jax.tree_util import tree_leaves

    if params is None:
        return False
    return any(_is_point_field(x) for x in tree_leaves(params, is_leaf=_is_point_field))


@dataclasses.dataclass
class MethodContext:
    """Everything the drivers need, built once per solve."""

    problem: object
    dofmap: object
    interior: object  # BasisData, or ChunkedBasis when chunking
    neumann: tuple  # ((Neumann, BasisData), ...)
    _assembly: CSRPattern | None = dataclasses.field(default=None, init=False, repr=False)

    @property
    def space(self):
        return self.problem.space

    @property
    def chunked(self) -> bool:
        return isinstance(self.interior, ChunkedBasis)

    @property
    def assembly(self) -> CSRPattern:
        """Lazily obtain the cached CSR pattern for assembled methods."""
        if self._assembly is None:
            self._assembly = _build_csr_pattern(self.space, self.dofmap)
        return self._assembly

    def interior_chunks(self):
        """Yield ``(basis, dofs)`` for the interior set, chunked or not.

        A single-element sequence in the eager case, so consumers need only one
        code path. ``dofs`` is handed out separately because it is static
        connectivity and must stay concrete even when the chunk body is traced.
        """
        if self.chunked:
            for basis in self.interior:
                yield basis, np.asarray(basis.dofs)
        else:
            yield self.interior, np.asarray(self.interior.dofs)


def chunk_body(ctx, ps_chunk, order=None):
    """Wrap a per-chunk computation so its basis is recomputed, not stored.

    The returned decorator produces a function of the space's differentiable
    leaves plus whatever else the caller passes; ``jax.checkpoint`` then makes
    the backward pass re-evaluate the basis instead of keeping it alive, which
    is the whole point of chunking.
    """
    space = ctx.space
    if order is None:
        order = 2 if ctx.problem.needs_hessian else 1

    def decorate(body):
        @jax.checkpoint
        def wrapped(cpts, wgts, extraction, elem_vertex, *args):  # noqa: D401
            local = dataclasses.replace(
                space,
                cpts=cpts,
                wgts=wgts,
                extraction=extraction,
                elem_vertex=elem_vertex,
            )
            return body(evaluate(local, ps_chunk, order), *args)

        def call(*args):
            return wrapped(
                space.cpts,
                space.wgts,
                space.extraction,
                jnp.asarray(space.elem_vertex),
                *args,
            )

        return call

    return decorate


def build_context(problem, params, quadrature=None, chunk=None) -> MethodContext:
    """Resolve boundary conditions and evaluate the bases a method needs."""
    space = problem.space
    n_q = quadrature if quadrature is not None else problem.quadrature

    prescribed, values = problem.resolve_dirichlet(params)
    dofmap = make_dofmap(space.n_dofs, prescribed, values)

    order = 2 if problem.needs_hessian else 1
    interior = evaluate(space, P.gauss(space, n_q), order=order, chunk=chunk)

    neumann = []
    for bc, bset in problem.resolve_neumann():
        neumann.append((bc, evaluate(space, P.boundary_gauss(space, bset, n_q), order=1)))

    return MethodContext(problem=problem, dofmap=dofmap, interior=interior, neumann=tuple(neumann))


def _build_csr_pattern(space, dofmap) -> CSRPattern:
    """Build the unique free-free sparsity and element scatter map once.

    The flattened order of ``scatter`` exactly matches
    ``K_e.reshape(-1)``: element, local row, local column.  CSR entries are
    lexicographically sorted by construction, so SciPy can consume them
    directly without a COO-to-CSR conversion or duplicate reduction.
    """
    return _cached_csr_pattern(
        space.elem_dofs,
        space.n_dofs,
        space.vec,
        tuple(int(d) for d in np.asarray(dofmap.free)),
    )


# Entries per chunk when building the sparsity: 4e6 int64 keys is 32 MB, small
# enough that the transient never dominates and large enough that the Python
# loop is short.
_PATTERN_CHUNK = 4_000_000


@functools.lru_cache(maxsize=32)
def _cached_csr_pattern(elem_dofs, n_dofs, vec, free_dofs) -> CSRPattern:
    """Cache structural assembly data across repeated solve contexts."""
    n = len(free_dofs)
    full_to_free = np.full(n_dofs + vec, -1, dtype=np.int64)
    full_to_free[np.asarray(free_dofs, dtype=int)] = np.arange(n, dtype=np.int64)

    gdofs = np.asarray(expand_dofs(np.asarray(elem_dofs), vec))
    free = full_to_free[gdofs]
    n_elems, n_local = free.shape

    # One key per (element, local row, local column). Materialising all of them
    # at once is what used to dominate the memory of a 3D remesh: 114 local dofs
    # over 16000 elements is 2.1e8 entries, and the masked gathers and the sort
    # inside np.unique each want their own copy of that -- several gigabytes to
    # produce a few million distinct values. Chunking over elements bounds the
    # transient without changing the result.
    step = max(1, _PATTERN_CHUNK // max(n_local * n_local, 1))

    def block_keys(lo):
        block = free[lo : lo + step]
        keys = block[:, :, None] * n + block[:, None, :]
        return keys, (block[:, :, None] >= 0) & (block[:, None, :] >= 0)

    starts = range(0, n_elems, step)
    distinct = [np.unique(k[v]) for k, v in (block_keys(lo) for lo in starts)]
    unique = (
        np.unique(np.concatenate(distinct)) if distinct else np.zeros(0, dtype=np.int64)
    )

    # int32 halves the retained scatter-map footprint and is the native index
    # width of JAX sparse operations.  A sentinel at ``nnz`` receives entries
    # that do not belong to the free-free matrix.
    if len(unique) >= np.iinfo(np.int32).max:
        raise OverflowError("CSR pattern has too many nonzeros for int32 indexing")
    scatter = np.full((n_elems, n_local, n_local), len(unique), dtype=np.int32)
    for lo in starts:
        keys, valid = block_keys(lo)
        # every valid key is in `unique`, so searchsorted is the same map
        # np.unique(..., return_inverse=True) would have handed back.
        scatter[lo : lo + step][valid] = np.searchsorted(
            unique, keys[valid]
        ).astype(np.int32, copy=False)

    if n:
        row_indices = (unique // n).astype(np.int32, copy=False)
        column_indices = (unique % n).astype(np.int32, copy=False)
        counts = np.bincount(row_indices, minlength=n)
    else:
        row_indices = np.zeros(0, dtype=np.int32)
        column_indices = np.zeros(0, dtype=np.int32)
        counts = np.zeros(0, dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32, copy=False)

    return CSRPattern(
        indptr=indptr,
        column_indices=column_indices,
        row_indices=row_indices,
        scatter=scatter,
        shape=(n, n),
        symmetric=True,
    )


def gather_local(u_full, dofs, vec):
    """Element-local coefficients, shape ``(n_e, n_local, vec)``."""
    idx = expand_dofs(np.asarray(dofs) if isinstance(dofs, np.ndarray) else dofs, vec)
    return u_full[idx].reshape(*jnp.shape(dofs), vec)


def _field_at_points(basis, u_local):
    """Solution and its gradient at every quadrature point of every element."""
    u_q = jnp.einsum("eqn,enc->eqc", basis.R, u_local)
    grad_q = jnp.einsum("eqdn,enc->eqcd", basis.dR, u_local)
    return u_q, grad_q


def element_energy(u_local, R, dR, x, w, problem, params, point_axes=None, d2R=None):
    """Potential energy of one element (volume terms only).

    ``u_local`` is ``(n_local, vec)``. ``point_axes`` is the vmap spec for
    ``params`` over quadrature points; ``None`` means no per-point fields.

    ``d2R`` is supplied for second-gradient problems (``needs_hessian``), and
    the kernel then also receives ``hess_u`` of shape ``(vec, n_d2)`` in the
    same Voigt order the evaluator uses. Nothing else changes: the Galerkin
    tangent is still the autodiff Hessian of this energy.
    """
    u_q = jnp.einsum("qn,nc->qc", R, u_local)
    grad_q = jnp.einsum("qdn,nc->qcd", dR, u_local)

    axes = None if point_axes is None else point_axes
    if d2R is None:
        psi = jax.vmap(problem.energy, in_axes=(0, 0, 0, axes))(grad_q, u_q, x, params)
    else:
        hess_q = jnp.einsum("qkn,nc->qck", d2R, u_local)
        psi = jax.vmap(
            lambda g, uu, xx, pp, hh: problem.energy(g, uu, xx, pp, hess_u=hh),
            in_axes=(0, 0, 0, axes, 0),
        )(grad_q, u_q, x, params, hess_q)

    work = 0.0
    if problem.source is not None:
        f = jax.vmap(problem.source, in_axes=(0, axes))(x, params)
        work = jnp.sum(w * jnp.sum(f.reshape(u_q.shape) * u_q, axis=-1))

    return jnp.sum(w * psi) - work


def neumann_work(u_local, basis, bc, params):
    """Work done by a traction/flux on one boundary element set."""
    u_q = jnp.einsum("eqn,enc->eqc", basis.R, u_local)
    t = jax.vmap(jax.vmap(bc.value, in_axes=(0, 0, None)), in_axes=(0, 0, None))(
        basis.x, basis.normal, params
    )
    return jnp.sum(basis.w * jnp.sum(t.reshape(u_q.shape) * u_q, axis=-1))


def total_energy(u_free, params, ctx: MethodContext):
    """Total potential energy as a function of the free dofs."""
    return energy_of_dofs(ctx.dofmap.lift(u_free), params, ctx)


def energy_of_dofs(u, params, ctx: MethodContext):
    """Total potential energy as a function of the full dof vector.

    Differentiating this gives the residual on *every* dof. At the solution the
    free-dof entries vanish and the prescribed-dof entries are the reaction
    forces, which is the cheapest exact way to get them.
    """
    problem = ctx.problem
    vec = problem.space.vec

    # Padded so that a hierarchical space's reserved dof index lands on an
    # appended zero rather than out of bounds; its basis values are zero, so
    # the coefficient it picks up cannot influence anything.
    u_pad = pad_coeffs(u, vec)

    values, elem_axes, point_axes = split_point_fields(params)

    def block(basis, u_local, vals):
        d2 = basis.d2R if problem.needs_hessian else None
        per_elem = jax.vmap(
            element_energy,
            in_axes=(0, 0, 0, 0, 0, None, elem_axes, None, None if d2 is None else 0),
        )(u_local, basis.R, basis.dR, basis.x, basis.w, problem, vals, point_axes, d2)
        return jnp.sum(per_elem)

    energy = 0.0
    if ctx.chunked:
        for (lo, hi), ps_chunk in _chunk_pointsets(ctx.interior):
            dofs = np.asarray(ctx.space.elem_dofs)[np.asarray(ps_chunk.elems)]
            u_local = u_pad[expand_dofs_traced(dofs, vec)].reshape(
                len(dofs), ctx.space.n_local, vec
            )
            vals = _slice_point_fields(params, values, lo, hi)
            energy = energy + chunk_body(ctx, ps_chunk)(
                lambda basis, ul, vv: block(basis, ul, vv)
            )(u_local, vals)
    else:
        interior = ctx.interior
        u_local = u_pad[expand_dofs_traced(interior.dofs, vec)].reshape(
            interior.n_elems, interior.n_local, vec
        )
        energy = block(interior, u_local, values)

    for bc, basis in ctx.neumann:
        bu = u_pad[expand_dofs_traced(basis.dofs, vec)].reshape(
            basis.n_elems, basis.n_local, vec
        )
        energy = energy - neumann_work(bu, basis, bc, params)

    return energy


def _chunk_pointsets(chunked):
    """``((lo, hi), PointSet)`` for each chunk of a :class:`ChunkedBasis`."""
    from jaxiga.space.pointset import slice_elements

    for lo, hi in chunked.slices:
        yield (lo, hi), slice_elements(chunked.ps, lo, hi)


def _slice_point_fields(params, values, lo, hi):
    """Restrict any PointField arrays in ``values`` to one chunk of elements."""
    from jax.tree_util import tree_map

    if params is None:
        return None
    return tree_map(
        lambda p, v: v[lo:hi] if _is_point_field(p) else v,
        params,
        values,
        is_leaf=_is_point_field,
    )


def expand_dofs_traced(scalar_dofs, vec: int):
    """Interleaved vector dofs, for either concrete or traced index arrays.

    Connectivity is normally static, so the NumPy path is taken and the result
    can be used for host-side indexing; the traced branch exists only for
    callers that genuinely hold a traced index array.
    """
    if vec == 1:
        return scalar_dofs
    if isinstance(scalar_dofs, np.ndarray):
        return expand_dofs(scalar_dofs, vec)
    return (vec * scalar_dofs[..., :, None] + jnp.arange(vec)).reshape(
        *scalar_dofs.shape[:-1], -1
    )


def residual(u_free, params, ctx: MethodContext):
    """Discrete residual: the gradient of the total potential energy."""
    return jax.grad(total_energy, argnums=0)(u_free, params, ctx)


def reactions(sol, ctx: MethodContext | None = None, params=None):
    """Reaction forces at the prescribed dofs of a solved problem.

    Returns a full-length vector that is zero on free dofs and holds the
    reaction on each prescribed dof.
    """
    if ctx is None:
        ctx = build_context(sol.problem, params if params is not None else sol.params)
    p = params if params is not None else sol.params
    r = jax.grad(energy_of_dofs, argnums=0)(sol.u, p, ctx)
    return jnp.zeros_like(r).at[ctx.dofmap.prescribed].set(r[ctx.dofmap.prescribed])
