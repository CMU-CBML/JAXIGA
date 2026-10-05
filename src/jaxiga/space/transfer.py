"""Transfer of fields between two hierarchical spaces on the same patches.

An adaptive time-stepping scheme refines its mesh while it runs, and everything
it carries forward -- the damage field, the history field, the last displacement
-- has to move onto the new mesh. Two facts make that exact and cheap here:

* Refinement only ever *subdivides*, so every element of the new mesh sits
  inside exactly one element of the old one. That parent is found from the
  element keys, by shifting the multi-index, rather than by any geometric
  search.
* The spline spaces are nested, so a coarse field has an exact representation
  on the refined mesh. An :math:`L^2` projection therefore reproduces it to
  round-off -- it is a change of basis, not an approximation.

The same parent map moves quadrature-point data, which is not a spline field
and cannot be projected: the history field of a phase-field model is a running
maximum, and projecting it would overshoot near the crack and inject damage.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from jaxiga.space.evaluation import evaluate
from jaxiga.space.pointset import PointSet
from jaxiga.space.bernstein import element_spans


@dataclasses.dataclass(frozen=True)
class ParentMap:
    """Which element of the coarse mesh each fine element came from.

    Attributes
    ----------
    parent : (n_fine,) int array
        Coarse element index containing each fine element.
    scale, shift : (n_fine, dim) arrays
        The affine map from the fine element's reference cube to the coarse
        one's: ``ref_coarse = scale * ref_fine + shift``. Both reduce to
        ``(1, 0)`` for elements that were not refined.
    """

    parent: np.ndarray
    scale: np.ndarray
    shift: np.ndarray

    @property
    def n_fine(self) -> int:
        return len(self.parent)

    def to_coarse(self, ref):
        """Map reference points of the fine elements into the coarse ones.

        ``ref`` is ``(n_q, dim)`` (shared across elements) or
        ``(n_fine, n_q, dim)``; the result is always per-element.
        """
        ref = np.asarray(ref)
        if ref.ndim == 2:
            ref = np.broadcast_to(ref, (self.n_fine,) + ref.shape)
        return ref * self.scale[:, None, :] + self.shift[:, None, :]


def _level0_dims(space, patches):
    """Elements per direction on each patch of the unrefined mesh."""
    dims = []
    for patch in patches:
        dims.append(
            tuple(
                len(element_spans(kv, patch.degree[d])[0])
                for d, kv in enumerate(patch.knot_arrays())
            )
        )
    return dims


def parent_map(coarse, fine) -> ParentMap:
    """Locate every element of ``fine`` inside ``coarse``.

    Both spaces must be built on the same patches, and ``fine`` must be a
    refinement of ``coarse`` -- otherwise some fine element has no ancestor in
    the coarse mesh and there is nothing to transfer from.

    The lookup walks up the refinement tree by integer arithmetic. A level-``l``
    element with multi-index ``i`` has children ``2*i + offset``, so the
    ancestor of a level-``L`` element at level ``l`` is ``i >> (L - l)``; the
    first ancestor that is an *active* element of the coarse mesh is the parent.
    """
    if coarse.patches is None or fine.patches is None:
        raise ValueError("both spaces must carry their patches to be related")
    if coarse.elem_key is None or fine.elem_key is None:
        raise ValueError(
            "field transfer needs hierarchical element keys; build both spaces "
            "through refine_elements (an empty mark list is enough)"
        )

    patches = list(coarse.patches)
    dim = coarse.dim
    dims0 = _level0_dims(coarse, patches)
    keys_c = np.asarray(coarse.elem_key)
    keys_f = np.asarray(fine.elem_key)

    # (patch, level, local) -> coarse element index
    index = {(int(p), int(l), int(m)): e for e, (p, l, m) in enumerate(keys_c)}

    def dims(p, l):
        return tuple(n * 2**l for n in dims0[p])

    parent = np.empty(len(keys_f), dtype=int)
    for e, (p, level, local) in enumerate(keys_f):
        p, level, local = int(p), int(level), int(local)
        idx = np.array(np.unravel_index(local, dims(p, level), order="F"))
        for l in range(level, -1, -1):
            flat = int(np.ravel_multi_index(tuple(idx >> (level - l)), dims(p, l), order="F"))
            hit = index.get((p, l, flat))
            if hit is not None:
                parent[e] = hit
                break
        else:  # pragma: no cover - only reachable for unrelated meshes
            raise ValueError(
                f"element {e} of the fine space has no ancestor in the coarse "
                "space; the two meshes are not related by refinement"
            )

    # Affine reference map, read off the parametric boxes of the two elements.
    box_f = np.asarray(fine.elem_vertex)
    box_c = np.asarray(coarse.elem_vertex)[parent]
    lo_f, hi_f = box_f[:, :dim], box_f[:, dim:]
    lo_c, hi_c = box_c[:, :dim], box_c[:, dim:]
    # The key lookup succeeds for any pair of meshes that happen to share index
    # ranges, so confirm geometrically that each fine element really is inside
    # the element it was matched to.
    tol = 1e-9 * np.maximum(hi_c - lo_c, 1.0)
    if not (np.all(lo_f >= lo_c - tol) and np.all(hi_f <= hi_c + tol)):
        raise ValueError(
            "the fine space is not a refinement of the coarse one: some element "
            "is not contained in the coarse element with its index"
        )

    scale = (hi_f - lo_f) / (hi_c - lo_c)
    shift = (lo_f + hi_f - lo_c - hi_c) / (hi_c - lo_c)
    return ParentMap(parent=parent, scale=scale, shift=shift)


def coarse_basis_at(coarse, pmap: ParentMap, ps: PointSet):
    """The coarse basis evaluated at the points ``ps`` names on the fine mesh.

    The returned :class:`~jaxiga.space.evaluation.BasisData` has one "element"
    per *fine* element, so its values line up index-for-index with a basis
    evaluated on the fine space at the same point set.
    """
    return evaluate(
        coarse,
        PointSet(
            elems=pmap.parent,
            ref=jnp.asarray(pmap.to_coarse(ps.ref)),
            weights=None,
            tag="grid",
        ),
    )


def project(values, fine, basis_fine, *, solve=None):
    """:math:`L^2`-project point values onto ``fine``, returning dof coefficients.

    ``values`` is ``(n_elems, n_q)`` (scalar) or ``(n_elems, n_q, vec)``, given
    at the quadrature points of ``basis_fine``.

    A field in a nested coarse space satisfies this discrete mass equation
    exactly when its samples use the same quadrature as the mass matrix and
    that matrix is nonsingular. This reproduction property does not require
    exact polynomial integration: general NURBS and curved geometry produce
    rational integrands. Floating-point and linear-solve errors remain.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    values = np.asarray(values)
    scalar = values.ndim == 2
    if scalar:
        values = values[..., None]

    R = np.asarray(basis_fine.R)  # (n_e, n_q, n_local)
    w = np.asarray(basis_fine.w)  # (n_e, n_q)
    dofs = np.asarray(basis_fine.dofs)  # (n_e, n_local)
    n_scalar = fine.n_scalar_basis

    # Element mass matrices and load vectors, then one scatter each. The
    # padding dof of a hierarchical element carries identically zero basis
    # values, so it contributes nothing and is dropped after assembly.
    me = np.einsum("eqi,eq,eqj->eij", R, w, R)
    fe = np.einsum("eqi,eq,eqc->eic", R, w, values)

    rows = np.broadcast_to(dofs[:, :, None], me.shape).reshape(-1)
    cols = np.broadcast_to(dofs[:, None, :], me.shape).reshape(-1)
    M = sp.coo_matrix(
        (me.reshape(-1), (rows, cols)), shape=(n_scalar + 1, n_scalar + 1)
    ).tocsr()[:n_scalar, :n_scalar]

    f = np.zeros((n_scalar + 1, values.shape[-1]))
    np.add.at(f, dofs.reshape(-1), fe.reshape(-1, values.shape[-1]))
    f = f[:n_scalar]

    solve = solve or (lambda A, b: spla.spsolve(A.tocsc(), b))
    coeffs = np.stack([solve(M, f[:, c]) for c in range(f.shape[1])], axis=1)

    if scalar:
        return jnp.asarray(coeffs[:, 0])
    # interleave components the way a vector space numbers its dofs
    return jnp.asarray(coeffs.reshape(-1))


def carry_points(values, pmap: ParentMap, ref_fine, ref_coarse):
    """Move quadrature-point data to the fine mesh by nearest parent point.

    ``values`` is ``(n_coarse_elems, n_q)``. Each fine quadrature point takes
    the value of the closest quadrature point *within its parent element*,
    measured in the parent's reference cube.

    Nearest-neighbour rather than projection: the fields carried this way are
    running maxima and other non-smooth quantities, and a projection of the
    steep history spike at a crack overshoots, seeding damage that the physics
    never produced.
    """
    values = np.asarray(values)
    target = pmap.to_coarse(ref_fine)  # (n_fine, n_q, dim)
    src = np.asarray(ref_coarse)
    if src.ndim == 3:  # per-element coarse rule: take the parent's own points
        src = src[pmap.parent]
        d2 = ((target[:, :, None, :] - src[:, None, :, :]) ** 2).sum(-1)
    else:
        d2 = ((target[:, :, None, :] - src[None, None, :, :]) ** 2).sum(-1)
    nearest = np.argmin(d2, axis=-1)  # (n_fine, n_q)
    return jnp.asarray(values[pmap.parent[:, None], nearest])
