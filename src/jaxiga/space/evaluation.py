"""The single basis evaluator.

Replaces ``processing_splines.py``, ``processing_splines_1d.py``,
``processing_splines_3d.py``, ``processing_splines_DEM.py``,
``processing_splines_old.py`` and the loop-based evaluation inside
``utils_iga/assembly.py`` with one dimension-generic implementation.

Split of work
-------------
Bernstein bases at the reference points are static (point locations are
explicitly non-differentiable), so they are computed once in NumPy. Everything
downstream -- extraction, the rational NURBS correction, the push-forward to
physical derivatives -- is traced JAX, differentiable in control points,
weights and extraction operators.

Layout
------
:class:`BasisData` keeps the element axis, ``(n_elems, n_q, ...)``, rather
than the flat ``(n_pts, ...)`` of the architecture note. Assembly is
element-wise -- it needs the grouping to form element matrices -- and a flat
view is one reshape away, so keeping the structure avoids a scatter/gather
round trip on the hot path.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.space.bernstein import bernstein_tensor, voigt_pairs
from jaxiga.space.function_space import padded_control_data


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BasisData:
    """Evaluated basis, gradients and geometry at a point set.

    Attributes
    ----------
    R : (n_e, n_q, n_local)
        Rational (NURBS) basis values.
    dR : (n_e, n_q, dim, n_local)
        First derivatives with respect to physical coordinates.
    d2R : (n_e, n_q, n_d2, n_local) or None
        Second physical derivatives in Voigt order (2D: uu, uv, vv).
    x : (n_e, n_q, dim_phys)
        Physical coordinates.
    w : (n_e, n_q) or None
        Physical integration weights, i.e. reference weights times the
        Jacobian factor. ``None`` when the point set carries no weights.
    normal : (n_e, n_q, dim_phys) or None
        Outward unit normal, boundary sets only.
    dofs : (n_e, n_local) int array
        Global scalar dof indices per element. On a hierarchical space some
        entries are the reserved padding index (see
        :mod:`jaxiga.space.function_space`); the corresponding ``R``, ``dR``
        and ``d2R`` entries are identically zero.
    """

    R: jnp.ndarray
    dR: jnp.ndarray
    x: jnp.ndarray
    dofs: np.ndarray
    d2R: jnp.ndarray | None = None
    w: jnp.ndarray | None = None
    normal: jnp.ndarray | None = None
    tag: str = dataclasses.field(metadata=dict(static=True), default="interior")

    @property
    def n_elems(self) -> int:
        return self.R.shape[0]

    @property
    def n_q(self) -> int:
        return self.R.shape[1]

    @property
    def n_local(self) -> int:
        return self.R.shape[-1]

    def flat(self):
        """Flatten the element and point axes into one point axis."""
        n = self.n_elems * self.n_q
        return (
            self.R.reshape(n, -1),
            self.dR.reshape(n, self.dR.shape[2], -1),
            self.x.reshape(n, -1),
        )


def _rational_and_pushforward(N, dN, d2N, wgts, cpts, pairs):
    """NURBS correction and push-forward at a single point.

    Ports the mathematics of ``compute_NURBS``
    (``utils/processing_splines.py:127-174``), generalized over dimension.

    Parameters
    ----------
    N : (n_local,) B-spline values
    dN : (dim, n_local) parametric first derivatives
    d2N : (n_d2, n_local) parametric second derivatives, or None
    wgts : (n_local,) NURBS weights
    cpts : (n_local, dim_phys) control points
    pairs : Voigt index pairs, or ()

    Returns
    -------
    R, dR, d2R, x, detJ
    """
    M = N * wgts
    dM = dN * wgts
    W = jnp.sum(M)
    dW = jnp.sum(dM, axis=-1)

    # rational basis and its parametric derivatives (quotient rule)
    R = M / W
    dR = dM / W - M[None, :] * dW[:, None] / W**2

    d2R = None
    if d2N is not None:
        d2M = d2N * wgts
        d2W = jnp.sum(d2M, axis=-1)
        rows = []
        for k, (a, b) in enumerate(pairs):
            rows.append(
                d2M[k] / W
                - (dM[a] * dW[b] + dM[b] * dW[a] + M * d2W[k]) / W**2
                + 2 * M * dW[a] * dW[b] / W**3
            )
        d2R = jnp.stack(rows)

    # geometry map: dxdxi[a, i] = d x_i / d xi_a
    x = R @ cpts
    dxdxi = dR @ cpts

    if dxdxi.shape[0] != dxdxi.shape[1]:
        # A surface embedded in space (a shell midsurface). There is no
        # push-forward: derivatives stay parametric, and the measure is the
        # surface Jacobian |a_1 x a_2|. Shell kernels build their own metric.
        if dxdxi.shape != (2, 3):
            raise NotImplementedError("embedded geometries are supported for surfaces in 3D only")
        detJ = jnp.linalg.norm(jnp.cross(dxdxi[0], dxdxi[1]))
        return R, dR, d2R, x, detJ, dxdxi

    detJ = jnp.linalg.det(dxdxi)

    # first derivatives in physical coordinates
    dR_phys = jnp.linalg.solve(dxdxi, dR)

    d2R_phys = None
    if d2R is not None:
        d2xdxi2 = d2R @ cpts
        # T[(a,b), (i,j)] maps physical second derivatives to parametric ones
        T = jnp.stack(
            [
                jnp.stack(
                    [
                        dxdxi[a, i] * dxdxi[b, j]
                        + (dxdxi[a, j] * dxdxi[b, i] if i != j else 0.0)
                        for (i, j) in pairs
                    ]
                )
                for (a, b) in pairs
            ]
        )
        d2R_phys = jnp.linalg.solve(T, d2R - d2xdxi2 @ dR_phys)

    return R, dR_phys, d2R_phys, x, detJ, dxdxi


def _surface_measure(dxdxi, direction, dim):
    """Outward unit normal and surface Jacobian on a parametric side.

    ``dxdxi[a, i] = dx_i/dxi_a``, so its rows are the parametric tangents.
    The co-normal ``n ~ A^{-1} e_d`` is dimension-generic; the sign is applied
    by the caller from the side's end.
    """
    tangential = [a for a in range(dim) if a != direction]

    if len(tangential) == 0:  # 1D: the boundary is a point
        surf_jac = jnp.array(1.0)
    elif len(tangential) == 1:
        surf_jac = jnp.linalg.norm(dxdxi[tangential[0]])
    else:
        surf_jac = jnp.linalg.norm(jnp.cross(dxdxi[tangential[0]], dxdxi[tangential[1]]))

    e_d = jnp.zeros(dim).at[direction].set(1.0)
    normal = jnp.linalg.solve(dxdxi, e_d)
    return normal / jnp.linalg.norm(normal), surf_jac


def tensor_bernstein_jnp(ref, degree):
    """Tensor-product Bernstein at a **traced** reference point.

    First direction fastest, matching the local basis numbering. Used where the
    evaluation point itself is differentiated -- the collocation fallback
    residual and the geometry inversion in ``Solution.probe`` -- as opposed to
    the main path, where points are static and the NumPy routines apply.
    """
    from jaxiga.space.bernstein import bernstein_basis_jnp

    dim = len(degree)
    per_dir = [bernstein_basis_jnp(ref[d], degree[d]) for d in range(dim)]
    B = per_dir[-1]
    for d in range(dim - 2, -1, -1):
        B = jnp.einsum("...,b->...b", B, per_dir[d])
    return B.reshape(-1)


def reference_bernstein(space, ref):
    if space.cell_type == "simplex":
        from jaxiga.space.simplex import bernstein_jnp
        return bernstein_jnp(ref, space.degree[0])
    return tensor_bernstein_jnp(ref, space.degree)


def clip_reference(space, ref, kind=False):
    if space.cell_type == "simplex":
        positive = jnp.maximum(ref, 0.0)
        return positive / jnp.maximum(1.0, positive.sum())
    cube = jnp.clip(ref, -1.0, 1.0)
    if space.cell_type == "mixed":
        positive = jnp.maximum((ref + 1)/2, 0.)
        simplex = 2*positive/jnp.maximum(1., positive.sum())-1
        return jnp.where(kind, simplex, cube)
    return cube


def rational_at(space, elem, ref):
    """Rational basis, physical position and geometry Jacobian at a traced point.

    Returns ``(R, x, dx_dref)`` where ``R`` has shape ``(n_local,)``, ``x`` is
    ``(dim_phys,)`` and ``dx_dref[i, a] = dx_i / dref_a`` -- the ordering
    ``jax.jacfwd`` produces, i.e. already the Newton matrix for inverting the
    geometry map.
    """
    dofs = np.asarray(space.elem_dofs)[elem]
    cpts_all, wgts_all = padded_control_data(space)
    cpts = cpts_all[dofs]
    wgts = wgts_all[dofs]
    C = space.extraction[elem]

    def basis(r):
        M = (C @ reference_bernstein(space, r)) * wgts
        return M / jnp.sum(M)

    R = basis(ref)
    dx_dref = jax.jacfwd(lambda r: basis(r) @ cpts)(ref)
    return R, R @ cpts, dx_dref


@dataclasses.dataclass(frozen=True)
class ChunkedBasis:
    """A basis evaluation deferred into chunks of elements.

    ``BasisData`` over a whole Gauss set costs
    ``O(n_elems * n_q * n_local * (1 + dim))`` floats, which for 3D at p >= 3 is
    the dominant memory cost of a solve. Iterating this object yields the basis
    a few elements at a time; the methods layer contracts each chunk and keeps
    only the scattered result, so peak memory is set by the chunk, not the mesh.

    Under differentiation each chunk is wrapped in ``jax.checkpoint`` by the
    consumer, so the basis is recomputed in the backward pass rather than
    stored -- without that, autodiff would retain exactly the arrays chunking
    exists to avoid.
    """

    space: object
    ps: object
    order: int = 1
    chunk: int = 1

    @property
    def n_elems(self) -> int:
        return self.ps.n_elems

    @property
    def slices(self):
        n, c = self.ps.n_elems, max(1, int(self.chunk))
        return [(i, min(i + c, n)) for i in range(0, n, c)]

    @property
    def dofs(self):
        """Gather indices over the whole set; static, so this is cheap."""
        return np.asarray(self.space.elem_dofs)[np.asarray(self.ps.elems)]

    @property
    def n_local(self) -> int:
        return self.space.n_local

    def __len__(self):
        return len(self.slices)

    def __iter__(self):
        from jaxiga.space.pointset import slice_elements

        for lo, hi in self.slices:
            yield evaluate(self.space, slice_elements(self.ps, lo, hi), self.order)


def evaluate(space, ps, order: int = 1, chunk: int | None = None):
    """Evaluate the basis of ``space`` at the points of ``ps``.

    Parameters
    ----------
    space : FunctionSpace
    ps : PointSet
    order : int
        Highest derivative order to compute (1 or 2). Order 2 is needed by
        collocation and by the residual error estimator.
    chunk : int, optional
        Elements per chunk. When given, nothing is evaluated here: a
        :class:`ChunkedBasis` is returned and evaluation happens inside the
        assembly loop, one chunk at a time.

    Returns
    -------
    BasisData, or ChunkedBasis when ``chunk`` is given.
    """
    if chunk is not None:
        return ChunkedBasis(space=space, ps=ps, order=order, chunk=int(chunk))
    dim = space.dim
    degree = space.degree
    pairs = voigt_pairs(dim) if order >= 2 else ()

    elems = np.asarray(ps.elems)
    n_e = len(elems)

    # ---- setup stage: Bernstein bases at the (static) reference points ----
    ref = np.asarray(ps.ref)
    basis_fn = bernstein_tensor
    if space.cell_type == "simplex":
        from jaxiga.space.simplex import bernstein_simplex
        basis_fn = bernstein_simplex
    if ps.shared_ref:
        B, dB, d2B = basis_fn(ref, degree, order=order)
    else:
        flat = ref.reshape(-1, dim)
        B, dB, d2B = basis_fn(flat, degree, order=order)
        n_q = ref.shape[1]
        B = B.reshape(n_e, n_q, -1)
        dB = dB.reshape(n_e, n_q, dim, -1)
        d2B = d2B.reshape(n_e, n_q, len(pairs), -1) if d2B is not None else None

    B = jnp.asarray(B)
    dB = jnp.asarray(dB)
    d2B = jnp.asarray(d2B) if d2B is not None else None

    # Per-element chain-rule factors from the reference cube to patch
    # parameters. These come from leaves, so they must be handled with jnp:
    # inside jit they are tracers (see the staticness note in function_space).
    boxes = jnp.asarray(space.elem_vertex)[elems]
    scale = 2.0 / (boxes[:, dim:] - boxes[:, :dim])  # (n_e, dim)

    if space.cell_type == "simplex":
        scale = jnp.ones_like(scale)

    C = space.extraction[elems]  # (n_e, n_local, n_bernstein)
    # connectivity is static metadata, so this stays a concrete NumPy array
    dofs = np.asarray(space.elem_dofs)[elems]  # (n_e, n_local)
    # Gather through the padded control data: on a hierarchical space `dofs`
    # may name the reserved padding index, whose extraction rows are zero.
    cpts_all, wgts_all = padded_control_data(space)
    cpts = cpts_all[dofs]  # (n_e, n_local, dim_phys)
    wgts = wgts_all[dofs]  # (n_e, n_local)

    # ---- traced stage ----
    # Bezier extraction: N = C @ B. Keeping B shared across elements when the
    # rule allows is what keeps this from allocating one basis copy per element.
    if ps.shared_ref:
        N = jnp.einsum("eij,qj->eqi", C, B)
        dN = jnp.einsum("eij,qdj->eqdi", C, dB)
        d2N = jnp.einsum("eij,qkj->eqki", C, d2B) if d2B is not None else None
    else:
        N = jnp.einsum("eij,eqj->eqi", C, B)
        dN = jnp.einsum("eij,eqdj->eqdi", C, dB)
        d2N = jnp.einsum("eij,eqkj->eqki", C, d2B) if d2B is not None else None

    dN = dN * scale[:, None, :, None]
    if d2N is not None:
        pair_scale = jnp.stack([scale[:, a] * scale[:, b] for (a, b) in pairs], axis=1)
        d2N = d2N * pair_scale[:, None, :, None]

    per_point = jax.vmap(  # over quadrature points
        jax.vmap(  # over elements
            _rational_and_pushforward, in_axes=(0, 0, 0 if d2N is not None else None, 0, 0, None)
        ),
        in_axes=(1, 1, 1 if d2N is not None else None, None, None, None),
        out_axes=1,
    )
    R, dR, d2R, x, detJ, dxdxi = per_point(N, dN, d2N, wgts, cpts, pairs)

    # ---- integration weights and boundary normals ----
    w = None
    normal = None
    if ps.sides is not None and dxdxi.shape[-1] != dim:
        raise NotImplementedError(
            "boundary integrals on embedded surfaces are not implemented; "
            "apply shell loads as surface sources or point loads"
        )
    if ps.sides is not None and space.cell_type == "mixed":
        kinds = np.asarray(space.element_types)[elems]
        normals_ref = []
        for kind, side in zip(kinds, np.asarray(ps.sides)):
            if kind:
                nr = np.ones(dim) if side == 0 else -np.eye(dim)[side-1]
            else:
                nr = np.eye(dim)[side//2] * (2*(side%2)-1)
            normals_ref.append(nr)
        nr = jnp.asarray(normals_ref)
        mapped = jnp.linalg.solve(dxdxi, jnp.broadcast_to(nr[:,None,:,None],
                                                        (*dxdxi.shape[:2],dim,1)))[...,0]
        lengths = jnp.linalg.norm(mapped,axis=-1)
        normal = mapped/lengths[...,None]
        if ps.weights is not None:
            w = ps.weights*jnp.abs(detJ)*lengths/2**(dim-1)
    elif ps.sides is not None and space.cell_type == "simplex":
        # Unnormalized reference conormals also include the slanted face's
        # sqrt(dim) measure. Duffy face weights have unit-simplex measure.
        reference_normals = np.vstack([np.ones(dim), -np.eye(dim)])
        conormal = jnp.asarray(reference_normals[np.asarray(ps.sides)])
        mapped = jnp.linalg.solve(dxdxi, jnp.broadcast_to(conormal[:, None, :, None],
                                                        (*dxdxi.shape[:2], dim, 1)))[..., 0]
        lengths = jnp.linalg.norm(mapped, axis=-1)
        normal = mapped / lengths[..., None]
        if ps.weights is not None:
            w = ps.weights * jnp.abs(detJ) * lengths
    elif ps.sides is not None:
        sides = np.asarray(ps.sides)
        directions = sides // 2
        ends = sides % 2
        signs = jnp.asarray(np.where(ends == 1, 1.0, -1.0))

        normals, surf_jacs = [], []
        for d in range(dim):
            n_d, s_d = jax.vmap(jax.vmap(_surface_measure, in_axes=(0, None, None)),
                                in_axes=(0, None, None))(dxdxi, d, dim)
            normals.append(n_d)
            surf_jacs.append(s_d)
        pick = jnp.asarray(directions)
        normal = jnp.stack(normals)[pick, jnp.arange(n_e)] * signs[:, None, None]
        surf_jac = jnp.stack(surf_jacs)[pick, jnp.arange(n_e)]

        if ps.weights is not None:
            # tangential half-widths only: the pinned direction has no measure
            tang_scale = jnp.stack(
                [
                    jnp.prod(
                        jnp.stack([2.0 / scale[e, a] / 2.0 for a in range(dim) if a != d_e])
                    )
                    if dim > 1
                    else jnp.array(1.0)
                    for e, d_e in enumerate(directions)
                ]
            )
            w = ps.weights * surf_jac * tang_scale[:, None]
    elif ps.weights is not None:
        jac_ref_par = jnp.prod(2.0 / scale, axis=1) / 2.0**dim  # element volume factor
        w = ps.weights * jnp.abs(detJ) * jac_ref_par[:, None]

    return BasisData(
        R=R, dR=dR, d2R=d2R, x=x, w=w, normal=normal, dofs=dofs, tag=ps.tag
    )
