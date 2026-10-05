"""The Solution object.

Evaluation is pure JAX and therefore differentiable, so measurement misfits
built from ``sol(...)`` or ``sol.probe(...)`` work inside ``jax.grad``.
"""

from __future__ import annotations

import dataclasses
import os

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.methods._common import expand_dofs_traced
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate, rational_at, reference_bernstein, clip_reference
from jaxiga.space.function_space import pad_coeffs, padded_control_data


@dataclasses.dataclass(frozen=True)
class Solution:
    """A solved field, plus everything needed to evaluate and postprocess it."""

    space: object
    u: jnp.ndarray  # (n_dofs,) full lifted dof vector
    params: object = None
    problem: object = None
    stats: dict = dataclasses.field(default_factory=dict)

    # -- evaluation --------------------------------------------------------

    def _local(self, basis):
        vec = self.space.vec
        idx = expand_dofs_traced(basis.dofs, vec)
        return pad_coeffs(self.u, vec)[idx].reshape(basis.n_elems, basis.n_local, vec)

    def at(self, ps=None, basis=None):
        """Values at a point set; returns ``(n_e, n_q, vec)``."""
        basis = basis if basis is not None else evaluate(self.space, ps)
        return jnp.einsum("eqn,enc->eqc", basis.R, self._local(basis))

    def __call__(self, ps):
        return self.at(ps)

    def grad(self, ps=None, basis=None):
        """Gradients at a point set; returns ``(n_e, n_q, vec, dim)``."""
        basis = basis if basis is not None else evaluate(self.space, ps)
        return jnp.einsum("eqdn,enc->eqcd", basis.dR, self._local(basis))

    def field(self, name: str, ps=None, basis=None):
        """A derived field registered by the problem, e.g. ``"von_mises"``."""
        if self.problem is None:
            raise ValueError("this Solution carries no problem, so it has no derived fields")
        fields = self.problem.fields
        if name not in fields:
            raise KeyError(
                f"unknown field {name!r}; {type(self.problem).__name__} provides "
                f"{sorted(fields)}"
            )
        fn = fields[name]

        # Second-gradient problems derive their fields from the curvature, so
        # they need an order-2 evaluation and the extra kernel argument.
        needs_hess = getattr(self.problem, "needs_hessian", False)
        if basis is None:
            basis = evaluate(self.space, ps, order=2 if needs_hess else 1)
        elif needs_hess and basis.d2R is None:
            raise ValueError(
                f"{type(self.problem).__name__} needs second derivatives; pass a "
                f"basis built with evaluate(..., order=2), or pass a point set "
                f"and let this method evaluate it"
            )

        u_q = self.at(basis=basis)
        grad_q = self.grad(basis=basis)
        if not needs_hess:
            return jax.vmap(
                jax.vmap(fn, in_axes=(0, 0, 0, None)), in_axes=(0, 0, 0, None)
            )(grad_q, u_q, basis.x, self.params)

        hess_q = jnp.einsum("eqkn,enc->eqck", basis.d2R, self._local(basis))
        wrapped = lambda g, u, x, p, h: fn(g, u, x, p, hess_u=h)  # noqa: E731
        return jax.vmap(
            jax.vmap(wrapped, in_axes=(0, 0, 0, None, 0)), in_axes=(0, 0, 0, None, 0)
        )(grad_q, u_q, basis.x, self.params, hess_q)

    def sample(self, n: int = 20):
        """Uniform grid sample; returns ``(points, values)`` flattened."""
        ps = P.grid(self.space, n)
        basis = evaluate(self.space, ps)
        x = basis.x.reshape(-1, basis.x.shape[-1])
        vals = self.at(basis=basis).reshape(-1, self.space.vec)
        return x, vals

    def probe(self, x_phys, targets=None, steps: int = 12):
        """Evaluate at arbitrary physical points by inverting the geometry map.

        Locating which element contains each point is a host-side search, so it
        cannot run inside ``jit``. Pass ``targets`` from :func:`locate_points`
        to do that once up front; the remaining Newton refinement is pure JAX,
        vectorized over points, and both jittable and differentiable.

        Points outside every patch come back as NaN.
        """
        space = self.space
        vec = space.vec
        x_phys = jnp.atleast_2d(jnp.asarray(x_phys))

        if targets is None:
            targets = locate_points(space, x_phys)
        elems, ref0 = np.asarray(targets[0]), jnp.asarray(targets[1])

        # element indices are concrete, so every gather happens on the host and
        # the traced part is one batched Newton solve
        dofs = np.asarray(space.elem_dofs)[elems]           # (n_pts, n_local)
        cpts_all, wgts_all = padded_control_data(space)
        C = space.extraction[elems]                          # (n_pts, n_local, n_bernstein)
        cpts = cpts_all[dofs]                                # (n_pts, n_local, dim_phys)
        wgts = wgts_all[dofs]                                # (n_pts, n_local)
        u_local = pad_coeffs(self.u, vec)[expand_dofs_traced(dofs, vec)].reshape(
            len(elems), -1, vec
        )

        kinds = (np.asarray(space.element_types)[elems] if space.cell_type == "mixed"
                 else np.zeros(len(elems), dtype=int))

        def one(target, ref_init, Ce, cl, wl, ul, kind):
            def basis(r):
                M = (Ce @ reference_bernstein(space, r)) * wl
                return M / jnp.sum(M)

            def step(_, ref):
                x = basis(ref) @ cl
                dx_dref = jax.jacfwd(lambda r: basis(r) @ cl)(ref)
                # jacfwd gives dx_dref[i, a] = dx_i/dref_a, which is exactly
                # the Newton matrix -- no transpose
                return clip_reference(space, ref + jnp.linalg.solve(dx_dref, target - x), kind)

            ref = jax.lax.fori_loop(0, steps, step, ref_init)
            R = basis(ref)
            missed = jnp.linalg.norm(target - R @ cl) > 1e-6
            return jnp.where(missed, jnp.nan, R @ ul)

        return jax.vmap(one)(x_phys, ref0, C, cpts, wgts, u_local, kinds)

    # -- output ------------------------------------------------------------

    def cell_average(self, field=None, *, quadrature=None, chunk=32):
        """Volume-weighted field averages, evaluated at interior Gauss points.

        With no field name, average the solution itself. Useful for stress
        output when a geometry has collapsed edges with singular endpoint
        Jacobians: no derivatives are sampled at those endpoints.
        """
        result = []
        order = 2 if field and self.problem is not None and self.problem.needs_hessian else 1
        for basis in evaluate(self.space, P.gauss(self.space, quadrature), order=order, chunk=chunk):
            values = self.field(field, basis=basis) if field else self.at(basis=basis)
            weights = basis.w.reshape(basis.w.shape + (1,) * (values.ndim - 2))
            result.append(jnp.sum(weights * values, axis=1) / jnp.sum(weights, axis=1))
        return jnp.concatenate(result)

    def to_vtk(self, path: str, n: int = 10, fields=(), cell_fields=()):
        from jaxiga.post.vtk import write_vtk

        return write_vtk(self, path, n=n, fields=fields, cell_fields=cell_fields)

    def plot(self, field: str | None = None, n: int = 10, **kwargs):
        """Open an interactive pyvista window showing the solution.

        ``field`` names a derived field of the problem (``"von_mises"``,
        ``"stress"``, ...); the default plots the solution itself, coloured by
        magnitude for a vector field.

        The mesh is written through the same VTK path as :meth:`to_vtk`, so
        what is shown is what would be written. Requires ``pyvista``, which is
        an optional dependency.
        """
        try:
            import pyvista
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Solution.plot needs pyvista: pip install pyvista. "
                "Use to_vtk() to write a file for an external viewer instead."
            ) from exc

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "solution")
            self.to_vtk(path, n=n, fields=() if field is None else (field,))
            grid = pyvista.read(path + ".vtu")

        name = field
        if name is None:
            names = list(grid.point_data.keys())
            name = names[0] if names else None
        return grid.plot(scalars=name, **kwargs)

    def summary(self) -> str:
        return (
            f"Solution(n_dofs={self.u.shape[0]}, vec={self.space.vec}, "
            f"|u|_inf={float(jnp.max(jnp.abs(self.u))):.6g}, stats={self.stats})"
        )


def locate_points(space, x_phys, candidates: int = 12, steps: int = 20):
    """Find which element contains each physical point, and a starting guess.

    Host-side, so call it once outside any ``jit`` and pass the result to
    :meth:`Solution.probe`.

    A nearest-sample search alone is not enough: for a point close to an
    element boundary the nearest sample can belong to a neighbour that does not
    contain it, and the clipped Newton then never reaches the target. So the
    several nearest candidates are tried in order and the first whose inverted
    reference coordinate actually lands inside the element is taken.

    Returns ``(elems, ref0)``: the element index and initial reference
    coordinate for each point.
    """
    from scipy.spatial import cKDTree

    x_phys = np.atleast_2d(np.asarray(x_phys, dtype=float))
    dim = space.dim

    ps = P.grid(space, 6)
    basis = evaluate(space, ps)
    cand_x = np.asarray(basis.x).reshape(-1, basis.x.shape[-1])
    cand_e = np.repeat(np.asarray(ps.elems), basis.n_q)
    cand_ref = np.broadcast_to(
        np.asarray(ps.ref), (basis.n_elems, basis.n_q, dim)
    ).reshape(-1, dim)

    k = min(candidates, len(cand_x))
    _, nearest = cKDTree(cand_x).query(x_phys, k=k)
    nearest = np.atleast_2d(nearest)

    elems = np.empty(len(x_phys), dtype=int)
    ref0 = np.empty((len(x_phys), dim))
    tol = 1e-9

    for i, target in enumerate(x_phys):
        best = None
        seen = set()
        for j in nearest[i]:
            elem = int(cand_e[j])
            if elem in seen:
                continue
            seen.add(elem)

            ref = np.asarray(cand_ref[j], dtype=float)
            for _ in range(steps):
                _, x, dx_dref = rational_at(space, elem, jnp.asarray(ref))
                delta = target - np.asarray(x)
                if np.linalg.norm(delta) < tol:
                    break
                # clip every step: an unconstrained Newton can wander far
                # outside the element and then fail even for points that are
                # genuinely inside it
                ref = np.asarray(clip_reference(space, ref + np.linalg.solve(np.asarray(dx_dref), delta),
                    np.asarray(space.element_types)[elem] if space.cell_type == "mixed" else False))

            _, x, _ = rational_at(space, elem, jnp.asarray(ref))
            residual = float(np.linalg.norm(target - np.asarray(x)))

            if best is None or residual < best[2]:
                best = (elem, ref.copy(), residual)
            if residual < 1e-9:
                break

        elems[i], ref0[i] = best[0], best[1]

    return elems, ref0
