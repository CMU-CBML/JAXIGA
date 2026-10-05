"""Problem definition: pointwise physics plus boundary conditions.

A problem is defined by a pointwise energy density ``psi(grad_u, u, x, params)``.
Everything the solution methods need is derived from it by automatic
differentiation, so one definition serves the Galerkin, energy-minimization and
collocation paths.

Nonvariational problems may override ``flux`` and ``mass`` for collocation.
Galerkin and energy minimisation require ``energy``; their residuals and
tangents are derivatives of the discrete potential.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.space.function_space import expand_dofs


@dataclasses.dataclass(frozen=True)
class DirichletBC:
    """Prescribe solution values on part of the boundary.

    Parameters
    ----------
    value : float or callable
        A constant, or ``g(x, params)`` returning the value at a physical point.
    where : str or callable
        A boundary label, or a predicate ``pred(x) -> bool array`` over
        physical coordinates.
    component : int or None
        Which solution component to constrain; ``None`` constrains all of them.

    Note on ``where`` predicates: for Dirichlet conditions the predicate is
    evaluated at the **control points** of boundary dofs and selects dofs,
    not elements. Dirichlet data lives on dofs, and dof granularity is what
    makes single-point constraints (used to remove rigid-body modes)
    expressible. Neumann predicates select elements, since they are integrated.
    """

    value: float | Callable
    where: str | Callable
    component: int | None = None


@dataclasses.dataclass(frozen=True)
class ClampedBC(DirichletBC):
    """Clamp a boundary of a second-gradient problem: value *and* slope.

    A C^1 discretization needs two layers of control points constrained to fix
    the normal derivative, not one. This resolves to the boundary dofs plus the
    adjacent tensor-grid layer, which is exactly a zero normal slope for an
    open knot vector. Only homogeneous clamping is supported.
    """

    value: float | Callable = 0.0
    where: str = ""
    component: int | None = None

    def __post_init__(self):
        if not isinstance(self.where, str):
            raise TypeError("ClampedBC takes a boundary label, not a predicate")
        if callable(self.value) or float(self.value) != 0.0:
            raise ValueError(
                "only homogeneous clamping is supported; a non-zero slope needs "
                "the second control-point layer positioned explicitly"
            )


@dataclasses.dataclass(frozen=True)
class Neumann:
    """Prescribe a traction or flux on part of the boundary.

    ``value`` is ``t(x, n, params)`` returning a ``(vec,)`` array, where ``n``
    is the outward unit normal.
    """

    value: Callable
    where: str | Callable


class Problem:
    """Base class. Subclass and override :meth:`energy` (or ``flux``/``mass``)."""

    fields: dict = {}
    is_linear: bool = False
    # Opt-in for automatic CG selection, assuming admissible material
    # coefficients and boundary conditions that remove the nullspace.
    is_positive_definite: bool = False
    # Second-gradient problems (Kirchhoff-Love plates and shells, strain
    # gradient elasticity). When True the interior Gauss set is evaluated at
    # order 2 and ``energy`` also receives ``hess_u`` of shape ``(vec, n_d2)``
    # in Voigt order. C^1 continuity comes free inside a patch at p >= 2, which
    # is the capability that separates IGA from standard FEM here.
    needs_hessian: bool = False

    def __init__(
        self,
        space,
        *,
        dirichlet: Sequence[DirichletBC] = (),
        neumann: Sequence[Neumann] = (),
        source: Callable | None = None,
        quadrature: int | None = None,
    ):
        if space.cell_type in {"simplex", "mixed"} and self.needs_hessian:
            raise NotImplementedError("simplex spaces are C0; fourth-order forms require C1 coupling")
        self.space = space
        self.dirichlet = tuple(dirichlet)
        self.neumann = tuple(neumann)
        self.source = source
        self.quadrature = quadrature

        for bc in self.dirichlet:
            if bc.component is not None and not (0 <= bc.component < space.vec):
                raise ValueError(
                    f"DirichletBC component {bc.component} is out of range for a "
                    f"space with vec={space.vec}"
                )

        # Which dofs each condition constrains is fixed at construction, using
        # concrete control-point locations: `where` predicates are setup-time
        # by contract (they cannot depend on params, and must not change under
        # tracing). Only the prescribed *values* are recomputed per solve, so
        # they may be traced and carry gradients.
        cpts = np.asarray(space.cpts)
        self._dirichlet_dofs = tuple(
            (bc, _resolve_dirichlet_dofs(space, bc, cpts)) for bc in self.dirichlet
        )
        # Neumann selections are stored on the problem, never written back into
        # the space: two problems may select different parts of the same labeled
        # side, and a registry keyed by anything less than the selection itself
        # would let one silently overwrite the other.
        self._neumann_sets = tuple(
            (bc, _resolve_neumann_set(space, bc.where)) for bc in self.neumann
        )

    # -- pointwise kernels -------------------------------------------------

    def energy(self, grad_u, u, x, params, hess_u=None):
        """Energy density at one material point.

        ``grad_u`` is ``(vec, dim)``, ``u`` is ``(vec,)``, ``x`` is ``(dim,)``.
        ``hess_u`` is ``(vec, n_d2)`` in Voigt order, and is passed only when
        the problem sets ``needs_hessian``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} defines neither energy() nor flux()/mass(); "
            "override energy() to use the Galerkin or energy methods"
        )

    def flux(self, grad_u, u, x, params):
        """``d psi / d grad_u``, shape ``(vec, dim)`` (JAX-FEM tensor map)."""
        return jax.grad(self.energy, argnums=0)(grad_u, u, x, params)

    def mass(self, grad_u, u, x, params):
        """``d psi / d u``, shape ``(vec,)`` (JAX-FEM mass map)."""
        return jax.grad(self.energy, argnums=1)(grad_u, u, x, params)

    # -- problem rebinding -------------------------------------------------

    def with_space(self, space) -> "Problem":
        """A copy of this problem on a different (same-connectivity) space.

        Used for shape derivatives: the geometry changes, the physics does not.
        """
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.space = space
        return clone

    # -- boundary condition resolution (setup stage) ----------------------

    def resolve_dirichlet(self, params=None):
        """Prescribed global dof indices and their values.

        Values are interpolated at control points, which is exact for the
        constant data used throughout the examples. Non-constant data is
        approximated; L2 projection of boundary data is the documented v2
        upgrade.
        """
        space = self.space
        # Traced-safe: under a shape derivative these are tracers.
        cpts = space.cpts

        entries = {}  # global dof -> value (possibly traced)
        provenance = {}  # global dof -> (bc index, scalar dof) for error messages

        for bc_idx, (bc, scalar_dofs) in enumerate(self._dirichlet_dofs):
            comps = range(space.vec) if bc.component is None else [bc.component]

            for sd in scalar_dofs:
                if callable(bc.value):
                    # may legitimately depend on params, so it can be traced
                    val = jnp.atleast_1d(jnp.asarray(bc.value(cpts[sd], params)))
                else:
                    # a literal stays concrete: wrapping it in jnp inside a jit
                    # trace would turn a constant into a tracer for no reason
                    val = np.atleast_1d(np.asarray(bc.value, dtype=float))

                for c in comps:
                    g = int(space.vec * sd + c)
                    v = val[c] if val.shape[0] == space.vec else val[0]

                    if g in entries:
                        prev_bc, prev_sd = provenance[g]
                        if _conflicts(entries[g], v):
                            raise ValueError(
                                f"conflicting Dirichlet conditions on dof {g} "
                                f"(component {c} of scalar basis {sd}): boundary "
                                f"conditions {prev_bc} and {bc_idx} prescribe "
                                f"different values"
                            )
                    entries[g] = v
                    provenance[g] = (bc_idx, sd)

        if not entries:
            return np.zeros(0, dtype=int), jnp.zeros(0)

        dofs = np.array(sorted(entries), dtype=int)
        vals = [entries[int(d)] for d in dofs]
        # keep an all-constant set concrete; mixed sets promote to traced
        if any(isinstance(v, jax.core.Tracer) or isinstance(v, jnp.ndarray) for v in vals):
            values = jnp.stack([jnp.asarray(v) for v in vals])
        else:
            values = np.asarray(vals, dtype=float)
        return dofs, values

    def resolve_neumann(self):
        """``(condition, BoundarySet)`` for each Neumann condition."""
        return list(self._neumann_sets)


def _conflicts(a, b) -> bool:
    """Whether two prescribed values disagree, skipping the check when traced."""
    try:
        return abs(float(a) - float(b)) > 1e-12
    except Exception:
        # Traced values cannot be compared at trace time; the setup-stage run
        # that built this problem already validated the concrete case.
        return False


def _clamped_dofs(space, label) -> np.ndarray:
    """Boundary dofs of a label plus the adjacent control-point layer."""
    from jaxiga.geometry.multipatch import side_indices

    if space.cell_type in {"simplex", "mixed"}:
        raise NotImplementedError("ClampedBC requires C1 continuity; simplex spaces are C0")
    if space.is_hierarchical:
        raise NotImplementedError(
            "ClampedBC on a hierarchical space is not supported: the second "
            "control-point layer is not a single tensor-grid layer once levels "
            "mix. Clamp on a tensor-product space, or use a simply supported edge."
        )
    if space.patches is None:
        raise ValueError("ClampedBC needs a space built from patches")

    dofs = set(int(d) for d in np.asarray(space.boundary(label).dofs))
    found = False
    for i, patch in enumerate(space.patches):
        labels = patch.label_map
        for d in range(patch.dim):
            for end in (0, 1):
                side = f"{'uvw'[d]}{end}"
                if labels.get(side, f"patch{i}/{side}") != label:
                    continue
                found = True
                inner = side_indices(patch, side, offset=1)
                dofs.update(
                    int(g) for g in np.asarray(space.topology.local_to_global[i])[inner]
                )
    if not found:
        raise KeyError(f"no patch side carries the label {label!r}")
    return np.array(sorted(dofs), dtype=int)


def _resolve_dirichlet_dofs(space, bc, cpts) -> np.ndarray:
    if isinstance(bc, ClampedBC):
        return _clamped_dofs(space, bc.where)
    if isinstance(bc.where, str):
        return np.asarray(space.boundary(bc.where).dofs)

    # predicate over the union of boundary dofs, evaluated at control points
    all_boundary = np.unique(
        np.concatenate([np.asarray(b.dofs) for b in space.boundaries.values()])
    )
    mask = np.asarray(bc.where(cpts[all_boundary]), dtype=bool)
    if mask.shape != all_boundary.shape:
        raise ValueError(
            f"Dirichlet predicate returned shape {mask.shape}, expected "
            f"{all_boundary.shape} (one bool per boundary control point)"
        )
    selected = all_boundary[mask]
    if selected.size == 0:
        raise ValueError("Dirichlet predicate selected no boundary dofs")
    return selected


def _resolve_neumann_set(space, where):
    """The boundary elements a Neumann condition acts on.

    ``where`` is a label, or a predicate selecting boundary *elements* by the
    physical centre of the side each presents. Granularity is per element rather
    than per label because a traction is integrated, and must be able to cover
    part of a labeled side -- one edge of a square outer boundary that carries a
    single label, for instance.

    Returns a :class:`~jaxiga.space.function_space.BoundarySet`, which the
    caller keeps. Nothing is registered on the space: a selection is a property
    of the problem that made it.
    """
    from jaxiga.space.function_space import BoundarySet

    if isinstance(where, str):
        return space.boundary(where)

    elems, sides = [], []
    for bset in space.boundaries.values():
        elems.append(np.asarray(bset.elems))
        sides.append(np.asarray(bset.sides))
    if not elems:
        raise ValueError("the space has no boundary sets")

    elems = np.concatenate(elems)
    sides = np.concatenate(sides)
    # A boundary element can carry two labels where labeled sides meet, so
    # deduplicate before evaluating the predicate.
    _, unique = np.unique(np.stack([elems, sides], axis=1), axis=0, return_index=True)
    elems, sides = elems[np.sort(unique)], sides[np.sort(unique)]

    centres = _boundary_side_centres(space, elems, sides)
    mask = np.asarray(where(centres), dtype=bool).ravel()
    if mask.shape != elems.shape:
        raise ValueError(
            f"Neumann predicate returned shape {mask.shape}, expected {elems.shape} "
            f"(one bool per boundary element side centre)"
        )
    if not mask.any():
        raise ValueError(
            "Neumann predicate selected no boundary elements; it is evaluated at "
            "the physical centre of each boundary element side"
        )

    kept_elems, kept_sides = elems[mask], sides[mask]
    dofs = _dofs_of_sides(space, kept_elems, kept_sides)
    return BoundarySet(elems=kept_elems, sides=kept_sides, dofs=dofs)


def _dofs_of_sides(space, elems, sides) -> np.ndarray:
    """Scalar dofs supported on the given (element, side) pairs."""
    elem_dofs = np.asarray(space.elem_dofs)
    dofs = set()
    for e in elems:
        dofs.update(int(d) for d in elem_dofs[int(e)])
    return np.array(sorted(dofs), dtype=int)


def _boundary_side_centres(space, elems, sides) -> np.ndarray:
    """Physical centre of the side each boundary element presents.

    Evaluated through the geometry map, so it is exact for curved boundaries
    rather than a control-point average.
    """
    from jaxiga.space.evaluation import evaluate
    from jaxiga.space.pointset import PointSet

    dim = space.dim
    ref = np.zeros((len(elems), 1, dim))
    for i, code in enumerate(sides):
        direction, end = divmod(int(code), 2)
        ref[i, 0, direction] = -1.0 if end == 0 else 1.0
    ps = PointSet(
        elems=np.asarray(elems), ref=ref, sides=np.asarray(sides), tag="side-centres"
    )
    return np.asarray(evaluate(space, ps).x)[:, 0, :]
