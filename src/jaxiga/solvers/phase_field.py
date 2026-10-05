r"""Staggered phase-field fracture with a mesh that follows the crack.

The AT2 model of brittle fracture needs several elements across a damage band
of width :math:`\ell`, but only *along the crack*. A mesh fine enough there and
coarse everywhere else is the difference between a tractable simulation and an
intractable one, and where the crack will go is not known in advance -- so the
mesh has to be built while the solve runs.

The scheme here follows the IGAPack-PhaseField MATLAB code: alternate
minimisation at each load increment, refining any element whose damage exceeds
a threshold, up to a maximum level. Damage is a sharper indicator than a
residual estimator for this problem, because the thing that must be resolved is
the band itself, whose location damage reports directly.

Refinement is hierarchical (THB) on the parametric domain, so refined meshes
are nested and the fields carried across a remesh transfer exactly; see
:mod:`jaxiga.space.transfer`. The damage field is projected, and the history
field -- a running maximum living at quadrature points, not a spline -- is
copied from the nearest point of the parent element.

Works in 2D and 3D. The subproblems are solved directly in 2D, where one
factorisation serves many sweeps, and by preconditioned conjugate gradients in
3D, where fill-in makes a factorisation cost more than the assembly feeding it
and it cannot be reused often enough to amortise.

Everything mesh-dependent is rebuilt on a remesh and cached in between: the
basis at the quadrature points, the two method contexts, the sparsity patterns
and the LU factorisations. Between remeshes the only thing that changes is the
prescribed displacement, which enters linearly.

Example
-------
::

    solver = StaggeredSolver(
        space, Material(E=210e3, nu=0.3), Fracture(Gc=2.7, ell=0.0125),
        dirichlet=[jx.DirichletBC([0.0, 0.0], where="bottom"),
                   jx.DirichletBC([0.0, 1.0], where="top")],
        reaction=("top", 1),
        crack=edge_crack_distance((0.5, 0.5)),
    )
    for step in solver.run(loads):
        print(step.load, step.force)
"""

from __future__ import annotations

import dataclasses
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.forms.library import (
    PhaseFieldDamage,
    PhaseFieldDisplacement,
    tensile_energy,
)
from jaxiga.methods._common import PointField, build_context, energy_of_dofs
from jaxiga.methods.galerkin import assemble_csr_values, rhs
from jaxiga.post.solution import Solution
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate
from jaxiga.space.hierarchical import refine_elements
from jaxiga.space.transfer import (
    carry_points,
    coarse_basis_at,
    parent_map,
    project,
)

__all__ = [
    "Material",
    "Fracture",
    "Adaptivity",
    "Step",
    "StaggeredSolver",
    "CachedLU",
    "DeviceCG",
    "JacobiCG",
    "seed_refine",
]


# --------------------------------------------------------------------------
# model description
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Material:
    """Isotropic elastic constants and the residual stiffness of broken material."""

    E: float
    nu: float
    plane: str = "strain"
    k: float = 1e-8

    def params(self):
        return {"E": self.E, "nu": self.nu, "k": self.k}


@dataclasses.dataclass(frozen=True)
class Fracture:
    """Griffith energy, regularisation length, and the strength of the crack seed.

    ``B`` scales the history spike that represents a pre-existing crack; the
    value only has to be large enough to drive damage to one there.
    """

    Gc: float
    ell: float
    B: float = 1e3


@dataclasses.dataclass(frozen=True)
class Adaptivity:
    """When and how far to refine.

    Attributes
    ----------
    threshold : float
        Refine an element if damage exceeds this at any of its quadrature
        points. Half is the usual choice: it sits on the steep flank of the AT2
        profile, so the marked set tracks the band rather than its long tails.
    max_level : int
        Deepest refinement level, counted from the base mesh.
    dilate : float
        Look-ahead, in units of ``ell``: also refine anything within this
        distance of a damaged element. Elements adjacent to a damaged one are
        always included, so ``0`` still keeps the mesh one layer ahead of the
        crack -- close to what the MATLAB reference does.

        This is the parameter that trades element count against rebuild count,
        and neither term is negligible. Too small and the crack outruns the
        mesh, forcing a remesh every couple of increments; too large and the
        band is refined far wider than the damage that justifies it. Around one
        is a reasonable starting point.
    max_per_step : int
        Cap on remeshes within one load increment. A crack that jumps several
        elements in one increment needs more than one pass, but an unbounded
        loop would let a diverging solve refine without end.
    enabled : bool
        ``False`` freezes the mesh, which is what the fixed-mesh reference runs.
    """

    threshold: float = 0.5
    max_level: int = 4
    dilate: float = 1.0
    max_per_step: int = 4
    enabled: bool = True


@dataclasses.dataclass
class Step:
    """The state at the end of one load increment."""

    index: int
    load: float
    force: float
    sweeps: int
    remeshed: bool
    n_elems: int
    n_dofs: int
    seconds: float
    space_u: object = None
    space_d: object = None
    sol_u: object = None
    sol_d: object = None
    # Incoming-damage relative residual used by the staggered stopping rule.
    residual: float = float("inf")
    converged: bool = False


# --------------------------------------------------------------------------
# linear algebra
# --------------------------------------------------------------------------


class CachedLU:
    """A SuperLU factorisation reused until iterative refinement stops working.

    Factorising costs about forty times what the triangular solves cost, the
    sparsity never changes while the mesh is fixed, and with small load
    increments the matrix barely moves between staggered sweeps. Reusing the
    factorisation and refining the result removes most of the linear-algebra
    cost; the residual test keeps the solve exact, and a factorisation that no
    longer works is simply rebuilt.
    """

    #: Host solver: ``_sweep`` must bring the matrix and right-hand side over.
    on_device = False

    def __init__(self, pattern, tol=1e-5):
        self.indptr, self.cols = pattern.indptr, pattern.column_indices
        self.shape, self.tol = pattern.shape, tol
        self.lu, self.refactor, self.calls = None, 0, 0

    def matrix(self, data):
        import scipy.sparse as sp

        return sp.csr_matrix(
            (np.asarray(data), self.cols, self.indptr), shape=self.shape
        )

    def relative_residual(self, data, b, x):
        """``|Ax - b| / |b|``, the staggered convergence measure."""
        return np.linalg.norm(self.matrix(data) @ x - b) / (np.linalg.norm(b) or 1.0)

    def solve(self, data, b):
        import scipy.sparse.linalg as spla

        self.calls += 1
        A = self.matrix(data).tocsc()
        nb = np.linalg.norm(b) or 1.0
        if self.lu is not None:
            x = self.lu.solve(b)
            r = b - A @ x
            if np.linalg.norm(r) > self.tol * nb:
                x = x + self.lu.solve(r)
                r = b - A @ x
            if np.linalg.norm(r) <= self.tol * nb:
                return x
        self.lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
        self.refactor += 1
        return self.lu.solve(b)

    def report(self):
        return (
            f"{self.calls} solves, {self.calls - self.refactor} of them reusing "
            "a factorisation"
        )


class JacobiCG:
    """Preconditioned conjugate gradients on the same sparsity pattern.

    The degraded-elasticity operator is symmetric positive definite -- the
    residual stiffness ``k`` keeps it so even inside a fully broken region --
    so conjugate gradients apply, and in 3D they are the only sensible choice.
    A direct factorisation of a 3D system fills in badly: measured on the
    notched cube, ``splu`` fill grows from 2.8 times the matrix at 3k degrees of
    freedom to 8.5 times at 15k, and the factorisation goes from 0.19s to 13s
    while the assembly that feeds it stays under half a second. Jacobi-CG
    solves the same systems in 0.1s and 1.1s.

    Iteration counts here are flat in the damage -- around 130, and slightly
    *fewer* once a crack has formed than before it -- so the near-singular
    stiffness of broken material is not the difficulty it might look like.

    Each solve warm-starts from the previous one, which is the same observation
    that makes :class:`CachedLU` pay in 2D: consecutive staggered sweeps ask
    almost the same question.
    """

    #: Host solver: ``_sweep`` must bring the matrix and right-hand side over.
    on_device = False

    def __init__(self, pattern, tol=1e-9, maxiter=5000):
        self.indptr, self.cols = pattern.indptr, pattern.column_indices
        self.shape, self.tol, self.maxiter = pattern.shape, tol, maxiter
        self.x0 = None
        self.calls, self.iterations, self.failures = 0, 0, 0

    def matrix(self, data):
        import scipy.sparse as sp

        return sp.csr_matrix(
            (np.asarray(data), self.cols, self.indptr), shape=self.shape
        )

    def relative_residual(self, data, b, x):
        """``|Ax - b| / |b|``, the staggered convergence measure."""
        return np.linalg.norm(self.matrix(data) @ x - b) / (np.linalg.norm(b) or 1.0)

    def solve(self, data, b):
        import scipy.sparse.linalg as spla

        self.calls += 1
        A = self.matrix(data)
        inverse_diagonal = 1.0 / A.diagonal()
        precondition = spla.LinearOperator(
            self.shape, matvec=lambda v: v * inverse_diagonal
        )
        count = [0]
        x, info = spla.cg(
            A,
            b,
            x0=self.x0,
            rtol=self.tol,
            M=precondition,
            maxiter=self.maxiter,
            callback=lambda _: count.__setitem__(0, count[0] + 1),
        )
        self.iterations += count[0]
        if info != 0:
            # Not fatal on its own -- report it and hand back the best iterate,
            # so a run says so rather than silently returning garbage.
            self.failures += 1
        self.x0 = x
        return x

    def report(self):
        return (
            f"{self.calls} CG solves, {self.iterations} iterations "
            f"({self.iterations / max(self.calls, 1):.0f} per solve)"
            + (f", {self.failures} did not converge" if self.failures else "")
        )


class DeviceCG:
    """The same Jacobi-CG as :class:`JacobiCG`, without leaving the device.

    ``JacobiCG`` is ``scipy.sparse.linalg.cg`` on the host, so every sweep
    moves the matrix and right-hand side of both subproblems across the bus and
    the result back -- four crossings per sweep -- and then does the arithmetic
    on one core. A three-dimensional run therefore sits at high CPU with the
    accelerator idle. This keeps the whole solve where the assembly already
    is.

    Measured on the notched-cube displacement systems, against the host solver
    at the same tolerance and to the same agreement with a direct
    factorisation (``paper/bench_device_cg.py``):

    ======  ==========  ==========  =======
     dofs    host CG     this        factor
    ======  ==========  ==========  =======
      3068     0.36 s     0.019 s     19.5
      7162     1.13 s     0.037 s     30.6
     15452     3.87 s     0.114 s     33.9
    ======  ==========  ==========  =======

    ``data`` and ``b`` are arguments of the compiled function rather than
    captured constants, so XLA does not fold them into the executable; only the
    sparsity indices are closed over. Compilation costs 0.8-5.0 s once per
    mesh, against the hundreds of sweeps a mesh serves.

    ``jax.scipy.sparse.linalg.cg`` does not report an iteration count, so
    unlike :class:`JacobiCG` this cannot say how many it took; ``calls`` is
    still counted, and convergence is checked by the caller's residual test.
    """

    #: Everything stays in device arrays; ``_sweep`` skips the host round trip.
    on_device = True

    def __init__(self, pattern, tol=1e-9, maxiter=5000):
        self.indptr, self.cols = pattern.indptr, pattern.column_indices
        self.shape, self.tol, self.maxiter = pattern.shape, tol, maxiter
        self.x0 = None
        self.calls, self.iterations, self.failures = 0, 0, 0
        self._build()

    def _build(self):
        n = self.shape[0]
        rows = jnp.asarray(np.repeat(np.arange(n), np.diff(self.indptr)))
        cols = jnp.asarray(self.cols)
        tol, maxiter = self.tol, self.maxiter

        def matvec_with(data):
            return lambda v: jnp.zeros(n, dtype=data.dtype).at[rows].add(
                data * v[cols]
            )

        @jax.jit
        def solve(data, b, x0):
            diagonal = (
                jnp.zeros(n, dtype=data.dtype)
                .at[rows]
                .add(jnp.where(rows == cols, data, 0.0))
            )
            safe = jnp.where(jnp.abs(diagonal) > 0, diagonal, 1.0)
            inv_diagonal = jnp.where(jnp.abs(diagonal) > 0, 1.0 / safe, 1.0)
            x, _ = jax.scipy.sparse.linalg.cg(
                matvec_with(data), b, x0=x0, tol=tol, atol=0.0,
                maxiter=maxiter, M=lambda v: inv_diagonal * v,
            )
            return x

        @jax.jit
        def relative_residual(data, b, x):
            r = matvec_with(data)(x) - b
            nb = jnp.linalg.norm(b)
            return jnp.linalg.norm(r) / jnp.where(nb > 0, nb, 1.0)

        self._solve, self._relative_residual = solve, relative_residual

    def solve(self, data, b):
        self.calls += 1
        data, b = jnp.asarray(data), jnp.asarray(b)
        x0 = self.x0 if self.x0 is not None else jnp.zeros_like(b)
        x = self._solve(data, b, x0)
        self.x0 = x
        return x

    def relative_residual(self, data, b, x):
        """``|Ax - b| / |b|``, computed on the device."""
        return self._relative_residual(jnp.asarray(data), jnp.asarray(b),
                                       jnp.asarray(x))

    def release(self):
        """Drop the compiled kernels, as :meth:`_Mesh.release` does for its own."""
        for fn in (self._solve, self._relative_residual):
            fn.clear_cache()

    def report(self):
        return f"{self.calls} device CG solves"


# --------------------------------------------------------------------------
# initial mesh
# --------------------------------------------------------------------------


def seed_refine(space, crack, ell, levels, *, width=1.0):
    """Refine to ``levels`` around a pre-existing crack, before any solving.

    The initial crack is given geometry, not something the solver has to
    discover, and the history spike that represents it is only ``ell`` wide --
    on a coarse mesh it would fall between quadrature points. Resolving it up
    front is what makes the first load step meaningful; everything after is
    found by the damage indicator.
    """
    for _ in range(levels):
        basis = evaluate(space, P.gauss(space))
        dist = np.asarray(jax.vmap(jax.vmap(crack))(basis.x))
        near = (dist <= width * ell).any(axis=1)
        level = np.asarray(space.elem_key)[:, 1]
        marked = np.flatnonzero(near & (level < levels))
        if not marked.size:
            break
        space = refine_elements(space, marked)
    return space


# --------------------------------------------------------------------------
# per-mesh state
# --------------------------------------------------------------------------


class _Mesh:
    """Everything tied to one mesh: spaces, contexts, patterns, factorisations.

    Rebuilt from scratch on a remesh and reused for every load step and every
    staggered sweep in between. Going through :func:`jaxiga.solve` instead would
    rebuild all of it on each of the thousands of solves a run performs.
    """

    def __init__(self, space_d, material, fracture, dirichlet, reaction, linear):
        self.Vd = space_d
        # Scalar connectivity, extraction and boundaries do not depend on the
        # number of components, so the displacement space is the damage space
        # with a different `vec` -- rebuilding it would repeat the most
        # expensive part of a remesh for nothing.
        self.Vu = dataclasses.replace(space_d, vec=space_d.dim)
        self.basis = evaluate(self.Vu, P.gauss(self.Vu))
        self.ref = P.gauss(self.Vu).ref
        self.material, self.fracture = material, fracture

        self.disp = PhaseFieldDisplacement(self.Vu, plane=material.plane, dirichlet=dirichlet)
        self.damage = PhaseFieldDamage(self.Vd)

        # Physical element centres and half-extents, for dilating a marked set.
        # Read off the quadrature points, so this stays valid for curved
        # geometry and in any dimension.
        pts = np.asarray(self.basis.x)
        self.centre = pts.mean(axis=1)
        self.radius = np.linalg.norm(0.5 * (pts.max(axis=1) - pts.min(axis=1)), axis=1)

        zero = jnp.zeros((self.basis.n_elems, self.basis.n_q))
        mat = material.params()
        self.ctx_u = build_context(self.disp, {**mat, "phi": PointField(zero)})
        self.ctx_d = build_context(
            self.damage,
            {"Gc": fracture.Gc, "l": fracture.ell, "H": PointField(zero)},
        )
        self.lu_u = linear(self.ctx_u.assembly)
        self.lu_d = linear(self.ctx_d.assembly)

        base = self.ctx_u.dofmap.values
        free = np.asarray(self.ctx_u.dofmap.free)
        pres = np.asarray(self.ctx_u.dofmap.prescribed)
        label, comp = reaction
        vec = self.Vu.vec
        self.react = jnp.asarray(vec * np.asarray(self.Vu.boundaries[label].dofs) + comp)

        ctx_u, ctx_d, disp, damage = self.ctx_u, self.ctx_d, self.disp, self.damage
        n_dofs, Gc, ell = self.Vu.n_dofs, fracture.Gc, fracture.ell
        react = self.react

        def _u_params(phi):
            return {**mat, "phi": PointField(phi)}

        def _d_params(hist):
            return {"Gc": Gc, "l": ell, "H": PointField(hist)}

        self.u_params, self.d_params = _u_params, _d_params

        @jax.jit
        def assemble_u(load, phi):
            dm = dataclasses.replace(ctx_u.dofmap, values=base * load)
            ctx = dataclasses.replace(ctx_u, dofmap=dm)
            p = _u_params(phi)
            return rhs(disp, p, ctx), assemble_csr_values(ctx, p)

        @jax.jit
        def lift_u(u_free, load):
            return jnp.zeros(n_dofs).at[free].set(u_free).at[pres].set(base * load)

        @jax.jit
        def assemble_d(hist):
            p = _d_params(hist)
            return rhs(damage, p, ctx_d), assemble_csr_values(ctx_d, p)

        @jax.jit
        def driving_force(u_full, phi):
            """Tensile strain energy at the quadrature points, for the history."""
            sol = Solution(self.Vu, u_full, _u_params(phi), disp)
            return tensile_energy(sol, mat, self.basis)

        @jax.jit
        def reaction_sum(u_full, phi):
            r = jax.grad(energy_of_dofs, argnums=0)(u_full, _u_params(phi), ctx_u)
            return jnp.sum(r[react])

        self.assemble_u, self.lift_u = assemble_u, lift_u
        self.assemble_d, self.reaction_sum = assemble_d, reaction_sum
        self.driving_force = driving_force

    def release(self):
        """Drop the compiled kernels of a mesh that is no longer the current one.

        Each of the five jitted closures above closes over this mesh's basis
        and method contexts, and XLA folds them into the executable as
        constants: the basis gradients alone are
        ``n_elems * n_local * n_q * dim`` floats, a quarter of a gigabyte on a
        3D mesh of ten thousand elements. Dropping the Python objects does not
        release them, because the compiled executable is held in the jit cache
        of each function, keyed on the argument shapes. An adaptive run would
        therefore accumulate one mesh worth of device memory per remesh --
        measured on the notched cube, enough to exhaust a 40 GB device after a
        handful of them -- so the retired executables are released here, where
        the mesh they belong to goes out of use.
        """
        for fn in (self.assemble_u, self.lift_u, self.assemble_d,
                   self.reaction_sum, self.driving_force):
            fn.clear_cache()
        # A device linear solver compiles its own kernels over this mesh's
        # sparsity, so they go the same way.
        for solver in (self.lu_u, self.lu_d):
            if hasattr(solver, "release"):
                solver.release()

    def seed_history(self, crack):
        """History values representing the pre-existing crack, at quadrature points."""
        if crack is None:
            return jnp.zeros((self.basis.n_elems, self.basis.n_q))
        f = self.fracture
        dist = jax.vmap(jax.vmap(crack))(self.basis.x)
        return jnp.where(
            dist <= f.ell / 2,
            f.B * f.Gc * (1.0 - dist / (f.ell / 2)) / (2.0 * f.ell),
            0.0,
        )


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------


class _RetiredCounters:
    """What a retired linear solver contributed, without what it was holding.

    ``linear_report`` sums counters over every mesh the run has had, which is
    why retired solvers were kept. Keeping the solver object itself also pins
    its payload for the rest of the run, though -- a SuperLU factorisation in
    2D, a warm-start vector in 3D -- so only the counters are kept here.
    """

    __slots__ = ("calls", "refactor", "iterations", "failures")

    def __init__(self, solver):
        self.calls = solver.calls
        self.refactor = getattr(solver, "refactor", 0)
        self.iterations = getattr(solver, "iterations", 0)
        self.failures = getattr(solver, "failures", 0)


class StaggeredSolver:
    """Alternate minimisation of the phase-field energy, with adaptive remeshing.

    Parameters
    ----------
    space : FunctionSpace
        Scalar (``vec=1``) hierarchical space for the damage field. The
        displacement space is built on the same mesh with ``vec=dim``.
    material, fracture : Material, Fracture
    dirichlet : list of DirichletBC
        Conditions on the displacement. The *driven* condition carries value
        ``1``; the applied displacement of each step multiplies the whole
        prescribed vector, which is exact because the subproblem is linear.
    reaction : (str, int)
        Boundary label and component whose reaction is reported as the force.
    crack : callable, optional
        ``crack(x)`` returns the distance from a point to a pre-existing crack.
    adaptivity : Adaptivity
    tol : float
        Convergence tolerance of the staggered iteration, measured as the
        relative residual of the damage system at the *previous* damage field.
        A bound on the damage increment cannot be used: on the softening branch
        the crack advances during the sweeps, so the increment never vanishes at
        fixed load, while the residual does.
    max_sweeps : int
        Cap on staggered sweeps per increment. Alternate minimisation converges
        linearly at a rate that degrades as the increment grows, so a run that
        regularly hits this cap is being driven with increments that are too
        large -- the reported ``sweeps`` per step is the diagnostic.
    linear : {"auto", "direct", "cg"}
        How to solve each subproblem. ``"auto"`` picks by dimension, and the
        choice is not close: in 2D a cached factorisation is reused across most
        sweeps and wins easily, while in 3D fill-in makes the factorisation cost
        an order of magnitude more than the assembly feeding it, and it cannot
        be reused often enough to amortise. See :class:`CachedLU` and
        :class:`JacobiCG`.
    linear_tol : float, optional
        Residual tolerance for the linear solve. Defaults to 1e-5 for the direct
        path, where it decides when a reused factorisation is no longer good
        enough, and 1e-9 for CG, where it is the convergence test.
    """

    def __init__(
        self,
        space,
        material: Material,
        fracture: Fracture,
        *,
        dirichlet,
        reaction,
        crack=None,
        adaptivity: Adaptivity = Adaptivity(),
        tol: float = 1e-4,
        max_sweeps: int = 100,
        linear: str = "auto",
        linear_tol: float | None = None,
        verbose: bool = False,
    ):
        if space.vec != 1:
            raise ValueError(f"the damage space must be scalar, got vec={space.vec}")
        if space.hierarchy is None:
            space = refine_elements(space, [])
        self.material, self.fracture = material, fracture
        self.dirichlet, self.reaction_spec = list(dirichlet), tuple(reaction)
        self.crack, self.adaptivity = crack, adaptivity
        self.tol, self.max_sweeps = tol, max_sweeps
        self.verbose = verbose

        if linear == "auto":
            linear = "direct" if space.dim < 3 else "cg"
        if linear == "direct":
            tolerance = 1e-5 if linear_tol is None else linear_tol
            self.linear = lambda pattern: CachedLU(pattern, tolerance)
        elif linear == "cg":
            tolerance = 1e-9 if linear_tol is None else linear_tol
            self.linear = lambda pattern: JacobiCG(pattern, tolerance)
        elif linear == "device-cg":
            tolerance = 1e-9 if linear_tol is None else linear_tol
            self.linear = lambda pattern: DeviceCG(pattern, tolerance)
        else:
            raise ValueError(
                f"linear must be 'auto', 'direct', 'cg' or 'device-cg', "
                f"got {linear!r}"
            )
        self.linear_kind = linear

        self.mesh = _Mesh(
            space, material, fracture, self.dirichlet, self.reaction_spec, self.linear
        )
        self.hist = self.mesh.seed_history(crack)
        self.phi = jnp.zeros((self.mesh.basis.n_elems, self.mesh.basis.n_q))
        self.phi_dofs = jnp.zeros(self.mesh.Vd.n_dofs)
        self.remeshes = 0
        # Where the wall clock went. A run that spends more time rebuilding
        # meshes than solving on them wants a larger `dilate`, not a finer one.
        self.stats = {"remesh": 0.0, "rebuild": 0.0, "transfer": 0.0, "sweeps": 0}
        # Linear-solver counters live on the per-mesh solver objects, which are
        # thrown away at each remesh, so they are folded in here as meshes
        # retire. Reading them off the final mesh would report only the solves
        # since the last remesh -- which on a run that remeshes near the end is
        # almost none of them.
        self._retired = []

    def linear_report(self):
        """What the displacement solves cost, over the whole run."""
        solvers = self._retired + [self.mesh.lu_u]
        calls = sum(s.calls for s in solvers)
        if self.linear_kind == "cg":
            iterations = sum(s.iterations for s in solvers)
            failures = sum(s.failures for s in solvers)
            return (
                f"{calls} CG solves, {iterations} iterations "
                f"({iterations / max(calls, 1):.0f} per solve)"
                + (f", {failures} did not converge" if failures else "")
            )
        refactor = sum(s.refactor for s in solvers)
        return f"{calls} solves, {calls - refactor} of them reusing a factorisation"

    # -- marking and remeshing --------------------------------------------

    def _marked(self):
        """Elements to refine, or nothing if the mesh is still ahead of the crack.

        Two radii, and the distinction between them is what keeps the cost
        down. Refinement is *triggered* only when damage reaches an element
        that is not yet at the finest level -- the crack is about to run onto a
        coarse mesh and something has to be done now. When that happens the
        whole look-ahead zone is refined at once, so the crack can advance
        ``dilate`` regularisation lengths before the question arises again.

        Triggering on the look-ahead zone instead would remesh every time the
        damaged set grew by a single element, which during propagation is every
        second load increment -- the look-ahead would then buy nothing at all,
        which is exactly what measurement showed.
        """
        a = self.adaptivity
        if not a.enabled:
            return np.zeros(0, dtype=int)
        m = self.mesh
        hot = np.asarray(jnp.max(self.phi, axis=1)) > a.threshold
        if not hot.any():
            return np.zeros(0, dtype=int)
        coarse = np.asarray(m.Vd.elem_key)[:, 1] < a.max_level

        from scipy.spatial import cKDTree

        # Two boxes that touch have centres no further apart than the sum of
        # their half-diagonals, so this reach is exactly adjacency; `dilate`
        # buys reach beyond it.
        tree = cKDTree(m.centre[hot])
        distance, _ = tree.query(m.centre)
        touching = m.radius + m.radius[hot].max()
        if not (coarse & (distance <= touching)).any():
            return np.zeros(0, dtype=int)
        reach = a.dilate * self.fracture.ell + touching
        return np.flatnonzero(coarse & (distance <= reach))

    def _refine_region(self, space, marked):
        """Refine the marked elements and their descendants down to ``max_level``.

        ``refine_elements`` bisects once. Stopping there would mean a coarse
        element ahead of the crack takes one remesh per level to resolve, and a
        remesh is the expensive operation here -- two spaces, two contexts, two
        sparsity patterns and four compiled kernels, all rebuilt. Carrying the
        selection down through the levels instead resolves the whole marked
        region in one rebuild.
        """
        selected = np.zeros(space.n_elems, dtype=bool)
        selected[marked] = True
        while True:
            level = np.asarray(space.elem_key)[:, 1]
            todo = np.flatnonzero(selected & (level < self.adaptivity.max_level))
            if not todo.size:
                return space
            refined = refine_elements(space, todo)
            selected = selected[parent_map(space, refined).parent]
            space = refined

    def _remesh(self, marked):
        """Refine, rebuild every mesh-bound object, and carry the state across."""
        t0 = time.perf_counter()
        old = self.mesh
        new_space = self._refine_region(old.Vd, marked)
        pmap = parent_map(old.Vd, new_space)
        new = _Mesh(
            new_space,
            self.material,
            self.fracture,
            self.dirichlet,
            self.reaction_spec,
            self.linear,
        )
        t1 = time.perf_counter()

        # Damage: nested spaces, so projecting the coarse field is a change of
        # basis and loses nothing.
        ps_new = P.gauss(new.Vu)
        coarse = Solution(old.Vd, self.phi_dofs).at(
            basis=coarse_basis_at(old.Vd, pmap, ps_new)
        )[..., 0]
        self.phi_dofs = project(coarse, new.Vd, new.basis)
        self.phi = jnp.clip(
            Solution(new.Vd, self.phi_dofs).at(basis=new.basis)[..., 0], 0.0, 1.0
        )

        # History: point data, carried from the nearest parent point, then
        # re-seeded so the initial crack stays sharp as the mesh sharpens.
        carried = carry_points(self.hist, pmap, ps_new.ref, old.ref)
        self.hist = jnp.maximum(carried, new.seed_history(self.crack))

        self._retired.append(_RetiredCounters(old.lu_u))
        self.mesh = new
        self.remeshes += 1
        # The state has been carried across, so nothing of the old mesh is
        # needed any more. Releasing it is explicit rather than left to
        # reference counting: see :meth:`_Mesh.release`.
        old.release()
        del old
        gc.collect()
        now = time.perf_counter()
        self.stats["rebuild"] += t1 - t0
        self.stats["transfer"] += now - t1
        self.stats["remesh"] += now - t0
        return new

    # -- one staggered sweep ----------------------------------------------

    def _sweep(self, load, prev_phi_dofs):
        """One displacement solve followed by one damage solve.

        Returns the displacement solution and the relative residual of the
        damage system evaluated at the incoming damage field, which is the
        convergence measure: it is zero exactly when the incoming field already
        solved the system the current displacement produces.
        """
        m = self.mesh
        b, data = m.assemble_u(load, self.phi)
        # A device solver takes the assembled arrays as they are; a host one
        # needs them pulled across, which is two crossings here and two more
        # for the damage system below.
        if m.lu_u.on_device:
            u_free = m.lu_u.solve(data, b)
        else:
            u_free = m.lu_u.solve(np.asarray(data), np.asarray(b))
        u_full = m.lift_u(jnp.asarray(u_free), load)
        sol_u = Solution(m.Vu, u_full, m.u_params(self.phi), m.disp)

        self.hist = jnp.maximum(self.hist, m.driving_force(u_full, self.phi))

        bd, dd = m.assemble_d(self.hist)
        if not m.lu_d.on_device:
            dd, bd = np.asarray(dd), np.asarray(bd)
        res = np.inf
        if prev_phi_dofs is not None:
            res = float(m.lu_d.relative_residual(dd, bd, prev_phi_dofs))
        free_phi = m.lu_d.solve(dd, bd)
        self.phi_dofs = m.ctx_d.dofmap.lift(jnp.asarray(free_phi))
        self.phi = jnp.clip(
            Solution(m.Vd, self.phi_dofs).at(basis=m.basis)[..., 0], 0.0, 1.0
        )
        return sol_u, free_phi, res

    # -- the load loop ----------------------------------------------------

    def run(self, loads, *, keep_solutions=True, on_failure="warn"):
        """Iterate over load increments, yielding a :class:`Step` for each.

        ``loads`` is the sequence of prescribed displacement values (absolute,
        not increments). A generator, so a driver can render frames, write
        files or stop early without the solver knowing about any of it.
        ``on_failure`` is ``"warn"`` (return a flagged step) or ``"raise"``
        (stop before yielding an unconverged step). No step is silently
        described as converged when its sweep budget is exhausted.
        """
        import warnings

        if on_failure not in {"warn", "raise"}:
            raise ValueError("on_failure must be 'warn' or 'raise'")
        for index, load in enumerate(np.asarray(loads, dtype=float), start=1):
            t0 = time.perf_counter()
            prev, sweeps, remeshed, remesh_here = None, 0, False, 0
            sol_u = None

            while True:
                sol_u, prev, res = self._sweep(load, prev)
                sweeps += 1
                self.stats["sweeps"] += 1

                marked = self._marked()
                if marked.size and remesh_here < self.adaptivity.max_per_step:
                    new = self._remesh(marked)
                    remeshed, remesh_here, prev = True, remesh_here + 1, None
                    if self.verbose:
                        print(
                            f"      remesh at step {index}: {len(marked)} marked "
                            f"-> {new.Vd.n_elems} elements, {new.Vu.n_dofs} dofs"
                        )
                    continue  # the mesh changed; this step is not converged
                if res < self.tol or sweeps >= self.max_sweeps:
                    break

            converged = bool(np.isfinite(res) and res < self.tol)
            if not converged:
                message = (
                    f"fracture step {index} at load {load:g} did not converge "
                    f"after {sweeps} sweeps (relative damage residual {res:.3e})"
                )
                if on_failure == "raise":
                    raise RuntimeError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)

            # The reaction belongs to the displacement solve, so it is taken
            # with the damage that solve saw, not with the damage the sweep
            # went on to compute. At convergence the two coincide; before it,
            # pairing them would report a force no equilibrium state produced.
            m = self.mesh
            force = float(m.reaction_sum(sol_u.u, sol_u.params["phi"].values))
            yield Step(
                index=index,
                load=float(load),
                force=force,
                sweeps=sweeps,
                remeshed=remeshed,
                n_elems=int(m.Vd.n_elems),
                n_dofs=int(m.Vu.n_dofs),
                seconds=time.perf_counter() - t0,
                residual=float(res),
                converged=converged,
                space_u=m.Vu if keep_solutions else None,
                space_d=m.Vd if keep_solutions else None,
                sol_u=sol_u if keep_solutions else None,
                sol_d=(
                    Solution(m.Vd, self.phi_dofs, m.d_params(self.hist), m.damage)
                    if keep_solutions
                    else None
                ),
            )
