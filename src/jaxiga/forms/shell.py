r"""Kirchhoff--Love shells on NURBS surfaces.

The midsurface is a 2D patch with 3D control points. Its basis is evaluated
without a push-forward (see :func:`jaxiga.space.evaluation.evaluate`), so the
kernel receives *parametric* first and second derivatives of the displacement:
``grad_u[:, a] = u_{,a}`` and ``hess_u[:, k] = u_{,ab}`` in Voigt order
``(11, 12, 22)``.

The strains are written once, for the exact deformed midsurface
(Kiendl et al., CMAME 198, 2009):

* membrane: ``E_ab = (abar_a . abar_b - a_a . a_b) / 2``
* bending:  ``K_ab = a_{a,b} . a_3 - abar_{a,b} . abar_3``

with ``abar_a = a_a + u_{,a}`` and ``abar_3`` the unit normal of the deformed
surface. The linear shell uses their exact linearisation at ``u = 0``, which is
obtained by forward-mode automatic differentiation (``jax.jvp``) rather than
derived by hand; ``geometrically_nonlinear=True`` uses the strains themselves.
Covariant components are transformed to a local Cartesian frame, where the
isotropic plane-stress law gives ``n = t D eps`` and ``m = t^3/12 D kappa``.

The metric of the reference surface at each quadrature point (tangents ``a_a``
and their derivatives ``a_{a,b}``) is computed once at construction and enters
the kernel as :class:`~jaxiga.methods._common.PointField` parameters; use
:meth:`KirchhoffLoveShell.parameters` to build the ``params`` for a solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxiga.forms.problem import Problem


def _unit(v):
    return v / jnp.linalg.norm(v)


class KirchhoffLoveShell(Problem):
    """Isotropic Kirchhoff--Love shell (``vec=3`` displacement on a surface patch).

    ``params`` supplies ``E``, ``nu`` and the thickness ``t`` plus the reference
    geometry; :meth:`parameters` returns the complete dictionary. Loads per
    unit area enter through ``source=f(x, params)`` returning a 3-vector.

    Boundary conditions
    -------------------
    ``DirichletBC`` with ``component=`` fixes displacement components (a
    diaphragm, a simple support). A clamped edge additionally fixes the
    rotation and needs :class:`~jaxiga.forms.problem.ClampedBC`, which constrains
    the first two control-point rows of that side. Boundary integrals (edge
    loads) are not available on embedded surfaces.
    """

    needs_hessian = True
    is_positive_definite = True

    def __init__(self, space, *, geometrically_nonlinear=False, **kwargs):
        super().__init__(space, **kwargs)
        cpts = np.asarray(space.cpts)
        if space.dim != 2 or space.vec != 3 or cpts.shape[-1] != 3:
            raise ValueError(
                "KirchhoffLoveShell needs a surface patch in 3D (dim=2, 3D control points) "
                f"and a vector space with vec=3; got dim={space.dim}, vec={space.vec}, "
                f"control points in {cpts.shape[-1]}D"
            )
        if min(space.degree) < 2:
            raise ValueError(
                f"KirchhoffLoveShell needs degree >= 2 for C^1 continuity; got {space.degree}"
            )
        self.is_linear = not geometrically_nonlinear
        self._geometry = self._reference_geometry()

    # -- reference geometry at the Galerkin quadrature points ----------------

    def _reference_geometry(self):
        from jaxiga.methods._common import PointField
        from jaxiga.space.evaluation import evaluate
        from jaxiga.space.pointset import gauss

        basis = evaluate(self.space, gauss(self.space, self.quadrature), order=2)
        local = jnp.asarray(self.space.cpts)[np.asarray(basis.dofs)]  # (n_e, n_local, 3)
        a = jnp.einsum("eqdn,enc->eqdc", basis.dR, local)  # tangents a_1, a_2
        da = jnp.einsum("eqkn,enc->eqkc", basis.d2R, local)  # a_{1,1}, a_{1,2}, a_{2,2}
        return {"a": PointField(a), "da": PointField(da)}

    def parameters(self, E, nu, t, **extra):
        """``params`` for :func:`jaxiga.solve`: material, thickness and geometry."""
        return {"E": E, "nu": nu, "t": t, **self._geometry, **extra}

    # -- kinematics -----------------------------------------------------------

    @staticmethod
    def _covariant_strains(du, d2u, a, da):
        """Membrane and bending strains (covariant, Voigt 11, 22, 12) of u."""
        abar = a + du.T  # (2, 3)
        dabar = da + d2u.T  # (3, 3)
        n0 = _unit(jnp.cross(a[0], a[1]))
        n1 = _unit(jnp.cross(abar[0], abar[1]))
        g0, g1 = a @ a.T, abar @ abar.T
        E = 0.5 * jnp.array([g1[0, 0] - g0[0, 0], g1[1, 1] - g0[1, 1], g1[0, 1] - g0[0, 1]])
        b0, b1 = da @ n0, dabar @ n1  # Voigt (11, 12, 22)
        K = jnp.array([b0[0] - b1[0], b0[2] - b1[2], b0[1] - b1[1]])
        return E, K

    @staticmethod
    def _to_local(cov, a):
        """Covariant (11, 22, 12) tensor components -> local Cartesian Voigt (xx, yy, 2xy)."""
        n0 = jnp.cross(a[0], a[1])
        contra = jnp.linalg.solve(a @ a.T, a)  # rows a^1, a^2
        e1 = _unit(a[0])
        e2 = _unit(jnp.cross(n0, e1))
        T = jnp.stack([contra @ e1, contra @ e2])  # T[g, al] = e_g . a^al
        S = jnp.array([[cov[0], cov[2]], [cov[2], cov[1]]])
        C = T @ S @ T.T
        return jnp.array([C[0, 0], C[1, 1], 2.0 * C[0, 1]])

    def strains(self, grad_u, hess_u, params):
        """Local Cartesian membrane strain and curvature change, linear or exact."""
        a, da = params["a"], params["da"]

        def local(du, d2u):
            E, K = self._covariant_strains(du, d2u, a, da)
            return self._to_local(E, a), self._to_local(K, a)

        if self.is_linear:
            zero = (jnp.zeros_like(grad_u), jnp.zeros_like(hess_u))
            _, (eps, kappa) = jax.jvp(local, zero, (grad_u, hess_u))
            return eps, kappa
        return local(grad_u, hess_u)

    @staticmethod
    def material(params):
        nu = params["nu"]
        return params["E"] / (1.0 - nu**2) * jnp.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
        )

    def energy(self, grad_u, u, x, params, hess_u=None):
        eps, kappa = self.strains(grad_u, hess_u, params)
        D, t = self.material(params), params["t"]
        return 0.5 * t * eps @ D @ eps + 0.5 * t**3 / 12.0 * kappa @ D @ kappa
