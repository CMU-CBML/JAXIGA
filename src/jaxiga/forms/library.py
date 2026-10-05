"""Canned problems.

These double as reference implementations of the kernel interface: each is a
few lines defining an energy density, and every method plus every derived
field follows from it by differentiation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxiga.forms import materials
from jaxiga.forms.problem import Problem


def _strain_displacement(dR):
    """Engineering-Voigt strain-displacement matrices at quadrature points."""
    n_q, dim, n_local = dR.shape
    n_strain = 3 if dim == 2 else 6
    B = jnp.zeros((n_q, n_strain, n_local, dim), dtype=dR.dtype)
    if dim == 2:
        B = B.at[:, 0, :, 0].set(dR[:, 0])
        B = B.at[:, 1, :, 1].set(dR[:, 1])
        B = B.at[:, 2, :, 0].set(dR[:, 1])
        B = B.at[:, 2, :, 1].set(dR[:, 0])
    else:
        B = B.at[:, 0, :, 0].set(dR[:, 0])
        B = B.at[:, 1, :, 1].set(dR[:, 1])
        B = B.at[:, 2, :, 2].set(dR[:, 2])
        B = B.at[:, 3, :, 1].set(dR[:, 2])
        B = B.at[:, 3, :, 2].set(dR[:, 1])
        B = B.at[:, 4, :, 0].set(dR[:, 2])
        B = B.at[:, 4, :, 2].set(dR[:, 0])
        B = B.at[:, 5, :, 0].set(dR[:, 1])
        B = B.at[:, 5, :, 1].set(dR[:, 0])
    return B.reshape(n_q, n_strain, n_local * dim)


class Poisson(Problem):
    r"""``-div(a0 grad u) = f`` from ``psi = 1/2 a0 |grad u|^2``.

    ``params`` may supply ``a0`` (default 1), as a scalar or a callable
    ``a0(x, params)``.
    """

    is_linear = True
    is_positive_definite = True

    def energy(self, grad_u, u, x, params):
        a0 = _coefficient(params, "a0", 1.0, x)
        return 0.5 * a0 * jnp.sum(grad_u**2)

    def element_tangent(self, R, dR, x, w, params, point_axes=None, d2R=None):
        """Direct ``int a0 grad(N_i).grad(N_j)`` element kernel."""
        a0 = jax.vmap(
            lambda xx, pp: _coefficient(pp, "a0", 1.0, xx),
            in_axes=(0, point_axes),
        )(x, params)
        scalar = jnp.einsum("q,qdi,qdj->ij", w * a0, dR, dR)
        eye = jnp.eye(self.space.vec, dtype=scalar.dtype)
        return jnp.einsum("ij,cd->icjd", scalar, eye).reshape(
            self.space.n_local * self.space.vec,
            self.space.n_local * self.space.vec,
        )


class KirchhoffPlate(Problem):
    r"""Kirchhoff-Love plate bending: ``psi = 1/2 kappa^T D kappa``.

    The unknown is the transverse deflection ``w`` (``vec=1``) and the
    curvature is ``kappa = -(w_xx, w_yy, 2 w_xy)``, so the energy depends on the
    *second* derivatives of the solution. That makes this a fourth-order problem
    needing C^1 continuity, which a maximal-smoothness spline of degree ``p >= 2``
    provides inside a patch for free -- the capability that separates IGA from
    standard C^0 finite elements.

    ``params`` supplies ``E``, ``nu`` and the thickness ``t``.

    Boundary conditions
    -------------------
    A simply supported edge is the ordinary ``DirichletBC(0.0, where=...)``: the
    zero-moment condition is natural for this energy. A clamped edge additionally
    fixes the slope and needs :class:`~jaxiga.forms.problem.ClampedBC`.

    Multipatch C^1 coupling is out of scope: patches meet with C^0 continuity,
    so a patch interface behaves as a *hinge*. Use a single patch unless a hinge
    is what you mean.
    """

    is_linear = True
    is_positive_definite = True
    needs_hessian = True

    def __init__(self, space, **kwargs):
        super().__init__(space, **kwargs)
        if space.dim != 2 or space.vec != 1:
            raise ValueError(
                f"KirchhoffPlate needs a 2D scalar space (dim=2, vec=1); got "
                f"dim={space.dim}, vec={space.vec}"
            )
        if min(space.degree) < 2:
            raise ValueError(
                f"KirchhoffPlate needs degree >= 2 for C^1 continuity; got "
                f"{space.degree}. Elevate the patch first."
            )

    def bending_stiffness(self, params):
        E, nu, t = params["E"], params["nu"], params["t"]
        factor = E * t**3 / (12.0 * (1.0 - nu**2))
        return factor * jnp.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
        )

    @staticmethod
    def curvature(hess_u):
        """``kappa = -(w_xx, w_yy, 2 w_xy)`` from Voigt ``(uu, uv, vv)``."""
        h = hess_u[0]
        return -jnp.array([h[0], h[2], 2.0 * h[1]])

    def energy(self, grad_u, u, x, params, hess_u=None):
        kappa = self.curvature(hess_u)
        return 0.5 * kappa @ self.bending_stiffness(params) @ kappa

    def element_tangent(self, R, dR, x, w, params, point_axes=None, d2R=None):
        """Direct Kirchhoff ``int B_kappa^T D B_kappa`` element kernel."""
        D = jax.vmap(
            lambda xx, pp: self.bending_stiffness(pp),
            in_axes=(0, point_axes),
        )(x, params)
        B = -jnp.stack([d2R[:, 0], d2R[:, 2], 2.0 * d2R[:, 1]], axis=1)
        return jnp.einsum("q,qsi,qst,qtj->ij", w, B, D, B)

    # -- derived fields ----------------------------------------------------

    def moments(self, grad_u, u, x, params, hess_u=None):
        """Bending moments ``(M_xx, M_yy, M_xy)``."""
        return self.bending_stiffness(params) @ self.curvature(hess_u)

    @property
    def fields(self):
        return {"moments": self.moments, "curvature": lambda g, u, x, p, hess_u=None:
                self.curvature(hess_u)}


class LinearElasticity(Problem):
    r"""Small-strain elasticity from ``psi = 1/2 eps : C : eps``.

    ``params`` supplies ``E`` and ``nu``. ``plane`` selects plane stress or
    plane strain in 2D and is ignored in 3D.
    """

    is_linear = True
    is_positive_definite = True

    def __init__(self, space, *, plane: str = "stress", **kwargs):
        super().__init__(space, **kwargs)
        if plane not in ("stress", "strain"):
            raise ValueError(f"plane must be 'stress' or 'strain', got {plane!r}")
        self.plane = plane
        if space.vec != space.dim:
            raise ValueError(
                f"elasticity needs vec == dim; got vec={space.vec}, dim={space.dim}. "
                f"Build the space with FunctionSpace(patches, vec={space.dim})."
            )

    def C(self, params):
        return materials.constitutive(params["E"], params["nu"], self.space.dim, self.plane)

    def energy(self, grad_u, u, x, params):
        eps = materials.strain_voigt(grad_u)
        return 0.5 * eps @ self.C(params) @ eps

    def element_tangent(self, R, dR, x, w, params, point_axes=None, d2R=None):
        """Direct small-strain ``int B^T C B`` element kernel."""
        B = _strain_displacement(dR)
        C = jax.vmap(
            lambda xx, pp: self.C(pp),
            in_axes=(0, point_axes),
        )(x, params)
        return jnp.einsum("q,qsi,qst,qtj->ij", w, B, C, B)

    # -- derived fields ----------------------------------------------------

    def stress(self, grad_u, u, x, params):
        """Voigt stress as ``d psi / d eps``, so it tracks any material override."""
        eps = materials.strain_voigt(grad_u)
        return jax.grad(lambda e: 0.5 * e @ self.C(params) @ e)(eps)

    def strain(self, grad_u, u, x, params):
        return materials.strain_voigt(grad_u)

    def von_mises(self, grad_u, u, x, params):
        return materials.von_mises(self.stress(grad_u, u, x, params))

    @property
    def fields(self):
        return {
            "stress": self.stress,
            "strain": self.strain,
            "von_mises": self.von_mises,
            "displacement": lambda grad_u, u, x, params: u,
        }


class Hyperelasticity(Problem):
    """Compressible neo-Hookean solid; demonstrates the nonlinear path.

    ``params`` supplies ``E`` and ``nu`` (converted to Lame parameters), or
    ``mu`` and ``lam`` directly.
    """

    is_linear = False

    def energy(self, grad_u, u, x, params):
        dim = grad_u.shape[0]
        if "mu" in params:
            mu, lam = params["mu"], params["lam"]
        else:
            lam, mu = materials.lame(params["E"], params["nu"])

        F = jnp.eye(dim) + grad_u
        J = jnp.linalg.det(F)
        I1 = jnp.trace(F.T @ F)
        return mu / 2 * (I1 - dim) - mu * jnp.log(J) + lam / 2 * jnp.log(J) ** 2


def _coefficient(params, name, default, x):
    """Look up a coefficient that may be absent, constant, or a function of x."""
    if params is None or name not in params:
        return default
    value = params[name]
    return value(x, params) if callable(value) else value


# --------------------------------------------------------------------------
# Phase-field fracture
# --------------------------------------------------------------------------


def principal_strains(strain, eps: float = 1e-20, degenerate: float = 1e-12):
    """Principal strains from an engineering Voigt strain vector, 2D or 3D.

    Closed form in both cases -- a 3x3 eigensolver called once per quadrature
    point would dominate the cost of a phase-field run, and its derivative is
    harder to keep finite than the trigonometric formula's.

    2D takes ``[e_xx, e_yy, gamma_xy]`` and returns two values; 3D takes
    ``[e_xx, e_yy, e_zz, gamma_yz, gamma_xz, gamma_xy]`` and returns three,
    largest first. Engineering shears are halved on the way in, since the
    eigenvalues belong to the strain *tensor*.

    Two regularisations, both of which perturb the result far below any strain
    of interest and both of which exist to keep derivatives finite:

    ``eps`` guards the deviatoric magnitude, which vanishes under hydrostatic
    strain -- including the zero strain a simulation starts from. It carries
    units of strain squared and shifts eigenvalues by about its square root, so
    the default moves them by 1e-10.

    ``degenerate`` keeps the 3D formula's ``arccos`` off its endpoints, which is
    where two principal strains coincide. That is not an exotic state: uniaxial
    strain ``diag(a, b, b)`` sits exactly on it, so it happens constantly. The
    energies built from these values are symmetric functions of them and so are
    perfectly smooth across such a crossing -- it is only this particular
    formula that is not, and clipping bounds its derivative rather than letting
    it return NaN.
    """
    n = strain.shape[0]
    if n == 3:
        exx, eyy, gxy = strain[0], strain[1], strain[2]
        radius = jnp.sqrt((exx - eyy) ** 2 + gxy**2 + eps)
        mean = 0.5 * (exx + eyy)
        return jnp.stack([mean + 0.5 * radius, mean - 0.5 * radius])
    if n == 6:
        exx, eyy, ezz = strain[0], strain[1], strain[2]
        eyz, exz, exy = 0.5 * strain[3], 0.5 * strain[4], 0.5 * strain[5]

        # Trigonometric solution of the characteristic polynomial for a
        # symmetric 3x3 (Smith, 1961): shift by the mean, normalise by the
        # deviatoric magnitude, and read the three roots off one arccos.
        q = (exx + eyy + ezz) / 3.0
        dxx, dyy, dzz = exx - q, eyy - q, ezz - q
        p2 = dxx**2 + dyy**2 + dzz**2 + 2.0 * (exy**2 + exz**2 + eyz**2)
        p = jnp.sqrt(p2 / 6.0 + eps)

        det = (
            dxx * (dyy * dzz - eyz**2)
            - exy * (exy * dzz - eyz * exz)
            + exz * (exy * eyz - dyy * exz)
        )
        r = jnp.clip(det / (2.0 * p**3), -1.0 + degenerate, 1.0 - degenerate)
        phi = jnp.arccos(r) / 3.0

        e1 = q + 2.0 * p * jnp.cos(phi)
        e3 = q + 2.0 * p * jnp.cos(phi + 2.0 * jnp.pi / 3.0)
        return jnp.stack([e1, 3.0 * q - e1 - e3, e3])
    raise ValueError(
        f"a Voigt strain vector has 3 entries in 2D or 6 in 3D, got {n}"
    )


def spectral_split(strain, lam, mu, eps: float = 1e-20):
    r"""Miehe tension/compression split of the strain energy density.

    ``psi_pm = lam/2 <tr eps>_pm^2 + mu sum_i <eps_i>_pm^2``, where ``eps_i``
    are the principal strains. Returns ``(psi_plus, psi_minus)``. Only the
    tensile part drives damage, so that cracks do not form under compression.

    Works in 2D and 3D; ``strain`` is the engineering Voigt vector for either.
    In 2D the trace is taken over the in-plane strains only, so under plane
    stress the out-of-plane contraction does not enter the driving force.

    The kinks in the positive and negative parts are the model's own and are
    not smoothed; see :func:`principal_strains` for what ``eps`` does.

    Note that the AT2 damage subproblem only ever *evaluates* this, through the
    history field, so a staggered solve never differentiates it. The
    regularisations matter for a monolithic formulation.
    """
    e = principal_strains(strain, eps)
    tr = jnp.sum(e)

    pos = lambda v: jnp.maximum(v, 0.0)  # noqa: E731
    neg = lambda v: jnp.minimum(v, 0.0)  # noqa: E731

    psi_p = 0.5 * lam * pos(tr) ** 2 + mu * jnp.sum(pos(e) ** 2)
    psi_m = 0.5 * lam * neg(tr) ** 2 + mu * jnp.sum(neg(e) ** 2)
    return psi_p, psi_m


class PhaseFieldDisplacement(Problem):
    r"""Elasticity with stiffness degraded by a damage field.

    ``psi = ((1-phi)^2 + k) * 1/2 eps:C:eps``

    Degradation is **isotropic**: the whole stiffness is scaled, and the
    tension/compression split enters only the driving force via
    :func:`tensile_energy`. This is the variant the legacy
    ``pde_form_elast_pf_2d`` implements, and it is what makes the displacement
    subproblem linear once ``phi`` is frozen -- which is the entire point of
    staggering. Degrading only the tensile part instead would couple the split
    into the tangent and make this subproblem nonlinear.

    ``params`` supplies ``E``, ``nu``, ``plane`` (via the constructor), the
    damage ``phi`` as a :class:`~jaxiga.methods._common.PointField`, and
    optionally the residual stiffness ``k`` (default 1e-8, which keeps the
    tangent non-singular inside a fully broken region).

    One half of the staggered scheme; see :class:`PhaseFieldDamage`.
    """

    is_linear = True
    is_positive_definite = True

    def __init__(self, space, *, plane: str = "strain", **kwargs):
        super().__init__(space, **kwargs)
        self.plane = plane

    def energy(self, grad_u, u, x, params):
        C = materials.constitutive(params["E"], params["nu"], self.space.dim, self.plane)
        eps = materials.strain_voigt(grad_u)
        phi = params["phi"]
        k = params.get("k", 1e-8)
        return ((1.0 - phi) ** 2 + k) * 0.5 * eps @ C @ eps

    def element_tangent(self, R, dR, x, w, params, point_axes=None, d2R=None):
        """Direct degraded-elasticity ``int g(phi) B^T C B`` kernel."""
        B = _strain_displacement(dR)

        def material(xx, pp):
            C = materials.constitutive(pp["E"], pp["nu"], self.space.dim, self.plane)
            return ((1.0 - pp["phi"]) ** 2 + pp.get("k", 1e-8)) * C

        C = jax.vmap(material, in_axes=(0, point_axes))(x, params)
        return jnp.einsum("q,qsi,qst,qtj->ij", w, B, C, B)

    @property
    def fields(self):
        return {"displacement": lambda grad_u, u, x, p: u}


class PhaseFieldDamage(Problem):
    r"""AT2 damage evolution at frozen displacement.

    ``psi = Gc (phi^2 / (2 l) + (l/2) |grad phi|^2) + (1-phi)^2 H``

    ``params`` supplies ``Gc``, ``l`` and the history field ``H`` (a
    :class:`~jaxiga.methods._common.PointField`). ``H`` is the running maximum
    of the tensile strain energy, which is what makes damage irreversible.

    The space must be scalar (``vec=1``).
    """

    is_linear = True
    is_positive_definite = True

    def energy(self, grad_u, u, x, params):
        Gc, ell = params["Gc"], params["l"]
        phi = u[0]
        grad_phi = grad_u[0]
        H = params["H"]

        return Gc * (phi**2 / (2 * ell) + 0.5 * ell * jnp.sum(grad_phi**2)) + (
            1.0 - phi
        ) ** 2 * H

    def element_tangent(self, R, dR, x, w, params, point_axes=None, d2R=None):
        """Direct AT2 reaction-diffusion element kernel."""
        coefficients = jax.vmap(
            lambda xx, pp: (pp["Gc"] * pp["l"], pp["Gc"] / pp["l"] + 2 * pp["H"]),
            in_axes=(0, point_axes),
        )(x, params)
        diffusion, reaction = coefficients
        return jnp.einsum("q,qdi,qdj->ij", w * diffusion, dR, dR) + jnp.einsum(
            "q,qi,qj->ij", w * reaction, R, R
        )

    @property
    def fields(self):
        return {"damage": lambda grad_u, u, x, p: u}


def tensile_energy(sol_u, params, basis):
    """Tensile strain energy density at quadrature points, ``(n_elems, n_q)``.

    Feeds the history field of the staggered scheme.
    """
    lam, mu = materials.lame(params["E"], params["nu"])
    grad = sol_u.grad(basis=basis)

    def one(g):
        return spectral_split(materials.strain_voigt(g), lam, mu)[0]

    return jax.vmap(jax.vmap(one))(grad)


def initial_history(basis, crack_fn, B, ell, Gc):
    """Seed history field representing a pre-existing crack.

    ``crack_fn(x)`` returns the distance from a point to the crack. Points
    within ``l/2`` get a large history value, which forces damage there on the
    first solve. Port of ``history_edge_crack`` / ``history_plate_w_3holes``.
    """
    x = basis.x
    dist = jax.vmap(jax.vmap(crack_fn))(x)
    return jnp.where(dist <= ell / 2, B * Gc * (1 - dist / (ell / 2)) / (2 * ell), 0.0)


def edge_crack_distance(tip=(0.5, 0.5), axes=(0, 1)):
    """Distance to a straight edge crack running in from one side.

    ``tip`` is where the crack stops, in the plane spanned by ``axes``: ahead of
    it the distance is radial, behind it the perpendicular distance to the crack
    line. Feeding this to :func:`initial_history` seeds a pre-existing crack.

    ``axes`` selects which two coordinates the crack lies in, so the same
    function describes a 2D edge crack and a 3D crack that runs through the
    thickness -- ``axes=(0, 2)`` puts it in the x-z plane, unbroken in y, which
    is the through-crack of the standard notched-cube benchmark.
    """
    i, j = axes

    def dist(x):
        return jnp.where(
            x[i] > tip[0],
            jnp.sqrt((x[i] - tip[0]) ** 2 + (x[j] - tip[1]) ** 2),
            jnp.abs(x[j] - tip[1]),
        )

    return dist
