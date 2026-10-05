"""Material models as pointwise, traced functions.

Replaces the NumPy ``MaterialElast2D`` / ``MaterialElast3D`` classes. Building
the constitutive matrix with ``jnp`` rather than ``np`` is what makes ``E`` and
``nu`` differentiable, which inverse problems and parameter identification
need.

Strains and stresses use engineering Voigt notation, matching the legacy
assembly: 2D ``[eps_xx, eps_yy, gamma_xy]`` with ``gamma_xy = 2 eps_xy``, and
3D ``[eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]``.
"""

from __future__ import annotations

import jax.numpy as jnp


def plane_stress(E, nu):
    """(3, 3) plane-stress constitutive matrix."""
    E = jnp.asarray(E, dtype=float)
    nu = jnp.asarray(nu, dtype=float)
    return (
        E
        / (1 - nu**2)
        * jnp.array(
            [
                [jnp.ones_like(nu), nu, jnp.zeros_like(nu)],
                [nu, jnp.ones_like(nu), jnp.zeros_like(nu)],
                [jnp.zeros_like(nu), jnp.zeros_like(nu), (1 - nu) / 2],
            ]
        )
    )


def plane_strain(E, nu):
    """(3, 3) plane-strain constitutive matrix."""
    E = jnp.asarray(E, dtype=float)
    nu = jnp.asarray(nu, dtype=float)
    zero = jnp.zeros_like(nu)
    return (
        E
        / ((1 + nu) * (1 - 2 * nu))
        * jnp.array(
            [
                [1 - nu, nu, zero],
                [nu, 1 - nu, zero],
                [zero, zero, (1 - 2 * nu) / 2],
            ]
        )
    )


def isotropic_3d(E, nu):
    """(6, 6) isotropic 3D constitutive matrix."""
    E = jnp.asarray(E, dtype=float)
    nu = jnp.asarray(nu, dtype=float)
    z = jnp.zeros_like(nu)
    s = (1 - 2 * nu) / 2
    return (
        E
        / ((1 + nu) * (1 - 2 * nu))
        * jnp.array(
            [
                [1 - nu, nu, nu, z, z, z],
                [nu, 1 - nu, nu, z, z, z],
                [nu, nu, 1 - nu, z, z, z],
                [z, z, z, s, z, z],
                [z, z, z, z, s, z],
                [z, z, z, z, z, s],
            ]
        )
    )


def lame(E, nu):
    """Lame parameters ``(lambda, mu)``."""
    E = jnp.asarray(E, dtype=float)
    nu = jnp.asarray(nu, dtype=float)
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def constitutive(E, nu, dim, plane="stress"):
    """Dispatch to the right constitutive matrix for the dimension."""
    if dim == 3:
        return isotropic_3d(E, nu)
    if dim == 2:
        return plane_stress(E, nu) if plane == "stress" else plane_strain(E, nu)
    raise ValueError(f"elasticity needs dim 2 or 3, got {dim}")


def strain_voigt(grad_u):
    """Engineering strain in Voigt form from the displacement gradient.

    ``grad_u[c, d] = du_c/dx_d``.
    """
    dim = grad_u.shape[1]
    if dim == 2:
        return jnp.array(
            [grad_u[0, 0], grad_u[1, 1], grad_u[0, 1] + grad_u[1, 0]]
        )
    if dim == 3:
        return jnp.array(
            [
                grad_u[0, 0],
                grad_u[1, 1],
                grad_u[2, 2],
                grad_u[1, 2] + grad_u[2, 1],
                grad_u[0, 2] + grad_u[2, 0],
                grad_u[0, 1] + grad_u[1, 0],
            ]
        )
    raise ValueError(f"unsupported dimension {dim}")


def von_mises(stress):
    """Von Mises equivalent stress from a Voigt stress vector."""
    if stress.shape[0] == 3:
        sxx, syy, sxy = stress
        return jnp.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    sxx, syy, szz, syz, sxz, sxy = stress
    return jnp.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3 * (syz**2 + sxz**2 + sxy**2)
    )
