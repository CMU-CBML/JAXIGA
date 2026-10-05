"""Named geometry constructors.

Ports of ``utils_iga/Geom_examples.py``, returning :class:`Patch` objects with
sides pre-labeled so boundary conditions can be written against names rather
than parametric side codes.

The legacy classes specify control points in **weighted** form and divide by
the weights on construction (``Geometry2D.getUnweightedCpts``); the helpers
here follow the same convention, taking weighted input and storing unweighted
points, so the resulting patches are numerically identical to the legacy ones.

Legacy control points are laid out with the *second* direction fastest (geomdl
order); :func:`_from_geomdl_order` transposes them into this package's
first-direction-fastest convention.
"""

from __future__ import annotations

import numpy as np

from jaxiga.config import TOL
from jaxiga.geometry.nurbs import Patch

SQRT2_INV = 1.0 / np.sqrt(2.0)


def _from_geomdl_order(ctrlpts_weighted, weights, n_u, n_v, dim_phys=2):
    """Convert legacy (v-fastest, weighted) control data to this package's layout.

    Returns unweighted points ordered with u fastest, plus matching weights.
    """
    cw = np.asarray(ctrlpts_weighted, dtype=float)[:, :dim_phys]
    w = np.asarray(weights, dtype=float)

    # legacy flat index = i_u * n_v + i_v  ->  reshape (n_u, n_v), transpose, flatten
    cw = cw.reshape(n_u, n_v, dim_phys).transpose(1, 0, 2).reshape(n_u * n_v, dim_phys)
    w = w.reshape(n_u, n_v).transpose().reshape(n_u * n_v)
    return cw / w[:, None], w


def quadrilateral(corners) -> Patch:
    """Bilinear quadrilateral through four corners.

    Parameters
    ----------
    corners : (4, 2) array
        Corners **in counter-clockwise order** starting from the ``(u0, v0)``
        corner, i.e. ``[(0,0), (1,0), (1,1), (0,1)]`` for the unit square. This
        is the ordering used in the architecture guide's Poisson example, and
        the natural way to describe a quadrilateral; it is reordered internally
        into the tensor-product layout.

    Sides are labeled ``left``/``right``/``bottom``/``top``.
    """
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2):
        raise ValueError(f"expected 4 corners of dimension 2, got shape {corners.shape}")

    # counter-clockwise (A, B, C, D) -> tensor order (u0v0, u1v0, u0v1, u1v1)
    a, b, c, d = corners
    tensor_order = np.array([a, b, d, c])

    return Patch.create(
        knots=[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        degree=(1, 1),
        ctrl_pts=tensor_order,
        weights=np.ones(4),
        labels={"u0": "left", "u1": "right", "v0": "bottom", "v1": "top"},
    )


def rectangle(x0, y0, x1, y1) -> Patch:
    """Axis-aligned rectangle ``[x0, x1] x [y0, y1]``."""
    return quadrilateral([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def disk(center, radius) -> Patch:
    """Full disk as a single degree-(2,2) NURBS patch (9 control points)."""
    center = np.asarray(center, dtype=float)
    cpts_unit = np.array(
        [
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, 0.0],
        ]
    )
    weights = np.array([1.0, SQRT2_INV, 1.0, SQRT2_INV, 1.0, SQRT2_INV, 1.0, SQRT2_INV, 1.0])
    cpts = cpts_unit * radius + center
    cpts_weighted = cpts * weights[:, None]

    ctrl_pts, w = _from_geomdl_order(cpts_weighted, weights, 3, 3)
    return Patch.create(
        knots=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
        degree=(2, 2),
        ctrl_pts=ctrl_pts,
        weights=w,
        labels={"u0": "left", "u1": "right", "v0": "bottom", "v1": "top"},
    )


def quarter_annulus(radius_int, radius_ext) -> Patch:
    """Quarter annulus in the first quadrant.

    ``u`` runs radially (``u0`` = inner, ``u1`` = outer), ``v`` runs
    circumferentially from the x-axis to the y-axis.
    """
    cpts = np.array(
        [
            [radius_int, 0.0],
            [radius_int, radius_int],
            [0.0, radius_int],
            [radius_ext, 0.0],
            [radius_ext, radius_ext],
            [0.0, radius_ext],
        ]
    )
    weights = np.array([1.0, SQRT2_INV, 1.0, 1.0, SQRT2_INV, 1.0])
    cpts_weighted = cpts * weights[:, None]

    ctrl_pts, w = _from_geomdl_order(cpts_weighted, weights, 2, 3)
    return Patch.create(
        knots=[[0, 0, 1, 1], [0, 0, 0, 1, 1, 1]],
        degree=(1, 2),
        ctrl_pts=ctrl_pts,
        weights=w,
        labels={"u0": "inner", "u1": "outer", "v0": "sym_y", "v1": "sym_x"},
    )


# Control net of one plate-with-hole quadrant, in the legacy weighted form.
# u runs circumferentially (90 degrees, C^0 at the midpoint), v runs from the
# hole (v0) to the outer straight edges (v1).
_PWH_WEIGHTS = np.array(
    [1.0, 1.0, 0.853553390593274, 1.0, 0.853553390593274, 1.0, 0.853553390593274, 1.0, 1.0, 1.0]
)


def _plate_with_hole_net(rad_int, len_square):
    return np.array(
        [
            [-1.0 * rad_int, 0.0],
            [-1.0 * len_square, 0.0],
            [-0.853553390593274 * rad_int, 0.353553390593274 * rad_int],
            [-1.0 * len_square, 0.5 * len_square],
            [-0.603553390593274 * rad_int, 0.603553390593274 * rad_int],
            [-1.0 * len_square, 1.0 * len_square],
            [-0.353553390593274 * rad_int, 0.853553390593274 * rad_int],
            [-0.5 * len_square, 1.0 * len_square],
            [0.0, 1.0 * rad_int],
            [0.0, 1.0 * len_square],
        ]
    )


def plate_with_hole_quadrant(radius, side, quadrant) -> Patch:
    """One quadrant of a square plate with a central circular hole.

    Parameters
    ----------
    radius : float
        Hole radius.
    side : float
        Half-width of the full plate, i.e. the plate spans
        ``[-side, side]^2``.
    quadrant : int
        1..4, counter-clockwise from the first quadrant.

    Sides are labeled ``"hole"`` (v0) and ``"outer"`` (v1). The two radial cut
    edges lie on the coordinate axes; each is labeled ``"sym_x"`` if it lies on
    ``x = 0`` and ``"sym_y"`` if it lies on ``y = 0``, so symmetry conditions
    can be written by name regardless of which quadrant is used.
    """
    angles = {1: -np.pi / 2, 2: 0.0, 3: np.pi / 2, 4: np.pi}
    if quadrant not in angles:
        raise ValueError(f"quadrant must be 1, 2, 3 or 4; got {quadrant}")
    theta = angles[quadrant]

    net = _plate_with_hole_net(radius, side)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    net = net @ rot.T

    ctrl_pts, w = _from_geomdl_order(net, _PWH_WEIGHTS, 5, 2)

    patch = Patch.create(
        knots=[[0, 0, 0, 0.5, 0.5, 1, 1, 1], [0, 0, 1, 1]],
        degree=(2, 1),
        ctrl_pts=ctrl_pts,
        weights=w,
        labels={"v0": "hole", "v1": "outer"},
    )
    return patch.with_labels(**_axis_labels(patch))


def _axis_labels(patch: Patch) -> dict:
    """Label the u0/u1 edges by which coordinate axis they lie on."""
    n_u, n_v = patch.n_cp_per_dir
    cp = np.asarray(patch.ctrl_pts)
    labels = {}
    for side, i_u in (("u0", 0), ("u1", n_u - 1)):
        edge = cp[[i_u + j * n_u for j in range(n_v)]]
        if np.all(np.abs(edge[:, 0]) < TOL):
            labels[side] = "sym_x"
        elif np.all(np.abs(edge[:, 1]) < TOL):
            labels[side] = "sym_y"
    return labels


def plate_with_hole_sector(radius, side, sector) -> Patch:
    r"""One of four sectors of a square plate with a central circular hole, cut
    along the diagonals rather than along the axes.

    Parameters
    ----------
    radius : float
        Hole radius.
    side : float
        Half-width of the full plate, i.e. the plate spans ``[-side, side]^2``.
    sector : int
        0..3, counter-clockwise from the sector containing the positive x-axis.

    Sides are labeled ``"hole"`` (v0) and ``"outer"`` (v1); the two radial cut
    edges lie on the diagonals and are left unlabeled.

    Why this exists alongside :func:`plate_with_hole_quadrant`
    ---------------------------------------------------------
    The quadrant decomposition puts a corner of the square in the *middle* of
    each patch's outer edge, which forces a repeated interior knot -- and with
    it a $C^0$ line running diagonally across every patch. Cutting along the
    diagonals instead puts each corner at a patch corner, so the outer edge is
    one straight side and the hole edge is a single $90^\circ$ arc. Neither
    needs an interior knot: the knot vectors are ``[0,0,0,1,1,1]`` by
    ``[0,0,1,1]``, the coordinate axes become $C^{p-1}$ lines, and uniform
    refinement no longer carries twice as many elements circumferentially as
    radially. On the Kirsch benchmark that is worth about a factor of two and a
    half in degrees of freedom at a fixed energy-norm error.

    The trade-off is that the diagonals become patch interfaces, hence $C^0$
    lines, so four of them remain either way; and the points where the axes
    meet the hole are no longer control points, so a symmetry or point
    constraint written there has to be expressed differently.
    """
    if sector not in (0, 1, 2, 3):
        raise ValueError(f"sector must be 0, 1, 2 or 3; got {sector}")

    # 90-degree arc from 135 to 225 degrees, opposite the side x = -side; the
    # middle control point is the intersection of the end tangents, at
    # radius/cos(45 deg), and carries the weight cos(45 deg).
    c = radius * SQRT2_INV
    inner = np.array([[-c, c], [-radius * np.sqrt(2.0), 0.0], [-c, -c]])
    outer = np.array([[-side, side], [-side, 0.0], [-side, -side]])

    theta = sector * np.pi / 2 + np.pi  # sector 0 contains the +x axis
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    ctrl_pts = np.concatenate([inner, outer]) @ rot.T

    return Patch.create(
        knots=[[0, 0, 0, 1, 1, 1], [0, 0, 1, 1]],
        degree=(2, 1),
        ctrl_pts=ctrl_pts,
        weights=np.array([1.0, SQRT2_INV, 1.0, 1.0, SQRT2_INV, 1.0]),
        labels={"v0": "hole", "v1": "outer"},
    )


def plate_with_hole(radius, side) -> list:
    """Full square plate with a central circular hole, as four patches."""
    return [plate_with_hole_quadrant(radius, side, q) for q in (2, 3, 4, 1)]


def plate_with_hole_diagonal(radius, side) -> list:
    """Full square plate with a central circular hole, cut along the diagonals.

    The four-patch alternative to :func:`plate_with_hole` with no interior
    $C^0$ line; see :func:`plate_with_hole_sector`.
    """
    return [plate_with_hole_sector(radius, side, k) for k in range(4)]


def cuboid(lower, upper) -> Patch:
    """Trilinear axis-aligned box."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("lower and upper must each have 3 components")

    # first direction fastest
    corners = np.array(
        [
            [lower[0] if i == 0 else upper[0], lower[1] if j == 0 else upper[1],
             lower[2] if k == 0 else upper[2]]
            for k in (0, 1)
            for j in (0, 1)
            for i in (0, 1)
        ]
    )
    return Patch.create(
        knots=[[0, 0, 1, 1]] * 3,
        degree=(1, 1, 1),
        ctrl_pts=corners,
        weights=np.ones(8),
        labels={"u0": "left", "u1": "right", "v0": "bottom", "v1": "top",
                "w0": "back", "w1": "front"},
    )


def interval(x0=0.0, x1=1.0) -> Patch:
    """Linear 1D segment ``[x0, x1]``."""
    return Patch.create(
        knots=[[0.0, 0.0, 1.0, 1.0]],
        degree=(1,),
        ctrl_pts=np.array([[x0], [x1]], dtype=float),
        weights=np.ones(2),
        labels={"u0": "left", "u1": "right"},
    )


def graded_knots(n_fine, n_coarse, centre=0.5, half_width=0.15):
    """Interior knots clustered around ``centre``.

    Returns knots for :meth:`Patch.insert_knots`: ``n_fine`` uniform spans
    inside ``[centre - half_width, centre + half_width]`` and ``n_coarse``
    spans on either side. Knot insertion is exact and unconstrained in IGA, so
    resolution can be concentrated where a solution feature is expected --- a
    crack path, a boundary layer --- without refining the whole domain.

    Parameters
    ----------
    n_fine : int
        Spans inside the refined band.
    n_coarse : int
        Spans on each side of the band.
    centre, half_width : float
        Band location and half-extent in the parametric coordinate.
    """
    if not 0.0 < half_width < 0.5:
        raise ValueError(f"half_width must lie in (0, 0.5), got {half_width}")
    lo, hi = centre - half_width, centre + half_width
    if lo <= 0.0 or hi >= 1.0:
        raise ValueError(
            f"band [{lo}, {hi}] must lie strictly inside (0, 1); reduce "
            f"half_width or move centre"
        )

    knots = np.unique(
        np.concatenate([
            np.linspace(0.0, lo, n_coarse + 1)[:-1],
            np.linspace(lo, hi, n_fine + 1),
            np.linspace(hi, 1.0, n_coarse + 1)[1:],
        ])
    )
    return knots[(knots > TOL) & (knots < 1.0 - TOL)]


def triangle(vertices, labels=None):
    """Affine triangle through three positively oriented vertices; faces f0..f2."""
    from jaxiga.geometry.simplex import SimplexPatch
    if np.asarray(vertices).shape != (3, 2):
        raise ValueError("triangle vertices must have shape (3, 2)")
    return SimplexPatch.from_vertices(vertices, labels=labels)


def tetrahedron(vertices, labels=None):
    """Affine tetrahedron through four positively oriented vertices; faces f0..f3."""
    from jaxiga.geometry.simplex import SimplexPatch
    if np.asarray(vertices).shape != (4, 3):
        raise ValueError("tetrahedron vertices must have shape (4, 3)")
    return SimplexPatch.from_vertices(vertices, labels=labels)
