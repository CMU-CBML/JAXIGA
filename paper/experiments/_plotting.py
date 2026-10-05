"""Element-aware sampling and mesh outlines shared by the paper figures."""

import numpy as np


def slice_pointset(space, axis, value, n):
    """Points on the plane ``axis = value``, taken from element faces.

    Evaluating on a plane is far cheaper than probing scattered physical
    points: the plane coincides with element faces on a uniform mesh, so the
    reference coordinate along ``axis`` is known exactly and no geometry
    inversion is needed.
    """
    from jaxiga.space.pointset import PointSet

    ev = np.asarray(space.elem_vertex)
    dim = space.dim
    lo, hi = ev[:, axis], ev[:, dim + axis]

    sel = np.flatnonzero(np.abs(hi - value) < 1e-12)
    if sel.size:
        ref_axis = np.ones(sel.size)
    else:  # plane falls inside elements rather than on a face
        sel = np.flatnonzero((lo <= value + 1e-12) & (hi >= value - 1e-12))
        ref_axis = 2 * (value - lo[sel]) / (hi[sel] - lo[sel]) - 1
    if sel.size == 0:
        raise ValueError(f"no elements meet the plane {axis}={value}")

    others = [d for d in range(dim) if d != axis]
    grid = np.stack(
        np.meshgrid(*[np.linspace(-1.0, 1.0, n)] * len(others), indexing="ij"), -1
    ).reshape(-1, len(others))

    ref = np.zeros((sel.size, len(grid), dim))
    for k, d in enumerate(others):
        ref[:, :, d] = grid[:, k]
    ref[:, :, axis] = ref_axis[:, None]
    return PointSet(elems=sel, ref=ref, weights=None, tag="slice"), others


def element_triangulation(n_elems, n):
    """Triangulate a per-element sample grid without crossing the domain.

    ``matplotlib``'s default Delaunay triangulation of scattered points bridges
    holes and re-entrant corners. The element grid already knows the geometry,
    so triangles are built inside each element and never span a void.

    ``n`` is the number of samples per direction, points ordered with the first
    parametric direction fastest.
    """
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    c = lambda a, b: a + b * n  # noqa: E731
    lower = np.stack([c(i, j), c(i + 1, j), c(i + 1, j + 1)], axis=1)
    upper = np.stack([c(i, j), c(i + 1, j + 1), c(i, j + 1)], axis=1)
    cell = np.concatenate([lower, upper], axis=0)
    return np.concatenate([cell + e * n * n for e in range(n_elems)], axis=0)

def mesh_lines(space, n=9):
    """Element boundaries of a (possibly hierarchical) space in physical space.

    One geometry evaluation per patch: the parametric edge samples of every
    element of that patch are pushed through the patch map in a single call.
    """
    from jaxiga.geometry.nurbs import evaluate_patch

    ev, ep = np.asarray(space.elem_vertex), np.asarray(space.elem_patch)
    t, out = np.linspace(0.0, 1.0, n), []
    for i, patch in enumerate(space.patches):
        sel = np.flatnonzero(ep == i)
        if not sel.size:
            continue
        u0, v0, u1, v1 = ev[sel, 0], ev[sel, 1], ev[sel, 2], ev[sel, 3]
        one = np.ones_like(t)
        par = np.concatenate([
            np.stack([u0[:, None] + (u1 - u0)[:, None]*t, v0[:, None]*one], -1),
            np.stack([u0[:, None] + (u1 - u0)[:, None]*t, v1[:, None]*one], -1),
            np.stack([u0[:, None]*one, v0[:, None] + (v1 - v0)[:, None]*t], -1),
            np.stack([u1[:, None]*one, v0[:, None] + (v1 - v0)[:, None]*t], -1),
        ], axis=0)
        xy = evaluate_patch(patch, par.reshape(-1, 2)).reshape(par.shape[0], n, 2)
        out.extend(xy)
    return out


def box_slice_lines(space, box, axis, phys_value):
    """Element outlines on a plane ``axis = phys_value``, for a straight box mesh.

    The fracture examples all discretise a single axis-aligned rectangle or
    cuboid, so the reference-to-physical map is affine and decouples per axis:
    no geometry evaluation is needed, only rescaling ``elem_vertex`` -- which is
    parametric, in ``[0, 1]`` -- by the box bounds. Used for the 3D notched
    cube, where the crack plane is best read off a slice rather than a
    projection of the whole volume.
    """
    lower, upper = np.asarray(box[0], float), np.asarray(box[1], float)
    dim = space.dim
    p = (phys_value - lower[axis]) / (upper[axis] - lower[axis])
    ev = np.asarray(space.elem_vertex)
    lo, hi = ev[:, axis], ev[:, dim + axis]
    sel = np.flatnonzero((lo <= p + 1e-9) & (hi >= p - 1e-9))
    others = [d for d in range(dim) if d != axis]

    def to_phys(param, a):
        return lower[a] + param * (upper[a] - lower[a])

    segs = []
    for e in sel:
        a0, a1 = others
        x0, x1 = to_phys(ev[e, a0], a0), to_phys(ev[e, dim + a0], a0)
        y0, y1 = to_phys(ev[e, a1], a1), to_phys(ev[e, dim + a1], a1)
        segs += [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)],
                 [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]]
    return segs
