"""Multipatch topology: interface detection and conforming dof numbering.

Setup stage, pure NumPy. Called only by the :class:`FunctionSpace`
constructor; users never invoke it directly.

Where the legacy ``multipatch.py`` matches patches through 2D-specific vertex
and edge casework, this module works from **geometric coincidence of control
points**, which is dimension-generic. But coincidence alone is not evidence of
an interface, so nothing is merged on that basis: candidate side pairs are
identified first and then *verified*, and only the control points of a verified
interface are glued.

Verification of a candidate pair of sides
-----------------------------------------
1. the two sides carry the same number of control points;
2. those points coincide as sets, within ``tol``;
3. the bijection between them is a legal reorientation of the side -- a flip of
   tangential directions, and in 3D possibly a transpose -- rather than an
   arbitrary shuffle;
4. the tangential knot vectors agree under that reorientation, so the two
   patches discretize the interface identically;
5. NURBS weights agree at matched points;
6. **the outward normals are antiparallel**, so the patches lie on opposite
   sides of the shared face.

Step 6 is what separates a genuine interface from two patches that overlap.
Coincident control points say only that two boundaries touch; on a shared face
between neighbours the outward normals oppose, whereas for two copies of the
same region they agree. Without this test a duplicated patch is silently merged
into one, halving the dof count and doubling the assembled stiffness.

Contact that is not a whole side -- patches meeting at a corner in 2D, or along
an edge in 3D -- is legitimate and is merged separately. Coincident points that
are neither part of a verified interface nor confined to such a low-dimensional
feature indicate overlapping or partially abutting patches, and raise.

By default control points are merged only *across* patches. Within a patch,
the opt-in ``collapse_degenerate`` setting ties repeated equal-weight controls
on whole boundary faces that collapse in a tangential direction. Other
coincident control points stay distinct, since they are distinct basis functions.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import jax
import numpy as np
from scipy.spatial import cKDTree

from jaxiga.config import TOL
from jaxiga.geometry.nurbs import Patch, side_name


class NonConformingError(Exception):
    """Two patches meet along an interface whose discretizations do not match."""


class OrientationError(Exception):
    """Two patches meet with inconsistent orientation."""


@dataclasses.dataclass(frozen=True)
class Interface:
    """A shared side between two patches."""

    patch_a: int
    side_a: str
    patch_b: int
    side_b: str
    reversed: bool = False

    def __str__(self):
        order = "reversed" if self.reversed else "aligned"
        return (
            f"patch {self.patch_a} side {self.side_a} <-> "
            f"patch {self.patch_b} side {self.side_b} ({order})"
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Topology:
    """Global scalar-dof numbering for a collection of patches.

    ``local_to_global`` holds arrays, so it is a pytree child; the interface
    list and dof count are hashable metadata.
    """

    local_to_global: tuple  # one (n_cp_patch,) int array per patch
    n_global_dofs: int = dataclasses.field(metadata=dict(static=True), default=0)
    interfaces: tuple = dataclasses.field(metadata=dict(static=True), default=())

    def patch_dofs(self, patch_index: int) -> np.ndarray:
        return self.local_to_global[patch_index]


def side_indices(patch: Patch, side: str, offset: int = 0) -> np.ndarray:
    """Local control-point indices lying on the given side.

    Ordering follows the tensor-product layout with the lowest remaining
    direction fastest, so the same side of two patches is directly comparable
    once orientation is accounted for.

    ``offset`` steps inward: ``offset=1`` gives the layer just behind the side,
    which is what clamping a C^1 discretization needs.
    """
    from jaxiga.geometry.simplex import SimplexPatch
    if isinstance(patch, SimplexPatch):
        if side not in {f"f{i}" for i in range(patch.dim + 1)}:
            raise ValueError("simplex sides are f0,...,fd")
        if not 0 <= offset <= patch.degree[0]:
            raise ValueError("simplex layer offset outside the degree")
        return np.flatnonzero(patch.multi_indices[:, int(side[1:])] == offset)
    direction = "uvw".index(side[0])
    end = int(side[1:])
    n_per_dir = patch.n_cp_per_dir

    if not 0 <= offset < n_per_dir[direction]:
        raise ValueError(
            f"offset {offset} is outside the {n_per_dir[direction]} control points "
            f"along direction {direction}"
        )
    fixed = offset if end == 0 else n_per_dir[direction] - 1 - offset
    strides = np.array([1, *np.cumprod(n_per_dir[:-1])], dtype=int)

    ranges = [np.arange(n) for n in n_per_dir]
    ranges[direction] = np.array([fixed])
    mesh = np.meshgrid(*ranges, indexing="ij")
    return sum(m.ravel(order="F") * strides[d] for d, m in enumerate(mesh)).astype(int)


def _side_grid_shape(patch: Patch, side: str) -> tuple:
    """Control-point counts along the tangential directions of a side."""
    direction = "uvw".index(side[0])
    return tuple(n for d, n in enumerate(patch.n_cp_per_dir) if d != direction)


def _orientation_maps(shape):
    """Every legal reindexing of a side's tangential control-point grid.

    Flips of each tangential direction, and in 3D the transpose as well. A
    bijection between two matching sides that is not one of these is not a
    reparametrization of the same face, so the patches are mismatched.
    """
    import itertools

    n_t = len(shape)
    base = np.arange(int(np.prod(shape)) if n_t else 1).reshape(shape, order="F")
    maps = []
    for perm in itertools.permutations(range(n_t)):
        if tuple(np.array(shape, dtype=int)[list(perm)]) != tuple(shape):
            continue  # a transpose only makes sense between equal extents
        for flips in itertools.product([False, True], repeat=n_t):
            g = np.transpose(base, perm)
            for d, flip in enumerate(flips):
                if flip:
                    g = np.flip(g, axis=d)
            maps.append(g.ravel(order="F"))
    return maps


def _outward_normal(patch: Patch, side: str, h: float = 1e-6) -> np.ndarray:
    """Outward unit normal at the centre of a patch side.

    Finite differences of the geometry map: this is setup-stage code that needs
    only a direction, and the map is smooth.
    """
    from jaxiga.geometry.nurbs import evaluate_patch

    direction = "uvw".index(side[0])
    end = int(side[1:])
    dim = patch.dim

    centre = np.full(dim, 0.5)
    centre[direction] = 0.0 if end == 0 else 1.0

    # step inward along the pinned parameter; the outward normal opposes it
    inner = centre.copy()
    inner[direction] = centre[direction] + (h if end == 0 else -h)
    pair = evaluate_patch(patch, np.stack([centre, inner]))
    normal = -(pair[1] - pair[0])

    # project out the tangential directions, so the result is a true normal
    for d in range(dim):
        if d == direction:
            continue
        lo, hi = centre.copy(), centre.copy()
        lo[d] = max(0.0, centre[d] - h)
        hi[d] = min(1.0, centre[d] + h)
        pts = evaluate_patch(patch, np.stack([lo, hi]))
        t = pts[1] - pts[0]
        norm_t = np.linalg.norm(t)
        if norm_t > 1e-30:
            t = t / norm_t
            normal = normal - np.dot(normal, t) * t

    length = np.linalg.norm(normal)
    return normal / length if length > 1e-30 else np.zeros_like(normal)


def _knots_match(kv_a, kv_b, flipped: bool, tol: float) -> bool:
    """Whether two knot vectors describe the same 1D discretization."""
    a = np.asarray(kv_a, dtype=float)
    b = np.asarray(kv_b, dtype=float)
    if len(a) != len(b):
        return False
    if flipped:
        b = np.sort(1.0 - b)
    return bool(np.max(np.abs(a - b)) <= max(tol, 1e-12))


@dataclasses.dataclass(frozen=True)
class _SideRecord:
    """One side of one patch, with its control-point indices in grid order."""

    patch: int
    name: str
    local: np.ndarray
    shape: tuple


def _enumerate_sides(patches):
    out = []
    for i, patch in enumerate(patches):
        for d in range(patch.dim):
            for e in (0, 1):
                name = side_name(d, e)
                out.append(
                    _SideRecord(
                        i, name, side_indices(patch, name), _side_grid_shape(patch, name)
                    )
                )
    return out


def _sample_side(patch: Patch, side: str, n: int = 7) -> np.ndarray:
    """Physical points on a patch side, on a uniform parametric grid."""
    from jaxiga.geometry.nurbs import evaluate_patch

    direction = "uvw".index(side[0])
    end = int(side[1:])
    dim = patch.dim

    axes = []
    for d in range(dim):
        if d == direction:
            axes.append(np.array([0.0 if end == 0 else 1.0]))
        else:
            axes.append(np.linspace(0.0, 1.0, n))
    mesh = np.meshgrid(*axes, indexing="ij")
    params = np.stack([m.ravel() for m in mesh], axis=1)
    return evaluate_patch(patch, params)


def _hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two point clouds."""
    return max(
        float(cKDTree(b).query(a, k=1)[0].max()),
        float(cKDTree(a).query(b, k=1)[0].max()),
    )


def _overlap_fraction(pts_a, pts_b, cutoff) -> float:
    """Fraction of samples of each face that lie on the other."""
    fa = float(np.mean(cKDTree(pts_b).query(pts_a, k=1)[0] <= cutoff))
    fb = float(np.mean(cKDTree(pts_a).query(pts_b, k=1)[0] <= cutoff))
    return max(fa, fb)


def _face_contact(pa, sa, pb, sb, tol, samples=None) -> str:
    """Classify how two patch sides touch: ``"none"``, ``"partial"`` or ``"full"``.

    Compared by sampling rather than by control points, because two patches can
    discretize the same interface with different control nets and that case must
    be *detected* in order to be rejected, not quietly treated as unconnected.

    Telling a partial face overlap from a legitimate corner or edge touch is a
    question about the *dimension* of the contact set, so it is answered by
    sampling at two densities: a contact of full face dimension keeps the same
    fraction of samples as the grid is refined, while contact along a corner or
    an edge loses half of its share each time the grid doubles.
    """
    pts_a, pts_b = samples if samples is not None else (
        _sample_side(pa, sa.name), _sample_side(pb, sb.name)
    )

    lo = np.minimum(pts_a.min(axis=0), pts_b.min(axis=0))
    hi = np.maximum(pts_a.max(axis=0), pts_b.max(axis=0))
    scale = float(np.max(hi - lo))
    cutoff = max(tol, 1e-12) * 10 + 1e-9 * max(scale, 1.0)

    # cheap bounding-box reject before any distance computation
    if np.any(pts_a.min(axis=0) > pts_b.max(axis=0) + cutoff) or np.any(
        pts_b.min(axis=0) > pts_a.max(axis=0) + cutoff
    ):
        return "none"

    if _hausdorff(pts_a, pts_b) <= cutoff:
        return "full"

    n_coarse, n_fine = 9, 17
    f_coarse = _overlap_fraction(
        _sample_side(pa, sa.name, n_coarse), _sample_side(pb, sb.name, n_coarse), cutoff
    )
    if f_coarse <= 1.5 / n_coarse:
        return "none"  # at most a single row of samples: a corner or an edge
    f_fine = _overlap_fraction(
        _sample_side(pa, sa.name, n_fine), _sample_side(pb, sb.name, n_fine), cutoff
    )
    # a lower-dimensional contact set loses roughly half its share of samples
    return "partial" if f_fine > 0.6 * f_coarse else "none"


def _match_points(pts_a, pts_b, tol):
    """Bijection ``perm`` with ``pts_a[i]`` at ``pts_b[perm[i]]``, or ``None``."""
    if pts_a.shape != pts_b.shape:
        return None
    radius = max(tol, 1e-12)
    dist, perm = cKDTree(pts_b).query(pts_a, k=1, distance_upper_bound=radius)
    if not np.all(np.isfinite(dist)):
        return None
    if len(set(perm.tolist())) != len(perm):
        return None
    return perm


def _verify_interface(patches, sa: "_SideRecord", sb: "_SideRecord", tol, samples=None):
    """Check a candidate side pair; return ``(Interface, perm)`` or ``None``.

    Candidacy is decided geometrically -- do the two faces occupy the same
    region of space? -- and only then are the discretizations compared. Once a
    pair is a candidate, a failure of any remaining check raises rather than
    returning ``None``: the sides plainly *are* meant to be an interface, and
    silently recording them as unconnected is how a torn mesh gets assembled and
    solved without complaint.
    """
    pa, pb = patches[sa.patch], patches[sb.patch]

    # Candidacy is geometric: do these two faces occupy the same region?
    contact = _face_contact(pa, sa, pb, sb, tol, samples)
    if contact == "none":
        return None
    if contact == "partial":
        raise NonConformingError(
            f"patch {sa.patch} side {sa.name} and patch {sb.patch} side {sb.name} "
            f"overlap over part of their area but are not the same face. Patches "
            f"must meet along whole matching sides; an interface covering only "
            f"part of a face would leave the rest of it uncoupled."
        )

    if pa.degree != pb.degree:
        raise NonConformingError(
            f"patch {sa.patch} side {sa.name} and patch {sb.patch} side {sb.name} "
            f"describe the same surface but the patches have different degrees "
            f"({pa.degree} and {pb.degree}); mixed degrees are not supported. "
            f"Elevate both patches to a common degree."
        )
    if sa.shape != sb.shape:
        raise NonConformingError(
            f"patch {sa.patch} side {sa.name} and patch {sb.patch} side {sb.name} "
            f"describe the same surface but carry {sa.shape} and {sb.shape} control "
            f"points respectively, so they discretize the interface differently. "
            f"Refine both patches identically along the shared side."
        )

    cpts_a = np.asarray(pa.ctrl_pts)[sa.local]
    cpts_b = np.asarray(pb.ctrl_pts)[sb.local]
    perm = _match_points(cpts_a, cpts_b, tol)
    if perm is None:
        raise NonConformingError(
            f"patch {sa.patch} side {sa.name} and patch {sb.patch} side {sb.name} "
            f"describe the same surface with the same number of control points, but "
            f"those points do not coincide, so the two parametrizations of the "
            f"interface differ. Refine both patches identically along the shared side."
        )

    legal = _orientation_maps(sa.shape)
    if not any(np.array_equal(perm, m) for m in legal):
        raise OrientationError(
            f"patch {sa.patch} side {sa.name} and patch {sb.patch} side {sb.name} "
            f"have coincident control nets, but the correspondence between them is "
            f"not a flip or transpose of the face; the patches meet with an "
            f"inconsistent parametrization"
        )
    flipped = not np.array_equal(perm, np.arange(len(perm)))

    wa = np.asarray(pa.weights)[sa.local]
    wb = np.asarray(pb.weights)[sb.local][perm]
    if np.max(np.abs(wa - wb)) > max(tol, 1e-12):
        raise NonConformingError(
            f"patch {sa.patch} side {sa.name} and patch {sb.patch} side {sb.name} "
            f"share a control net but assign different NURBS weights (largest "
            f"difference {np.max(np.abs(wa - wb)):.3e}); the patches are not conforming"
        )

    dir_a = "uvw".index(sa.name[0])
    dir_b = "uvw".index(sb.name[0])
    kv_a = [kv for d, kv in enumerate(pa.knot_arrays()) if d != dir_a]
    kv_b = [kv for d, kv in enumerate(pb.knot_arrays()) if d != dir_b]
    if not all(
        _knots_match(ka, kb, False, tol) or _knots_match(ka, kb, True, tol)
        for ka, kb in zip(kv_a, kv_b)
    ):
        raise NonConformingError(
            f"patch {sa.patch} side {sa.name} and patch {sb.patch} side {sb.name} "
            f"have coincident control points but different knot vectors along the "
            f"interface, so they discretize it differently. Refine both patches "
            f"identically along the shared side."
        )

    n_a = _outward_normal(pa, sa.name)
    n_b = _outward_normal(pb, sb.name)
    alignment = float(np.dot(n_a, n_b))
    if alignment > -0.5:
        raise NonConformingError(
            f"patch {sa.patch} side {sa.name} and patch {sb.patch} side {sb.name} "
            f"have coincident control nets, but their outward normals are not "
            f"opposed (n_a . n_b = {alignment:+.3f}); the patches overlap rather "
            f"than abutting along this face. Two patches sharing an interface lie "
            f"on opposite sides of it."
        )

    return Interface(sa.patch, sa.name, sb.patch, sb.name, reversed=flipped), perm


def _on_side_intersection(patch: Patch) -> np.ndarray:
    """Control points lying on at least two sides of a patch.

    The corners in 2D and the edges in 3D: the features along which patches may
    legitimately touch without sharing a whole face.
    """
    counts = np.zeros(patch.n_cp, dtype=int)
    for d in range(patch.dim):
        for e in (0, 1):
            counts[side_indices(patch, side_name(d, e))] += 1
    return counts >= 2


def compute_topology(patches: Sequence[Patch], tol: float = TOL, *, collapse_degenerate=False) -> Topology:
    """Build the conforming global dof numbering for a patch collection.

    Candidate side pairs are verified before any control point is merged; see
    the module docstring for the list of checks.
    ``collapse_degenerate=True`` also ties repeated, equal-weight coefficients
    on tensor boundary faces that collapse in a tangential direction.

    Raises
    ------
    NonConformingError
        If two patches meet along a side whose control nets, knot vectors or
        weights disagree; if their outward normals show them overlapping rather
        than abutting; or if coincident control points are explained neither by
        a verified interface nor by corner or edge contact.
    OrientationError
        If a shared side is traversed inconsistently by the two patches.
    """
    from jaxiga.geometry.simplex import SimplexPatch, simplex_topology
    patches = list(patches)
    if any(isinstance(p, SimplexPatch) for p in patches):
        if collapse_degenerate:
            raise NotImplementedError("collapsed boundaries currently require tensor patches")
        if not all(isinstance(p, SimplexPatch) for p in patches):
            from jaxiga.space.mixed import build_mixed_space
            return build_mixed_space(patches, 1).topology
        if len({p.degree for p in patches}) != 1:
            raise ValueError("simplex patches must share a dimension and degree")
        return simplex_topology(patches, tol)
    n_cp = [p.n_cp for p in patches]
    offsets = np.concatenate([[0], np.cumsum(n_cp)]).astype(int)
    all_cpts = np.concatenate([np.asarray(p.ctrl_pts) for p in patches], axis=0)
    all_wgts = np.concatenate([np.asarray(p.weights) for p in patches], axis=0)
    patch_of = np.concatenate([np.full(n, i) for i, n in enumerate(n_cp)])

    parent = np.arange(len(all_cpts))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    if collapse_degenerate:
        # Only identify a whole boundary face that is constant in one of its
        # tangential parameters. Arbitrary coincident controls remain distinct.
        # Identical weights are required so the collapsed trace is single-valued.
        for patch_index, patch in enumerate(patches):
            grid = np.arange(patch.n_cp).reshape(patch.n_cp_per_dir, order="F")
            H = np.asarray(patch.homogeneous())
            for d in range(patch.dim):
                for end in (0, -1):
                    face = np.take(grid, end, axis=d)
                    for tangent in range(face.ndim):
                        rows = np.moveaxis(face, tangent, -1).reshape(-1, face.shape[tangent])
                        if np.allclose(H[rows], H[rows[:, :1]], rtol=0, atol=tol):
                            for row in rows:
                                for j in row[1:]:
                                    union(int(offsets[patch_index] + row[0]), int(offsets[patch_index] + j))

    # ---- verified interfaces --------------------------------------------
    sides = _enumerate_sides(patches)
    # Mesh import can create hundreds of patches. Cache the same face samples
    # used by _face_contact and reject disjoint boxes before expensive checks.
    # The global tolerance is conservative relative to every pair's tolerance.
    samples = [_sample_side(patches[s.patch], s.name) for s in sides]
    lower = np.array([pts.min(axis=0) for pts in samples])
    upper = np.array([pts.max(axis=0) for pts in samples])
    cutoff = max(tol, 1e-12) * 10
    if sides:
        cutoff += 1e-9 * max(float((upper.max(axis=0) - lower.min(axis=0)).max()), 1.0)
    interfaces = []
    explained = set()
    for a in range(len(sides)):
        candidates = np.flatnonzero(
            np.all(lower <= upper[a] + cutoff, axis=1)
            & np.all(upper >= lower[a] - cutoff, axis=1)
        )
        for b in candidates[candidates > a]:
            sa, sb = sides[a], sides[b]
            if sa.patch == sb.patch:
                continue
            verified = _verify_interface(patches, sa, sb, tol, (samples[a], samples[b]))
            if verified is None:
                continue
            iface, perm = verified
            interfaces.append(iface)
            for k, local_a in enumerate(sa.local):
                gi = int(offsets[sa.patch] + local_a)
                gj = int(offsets[sb.patch] + sb.local[perm[k]])
                union(gi, gj)
                explained.add((min(gi, gj), max(gi, gj)))

    # ---- corner / edge contact, and anything left unexplained ------------
    on_feature = (
        np.concatenate([_on_side_intersection(p) for p in patches])
        if patches
        else np.zeros(0, dtype=bool)
    )
    if len(patches) > 1:
        for i, j in cKDTree(all_cpts).query_pairs(r=max(tol, 1e-12)):
            if patch_of[i] == patch_of[j]:
                continue  # never merge within a patch (degenerate edges)
            if (min(i, j), max(i, j)) in explained or find(i) == find(j):
                continue
            if on_feature[i] and on_feature[j]:
                if abs(all_wgts[i] - all_wgts[j]) > max(tol, 1e-12):
                    raise NonConformingError(
                        f"patches {patch_of[i]} and {patch_of[j]} touch at "
                        f"{all_cpts[i]} but assign that point different NURBS "
                        f"weights ({all_wgts[i]} vs {all_wgts[j]})"
                    )
                union(i, j)
                continue
            raise NonConformingError(
                f"patches {patch_of[i]} and {patch_of[j]} have a coincident control "
                f"point at {all_cpts[i]} lying in the interior of a face of at least "
                f"one of them, yet they share no verified interface. The patches "
                f"overlap, or abut along only part of a face; neither is supported. "
                f"Conforming patches must meet along whole matching sides, or touch "
                f"only at corners or edges."
            )

    roots = np.array([find(i) for i in range(len(all_cpts))])
    _, global_index = np.unique(roots, return_inverse=True)
    n_global = int(global_index.max()) + 1 if len(global_index) else 0

    local_to_global = tuple(
        global_index[offsets[i] : offsets[i + 1]].astype(int) for i in range(len(patches))
    )
    return Topology(
        local_to_global=local_to_global,
        n_global_dofs=n_global,
        interfaces=tuple(interfaces),
    )
