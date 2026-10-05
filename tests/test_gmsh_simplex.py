"""Direct simplex import, high-order geometry, CAD labels and failure modes."""
from pathlib import Path

import numpy as np
import pytest

import jaxiga as jx
from jaxiga.geometry.gmsh import GmshImportError
from jaxiga.geometry.nurbs import evaluate_patch


gmsh = pytest.importorskip('gmsh')


def msh(path, xyz, elements, names=()):
    lines = ['$MeshFormat', '2.2 0 8', '$EndMeshFormat']
    if names:
        lines += ['$PhysicalNames', str(len(names))]
        lines += [f'{d} {tag} "{name}"' for d, tag, name in names]
        lines += ['$EndPhysicalNames']
    lines += ['$Nodes', str(len(xyz))]
    lines += [f'{i+1} ' + ' '.join(map(str, x)) for i, x in enumerate(xyz)]
    lines += ['$EndNodes', '$Elements', str(len(elements))]
    lines += [f'{i+1} {kind} 2 {physical} 1 ' + ' '.join(str(n+1) for n in nodes)
              for i, (kind, physical, nodes) in enumerate(elements)]
    lines += ['$EndElements']
    path.write_text('\n'.join(lines) + '\n')
    return path


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('degree', [1, 2, 3, 4])
@pytest.mark.parametrize('reverse', [False, True])
def test_polynomial_geometry_preserved(tmp_path, dim, degree, reverse):
    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        kind = gmsh.model.mesh.getElementType('Triangle' if dim == 2 else 'Tetrahedron', degree)
        _, _, _, count, ref, _ = gmsh.model.mesh.getElementProperties(kind)
        ref = np.asarray(ref).reshape(count, dim)
        xyz = np.zeros((count, 3))
        xyz[:, :dim] = ref
        xyz[:, 1] += .15*ref[:, 0]*(1-ref.sum(axis=1))
        if dim == 3:
            xyz[:, 2] += .08*ref[:, 0]*ref[:, 1]
        if reverse:
            xyz[:, 0] *= -1
        bary = np.random.default_rng(13).dirichlet(np.ones(dim+1), 23)
        points = bary[:, 1:].copy()
        if reverse:
            bary[:, [0, 1]] = bary[:, [1, 0]]
        query = np.zeros((len(bary), 3))
        query[:, :dim] = bary[:, 1:]
        _, B, _ = gmsh.model.mesh.getBasisFunctions(kind, query.ravel(), 'Lagrange')
        expected = np.asarray(B).reshape(-1, count) @ xyz[:, :dim]
    finally:
        gmsh.finalize()
    path = msh(tmp_path/'polynomial.msh', xyz,
               [(1 if dim == 2 else 2, 2, list(range(1, dim+1))),
                (kind, 3, list(range(count)))], [(dim-1, 2, 'boundary'), (dim, 3, 'domain')])
    mesh = jx.read_gmsh(path, cell_type='simplex')
    assert mesh.cell_type == 'simplex' and not mesh.subdivided
    assert mesh[0].degree == (degree,)*dim
    np.testing.assert_allclose(evaluate_patch(mesh[0], points), expected, atol=3e-13)
    elevated = mesh.elevate(degree+1, to_cad=False)
    np.testing.assert_allclose(evaluate_patch(elevated[0], points), expected, atol=3e-13)
    assert mesh.boundary_groups == {'boundary': ((0, 'f1' if reverse else 'f0'),)}
    assert mesh.cell_groups == {'domain': (0,)}
    assert mesh.min_scaled_jacobian > 0 and mesh.jacobian_lower_bounds[0] > 0
    with pytest.raises(NotImplementedError, match='CAD refitting'):
        mesh.elevate(degree+1)
    assert not gmsh.isInitialized()


def test_plate_curves_labels_and_auto_sizing():
    path = Path(jx.__file__).parent/'examples/geometries/perforated_plate.geo'
    mesh = jx.read_gmsh(path, cell_type='simplex', mesh_size='auto', geometry_order=2)
    assert not mesh.subdivided
    assert mesh.meshing_method == 'simplex'
    assert mesh.geometry_error < .012
    assert mesh.sizing_history
    space = jx.FunctionSpace(mesh)
    assert set(space.boundaries) == {'fixed', 'loaded', 'holes', 'free'}
    area = 72 - np.pi*(1+.8**2+.4**2) - 2*.8
    assert float(jx.integrate(lambda x: 1., space)) == pytest.approx(area, abs=.003)
    assert len(mesh.cell_groups['plate']) == len(mesh)
    # Euler characteristic: one component minus three holes.
    vertices = len(set(mesh._corner_tags.ravel()))
    edges = {tuple(sorted((cell[i], cell[j]))) for cell in mesh._corner_tags
             for i,j in ((0,1),(0,2),(1,2))}
    assert vertices-len(edges)+len(mesh) == -2


@pytest.mark.parametrize('suffix', ['.geo', '.step'])
def test_curved_tetrahedral_solid(tmp_path, suffix):
    path = tmp_path/('ball'+suffix)
    if suffix == '.geo':
        path.write_text('SetFactory("OpenCASCADE"); Sphere(1)={0,0,0,1}; '
                        'Physical Surface("wall")={1}; Physical Volume("solid")={1};\n')
    else:
        gmsh.initialize([], readConfigFiles=False)
        try:
            gmsh.option.setNumber('General.Terminal', 0)
            gmsh.model.occ.addSphere(0,0,0,1)
            gmsh.model.occ.synchronize()
            gmsh.write(str(path))
        finally:
            gmsh.finalize()
    mesh = jx.read_gmsh(path, cell_type='simplex', mesh_size=.8, geometry_order=2)
    space = jx.FunctionSpace(mesh)
    assert mesh.cell_type == 'simplex' and not mesh.subdivided
    assert all(p.n_cp == 10 for p in mesh)
    assert mesh.geometry_error < .015
    volume = jx.integrate(lambda x: 1., space)
    assert volume == pytest.approx(4*np.pi/3, rel=.008)
    if suffix == '.geo':
        assert set(space.boundaries) == {'wall'}
        assert len(mesh.cell_groups['solid']) == len(mesh)


def test_simplex_mode_disables_recombination(tmp_path):
    path=tmp_path/'quad.geo'
    path.write_text('SetFactory("OpenCASCADE"); Rectangle(1)={0,0,0,1,1}; '
                    'Transfinite Curve{:}=3; Transfinite Surface{1}; Recombine Surface{1};')
    mesh = jx.read_gmsh(path, cell_type='simplex', mesh_size=.4)
    assert all(p.n_cp == 3 for p in mesh)
    assert not mesh.subdivided


def test_invalid_simplex_inputs(tmp_path):
    xyz = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
    path = msh(tmp_path/'duplicate.msh', xyz, [(4,0,[0,1,2,3]), (4,0,[0,1,2,3])])
    with pytest.raises(GmshImportError, match='duplicate'):
        jx.read_gmsh(path, cell_type='simplex')
    path = msh(tmp_path/'degenerate.msh', [[0,0,0],[1,0,0],[2,0,0]], [(2,0,[0,1,2])])
    with pytest.raises(GmshImportError, match='degenerate'):
        jx.read_gmsh(path, cell_type='simplex')
    path = msh(tmp_path/'quad.msh', [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], [(3,0,[0,1,2,3])])
    with pytest.raises(GmshImportError, match='complete triangles/tetrahedra'):
        jx.read_gmsh(path, cell_type='simplex')
    with pytest.raises(ValueError, match='cell_type'):
        jx.read_gmsh(path, cell_type='invalid')
    assert not gmsh.isInitialized()


def test_fold_between_quality_samples_is_rejected(tmp_path):
    gmsh.initialize([], readConfigFiles=False)
    try:
        kind = gmsh.model.mesh.getElementType('Triangle', 3)
        _, _, _, count, ref, _ = gmsh.model.mesh.getElementProperties(kind)
        ref = np.asarray(ref).reshape(count, 2)
    finally:
        gmsh.finalize()
    x = ref[:, 0]
    xyz = np.column_stack([x**3/3 - .2*x**2 + .039*x, ref[:, 1], np.zeros(count)])
    # detJ=(u-.2)^2-.001 is positive at all u=k/8 sample points,
    # but negative inside a narrow interval; the Bernstein bound must catch it.
    path = msh(tmp_path/'fold.msh', xyz, [(kind, 0, list(range(count)))])
    with pytest.raises(GmshImportError, match='Jacobian'):
        jx.read_gmsh(path, cell_type='simplex', min_scaled_jacobian=0)


def test_nonmanifold_and_nonconforming_high_order_faces_rejected(tmp_path):
    xyz = [[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[.2,2,0]]
    path = msh(tmp_path/'nonmanifold.msh', xyz,
               [(2,0,[0,1,2]), (2,0,[1,0,3]), (2,0,[0,1,4])])
    with pytest.raises(GmshImportError, match='non-manifold'):
        jx.read_gmsh(path, cell_type='simplex')
    # The shared edge has two different midpoint node tags, even though
    # their coordinates coincide: this is not a conforming high-order mesh.
    xyz = [[0,0,0],[1,0,0],[0,1,0],[0,-1,0],
           [.5,0,0],[.5,.5,0],[0,.5,0], [.5,0,0],[0,-.5,0],[.5,-.5,0]]
    path = msh(tmp_path/'trace.msh', xyz, [(9,0,[0,1,2,4,5,6]), (9,0,[1,0,3,7,8,9])])
    with pytest.raises(GmshImportError, match='nonconforming high-order'):
        jx.read_gmsh(path, cell_type='simplex')
