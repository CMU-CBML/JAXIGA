# Example geometry data

`igapack_blade.mat` is a byte-for-byte copy of
`IGAPack/elasticity3D/ExampleData/blade.mat` in the upstream IGAPack source tree, used by
`IGAPack/elasticity3D/Blade.m` through `init3DGeometryGIFT.m`.

`igapack_connecting_rod.mat` is a byte-for-byte copy of
`IGAPack/elasticity3D/ExampleData/ConnRod.mat`, used by
`IGAPack/elasticity3D/ConnectingRodMP.m` through `init3DGeometryGIFTMP.m`.
It contains the corrected shaft interfaces already stored in that MAT file;
the Python example does not rerun `make_connecting_rod.m` or `fixConnRod.m`.

Both contain NURBS-toolbox structs with homogeneous control points, weights,
orders and knot vectors. They are bundled so installed examples need neither
MATLAB nor the IGAPack source tree. Python reverses negative parametric volume
orientations and raises degrees without changing the physical geometry.

The blade example uses its own fixed-root/transverse-tip loading. The connecting
rod follows the support/load regions and material of the MATLAB example, with
a different analysis discretization. Neither claims numerical reproduction of
IGAPack's adaptive PHT/GIFT studies. See `docs/igapack.md` in the source tree.

The bundled geometry inputs retain the IGAPack license in
[`LICENSE-IGAPack.txt`](LICENSE-IGAPack.txt).
