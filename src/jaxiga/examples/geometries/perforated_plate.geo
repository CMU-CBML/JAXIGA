// A 12 x 6 plate with unequal circular holes and an offset rounded slot.
SetFactory("OpenCASCADE");
Rectangle(1) = {0, 0, 0, 12, 6};
Disk(2) = {3, 3, 0, 1, 1};
Disk(3) = {9, 4, 0, 0.8, 0.8};
Rectangle(4) = {6.5, 1.1, 0, 2, 0.8};
Disk(5) = {6.5, 1.5, 0, 0.4, 0.4};
Disk(6) = {8.5, 1.5, 0, 0.4, 0.4};
slot() = BooleanUnion{ Surface{4}; Delete; }{ Surface{5, 6}; Delete; };
plate() = BooleanDifference{ Surface{1}; Delete; }{ Surface{2, 3, slot()}; Delete; };

eps = 1e-6;
fixed() = Curve In BoundingBox{-eps, -eps, -eps, eps, 6+eps, eps};
loaded() = Curve In BoundingBox{12-eps, -eps, -eps, 12+eps, 6+eps, eps};
holes() = Curve In BoundingBox{eps, eps, -eps, 12-eps, 6-eps, eps};
bottom() = Curve In BoundingBox{-eps, -eps, -eps, 12+eps, eps, eps};
top() = Curve In BoundingBox{-eps, 6-eps, -eps, 12+eps, 6+eps, eps};
Physical Curve("fixed") = {fixed()};
Physical Curve("loaded") = {loaded()};
Physical Curve("holes") = {holes()};
Physical Curve("free") = {bottom(), top()};
Physical Surface("plate") = {plate()};
// Coarse starting mesh; decrease this size for more geometry patches.
Mesh.MeshSizeMin = 2.0;
Mesh.MeshSizeMax = 2.0;
Mesh.MinimumCirclePoints = 12;
