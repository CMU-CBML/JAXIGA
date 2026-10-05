"""JAXIGA - differentiable isogeometric analysis on JAX.

Importing this package enables 64-bit floats (see :mod:`jaxiga.config`).

Quick start::

    import jaxiga as jx

    patch = jx.primitives.quadrilateral(
        [[0, 0], [1, 0], [1, 1], [0, 1]]
    ).elevate(3).refine(5)
    V = jx.FunctionSpace(patch)

    problem = jx.Poisson(V, dirichlet=[jx.DirichletBC(0.0, where="left")], source=f)
    sol = jx.solve(problem, params={"a0": 1.0})
    print(jx.errornorm(sol, exact, "L2"))
"""

from jaxiga import config  # noqa: F401  (enables x64 before anything else)

__version__ = "0.3.0"

from jaxiga.geometry import io, primitives
from jaxiga.geometry.io import read_gismo, read_json, read_gmsh, read_matlab_nurbs
from jaxiga.geometry.primitives import graded_knots
from jaxiga.geometry.multipatch import (
    Interface,
    NonConformingError,
    OrientationError,
    Topology,
)
from jaxiga.geometry.nurbs import Patch
from jaxiga.geometry.simplex import SimplexPatch
from jaxiga.geometry.mixed import split_to_simplices
from jaxiga.geometry.refinement import refine_cells, bezier_cells
from jaxiga.space import pointset
from jaxiga.space.evaluation import BasisData, evaluate
from jaxiga.space.function_space import DofMap, FunctionSpace, expand_dofs
from jaxiga.space.hierarchical import (
    Hierarchy,
    HierarchyError,
    dorfler_mark,
    refine_elements,
)
from jaxiga.space.pointset import PointSet, boundary_gauss, gauss, greville, grid
from jaxiga.forms import materials
from jaxiga.forms.library import (
    Hyperelasticity,
    KirchhoffPlate,
    LinearElasticity,
    PhaseFieldDamage,
    PhaseFieldDisplacement,
    Poisson,
)
from jaxiga.forms.problem import ClampedBC, DirichletBC, Neumann, Problem
from jaxiga.forms.shell import KirchhoffLoveShell
from jaxiga.methods import solve
from jaxiga.solvers.newton import history_list
from jaxiga.methods.galerkin import mass_matrix, mass_triplets
from jaxiga.solvers import dynamics, phase_field
from jaxiga.solvers.dynamics import generalized_alpha, newmark
from jaxiga.methods._common import PointField
from jaxiga.post.estimators import adapt, residual_indicator
from jaxiga.space.transfer import ParentMap, carry_points, parent_map, project
from jaxiga.post.norms import errornorm, integrate, norm
from jaxiga.post.solution import Solution, locate_points
from jaxiga.solvers.linear import LinearOptions

__all__ = [
    # geometry
    "Patch",
    "SimplexPatch",
    "split_to_simplices",
    "refine_cells",
    "bezier_cells",
    "read_matlab_nurbs",
    "primitives",
    "io",
    "read_json",
    "read_gismo",
    "read_gmsh",
    "graded_knots",
    "Interface",
    "Topology",
    "NonConformingError",
    "OrientationError",
    # space
    "FunctionSpace",
    "DofMap",
    "PointSet",
    "pointset",
    "gauss",
    "boundary_gauss",
    "greville",
    "grid",
    "evaluate",
    "BasisData",
    "expand_dofs",
    # adaptivity
    "refine_elements",
    "dorfler_mark",
    "residual_indicator",
    "adapt",
    "Hierarchy",
    "HierarchyError",
    # forms
    "Problem",
    "DirichletBC",
    "ClampedBC",
    "Neumann",
    "Poisson",
    "LinearElasticity",
    "Hyperelasticity",
    "KirchhoffPlate",
    "KirchhoffLoveShell",
    "PhaseFieldDisplacement",
    "PhaseFieldDamage",
    "PointField",
    "materials",
    # solving
    "solve",
    "LinearOptions",
    "history_list",
    # dynamics
    "mass_matrix",
    "mass_triplets",
    "dynamics",
    "newmark",
    "generalized_alpha",
    # phase-field fracture
    "phase_field",
    # transfer between refinement levels
    "ParentMap",
    "parent_map",
    "project",
    "carry_points",
    # post
    "Solution",
    "locate_points",
    "errornorm",
    "norm",
    "integrate",
]
