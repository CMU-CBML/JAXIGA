"""Phase-field fracture: single-edge-notched plate in tension.

Staggered scheme. Each load step alternates between two problems that share one
geometry:

* :class:`PhaseFieldDisplacement` -- elasticity whose stiffness is degraded by
  the current damage field;
* :class:`PhaseFieldDamage` -- AT2 damage driven by the history field, the
  running maximum of the tensile strain energy (which is what makes damage
  irreversible).

Coupling data flows between them as :class:`PointField` values at the
quadrature points.

Two ingredients are needed for the load--displacement curve to show the sharp
brittle drop this benchmark is known for, rather than a smeared ductile one:

1. *A graded mesh.* The regularisation length must be resolved by several
   elements. Rather than refining the whole square, knots are clustered around
   the crack path, which is exact and unconstrained in IGA. This gives
   ``l/h`` of about four for a quarter of the elements a uniform mesh would
   need.

2. *Adaptive load stepping.* Once the crack starts running, the reaction falls
   by most of its peak value over a very small displacement range. Uniform
   increments step straight over it. Here an increment that produces too large
   a load drop is rejected and halved, so the softening branch is traced
   rather than jumped.
"""

import time

import jax.numpy as jnp
import numpy as np

import jaxiga as jx
from jaxiga.forms.library import edge_crack_distance, initial_history, tensile_energy
from jaxiga.methods._common import reactions
from jaxiga.space import pointset as P
from jaxiga.space.evaluation import evaluate

# -- discretisation --------------------------------------------------------
DEG = 2
N_X = 40                 # uniform spans along the crack direction
N_FINE, N_COARSE = 48, 4  # spans inside / outside the refined band

# -- material and regularisation ------------------------------------------
E, NU = 210e3, 0.3   # N/mm^2
GC = 2.7             # N/mm
ELL = 0.0125         # regularisation length (the benchmark value)
B_SEED = 1e3

# -- load stepping ---------------------------------------------------------
U_MAX = 0.008
DU0, DU_MIN = 5e-4, 1.5e-5
DROP_FRAC = 0.09     # reject an increment that drops the load by more than this
MAX_STEPS = 110
STAGGER_TOL, MAX_STAGGERED = 1e-5, 12


patch = jx.primitives.rectangle(0.0, 0.0, 1.0, 1.0).elevate(DEG)
patch = patch.insert_knots(
    np.linspace(0.0, 1.0, N_X + 1)[1:-1],
    jx.graded_knots(N_FINE, N_COARSE, centre=0.5, half_width=max(6 * ELL, 0.08)),
)

Vu = jx.FunctionSpace(patch, vec=2)
Vd = jx.FunctionSpace(patch, vec=1)
basis = evaluate(Vu, P.gauss(Vu))
top_dofs = np.asarray(Vu.boundaries["top"].dofs)

h_min = np.diff(np.unique(np.asarray(patch.knots[1]))).min()
print(Vu.summary())
print(f"finest element in y: h = {h_min:.5f},  l/h = {ELL / h_min:.1f}")

history = initial_history(basis, edge_crack_distance((0.5, 0.5)), B_SEED, ELL, GC)
phi = jnp.zeros((basis.n_elems, basis.n_q))
damage_problem = jx.PhaseFieldDamage(Vd)


def staggered(u_top, phi_in, history_in):
    """One load step: alternate elasticity and damage until both settle."""
    disp = jx.PhaseFieldDisplacement(
        Vu,
        plane="strain",
        dirichlet=[
            jx.DirichletBC([0.0, 0.0], where="bottom"),
            jx.DirichletBC([0.0, u_top], where="top"),
        ],
    )
    ph, hist = phi_in, history_in
    for inner in range(MAX_STAGGERED):
        params = {"E": E, "nu": NU, "phi": jx.PointField(ph)}
        sol_u = jx.solve(disp, params=params)
        hist = jnp.maximum(hist, tensile_energy(sol_u, {"E": E, "nu": NU}, basis))
        sol_d = jx.solve(
            damage_problem, params={"Gc": GC, "l": ELL, "H": jx.PointField(hist)}
        )
        new = jnp.clip(sol_d.at(basis=basis)[..., 0], 0.0, 1.0)
        converged = float(jnp.max(jnp.abs(new - ph))) < STAGGER_TOL
        ph = new
        if converged:
            break
    force = float(jnp.sum(reactions(sol_u, params=params)[2 * top_dofs + 1]))
    return ph, hist, force, sol_d, inner + 1


print("\n step   u_top      F_y [N]    max phi   dmg area   stag   du")
u, du, f_last, f_peak, rejected = 0.0, DU0, None, 0.0, 0
curve = []
sol_d = None
t0 = time.perf_counter()

while u < U_MAX and len(curve) < MAX_STEPS:
    trial = min(u + du, U_MAX)
    ph, hist, force, sd, n_stag = staggered(trial, phi, history)

    # Reject on load *drops* only: during the elastic rise the reaction
    # legitimately changes a lot per step, and treating that as a failed
    # increment would shrink the step long before the crack starts running.
    if f_last is not None and (f_last - force) > DROP_FRAC * max(f_peak, 1.0) and du > DU_MIN:
        du /= 2
        rejected += 1
        continue

    rising = f_last is None or force >= f_last
    u, phi, history, sol_d = trial, ph, hist, sd
    f_peak = max(f_peak, force)
    f_last = force
    curve.append((u, force))

    area = float(jnp.sum(basis.w * phi))
    print(f" {len(curve):^5d} {u:.5f}  {force:9.2f}   {float(jnp.max(phi)):.5f}   "
          f"{area:.5f}   {n_stag:^4d}  {du:.1e}")

    if rising and du < DU0:
        du = min(du * 1.6, DU0)
    if f_peak > 0 and force < 0.03 * f_peak and len(curve) > 3:
        print(" crack through the ligament; stopping")
        break

elapsed = time.perf_counter() - t0

uu = np.array([c[0] for c in curve])
ff = np.array([c[1] for c in curve])
peak = int(np.argmax(ff))
hi = np.where(ff[peak:] <= 0.9 * ff[peak])[0]
lo = np.where(ff[peak:] <= 0.2 * ff[peak])[0]
span = (uu[peak + lo[0]] - uu[peak + hi[0]]) / uu[peak] if len(hi) and len(lo) else np.nan

print(f"\npeak {ff[peak]:.1f} N at u = {uu[peak]:.5f}, final {ff[-1]:.1f} N")
print(f"load falls 90% -> 20% of peak over {span * 100:.1f}% of the peak displacement")
print(f"{len(curve)} accepted steps, {rejected} rejected, {elapsed:.0f}s")

np.savetxt("pf2d_load_displacement.csv", np.column_stack([uu, ff]),
           delimiter=",", header="displacement,reaction", comments="")
sol_d.to_vtk("pf2d_damage", n=3)
print("wrote pf2d_damage.vtu and pf2d_load_displacement.csv")
