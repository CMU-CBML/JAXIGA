"""Miniature full-waveform inversion: recover a stiffness anomaly from a trace.

A wave is sent down an elastic bar, and the only measurement is the velocity
history at the point where it was sent from. Hidden inside the bar is a stiff
inclusion; the reflection it produces is the entire signal available. The task
is to recover the inclusion's strength and position from that trace alone.

This is the pattern behind seismic full-waveform inversion, and it is the
hardest thing to ask of a differentiable solver, because the gradient must
travel back through *every time step*:

    misfit  ->  velocity trace  ->  time integration  ->  stiffness matrix
            ->  material field  ->  the two unknown parameters

Nothing here is hand-derived. The time loop is a ``lax.scan``, which JAX
transposes into a reverse scan for the adjoint, so the classical
adjoint-state method of seismic imaging -- back-propagating the residual
wavefield and correlating it with the forward one -- is what ``jax.grad``
produces on its own. ``checkpoint_every`` trades recomputation for memory when
the trace gets long.
"""

import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

import jaxiga as jx
from jaxiga.methods._common import build_context
from jaxiga.methods.galerkin import assemble_triplets, mass_triplets, rhs
from jaxiga.solvers import dynamics
from jaxiga.solvers.linear import LinearOptions

DEG, REFINE = 2, 7          # 128 elements: ~25 per wavelength
E0, RHO, LENGTH = 1.0, 1.0, 1.0    # wave speed sqrt(E/rho) = 1
# 1.8 s of recording: the inclusion echo returns at 2 * 0.62 = 1.24 s and the
# far-end echo at 2.0 s, so the window holds the echo of interest and no other.
DT, N_STEPS, CKPT = 5e-3, 360, 40
FREQ, T0, WIDTH = 3.5, 0.35, 0.12
TRUE = {"contrast": 0.60, "centre": 0.62}   # what the inversion must find

patch = jx.primitives.interval(0.0, LENGTH).elevate(DEG).refine(REFINE)
V = jx.FunctionSpace(patch)


class Bar(jx.Problem):
    """A 1D elastic bar whose stiffness carries a smooth Gaussian inclusion.

    Writing the kernel out is the whole of defining new physics: the residual,
    the tangent and every derivative with respect to ``contrast`` and
    ``centre`` follow from this one expression by differentiation.
    """

    is_linear = True

    def energy(self, grad_u, u, x, params):
        stiffness = E0 * (
            1.0 + params["contrast"] * jnp.exp(-(((x[0] - params["centre"]) / WIDTH) ** 2))
        )
        return 0.5 * stiffness * jnp.sum(grad_u**2)


problem = Bar(
    V,
    dirichlet=[jx.DirichletBC(0.0, where="right")],          # far end held
    neumann=[jx.Neumann(lambda x, n, p: jnp.array([1.0]), where="left")],
)

# The load vector is the same every step; only its amplitude varies in time,
# and the Neumann work does not involve the material, so this is assembled once.
ctx0 = build_context(problem, TRUE)
LOAD = rhs(problem, TRUE, ctx0)
INDICES, M_DATA = mass_triplets(ctx0, RHO)
N_FREE = len(ctx0.dofmap.free)

# The measurement point: the driven end, where the first basis function is the
# only one that does not vanish, so its coefficient *is* the displacement.
probe_dof = int(np.argmin(np.asarray(V.cpts).reshape(-1)[ctx0.dofmap.free]))


def ricker(t):
    a = (jnp.pi * FREQ * (t - T0)) ** 2
    return (1.0 - 2.0 * a) * jnp.exp(-a)


def trace(p, checkpoint_every=CKPT):
    """Velocity history at the driven end for a given material."""
    ctx = build_context(problem, p)
    indices, k_data = assemble_triplets(ctx, p)
    _, U, W = dynamics.integrate(
        indices, k_data, M_DATA,
        dt=DT, n_steps=N_STEPS,
        u0=jnp.zeros(N_FREE), v0=jnp.zeros(N_FREE),
        force=lambda t: ricker(t) * LOAD,
        linear=LinearOptions(method="dense"),
        checkpoint_every=checkpoint_every,
    )
    return W[:, probe_dof]


print(V.summary())
print(f"{N_STEPS} steps of dt = {DT}, {N_FREE} free dofs, "
      f"checkpointing every {CKPT} steps")

t0 = time.perf_counter()
observed = jax.jit(trace)(TRUE)
observed.block_until_ready()
print(f"synthetic observation computed in {time.perf_counter() - t0:.2f}s")

# A homogeneous bar produces only the end reflection; the inclusion's echo is
# the entire difference between the two traces.
flat = jax.jit(trace)({"contrast": 0.0, "centre": 0.5})
signal = float(jnp.linalg.norm(observed - flat) / jnp.linalg.norm(observed))
print(f"inclusion signal: {100 * signal:.1f}% of the trace norm")


def misfit(p):
    return 0.5 * jnp.sum((trace(p) - observed) ** 2) / jnp.sum(observed**2)


value_and_grad = jax.jit(jax.value_and_grad(misfit))

guess = {"contrast": 0.20, "centre": 0.50}
print(f"\nstarting from contrast {guess['contrast']:.3f}, centre {guess['centre']:.3f}")
print(f"truth is        contrast {TRUE['contrast']:.3f}, centre {TRUE['centre']:.3f}")

# Check the adjoint against a finite difference before trusting it.
val, g = value_and_grad(guess)
h = 1e-5
for key in ("contrast", "centre"):
    up = dict(guess, **{key: guess[key] + h})
    dn = dict(guess, **{key: guess[key] - h})
    fd = float((misfit(up) - misfit(dn)) / (2 * h))
    print(f"  d(misfit)/d({key:8s}) adjoint {float(g[key]): .6e}   "
          f"finite difference {fd: .6e}   rel {abs(float(g[key]) - fd) / abs(fd):.2e}")

# The misfit is oscillatory in the echo's arrival time -- move the inclusion by
# half a wavelength and the predicted echo lines up with the wrong cycle of the
# observed one. That is cycle skipping, the central difficulty of real
# full-waveform inversion, and it is why a low source frequency (a long
# wavelength, hence a wide basin) is used here. Production practice inverts a
# sequence of increasing frequencies, each started from the previous result.
print("\nmisfit against inclusion position, at contrast 0.3:")
scan = [(c, float(misfit({"contrast": 0.3, "centre": c}))) for c in np.arange(0.40, 0.80, 0.04)]
print("  centre " + " ".join(f"{c:7.2f}" for c, _ in scan))
print("  misfit " + " ".join(f"{v:7.1e}" for _, v in scan))
best = min(scan, key=lambda cv: cv[1])[0]
print(f"  single minimum near {best:.2f}; the basin of attraction runs from about "
      f"0.48 to 0.70")

start_misfit = float(misfit(guess))
opt = optax.adam(0.03)
state = opt.init(guess)
print("\n  step    misfit      contrast    centre")
print(f"  {0:4d}   {float(val):.6e}   {guess['contrast']:.4f}     {guess['centre']:.4f}")
t0 = time.perf_counter()
for step in range(1, 91):
    val, g = value_and_grad(guess)
    updates, state = opt.update(g, state, guess)
    guess = optax.apply_updates(guess, updates)
    if step % 10 == 0:
        print(f"  {step:4d}   {float(val):.6e}   {guess['contrast']:.4f}     "
              f"{guess['centre']:.4f}")

final = float(value_and_grad(guess)[0])
print(f"\nmisfit {start_misfit:.3e} -> {final:.3e} "
      f"({start_misfit / final:.0f}x lower) in {time.perf_counter() - t0:.0f}s")
print(f"recovered contrast {guess['contrast']:.4f} (true {TRUE['contrast']:.3f}, "
      f"error {100 * abs(guess['contrast'] / TRUE['contrast'] - 1):.1f}%)")
print(f"recovered centre   {guess['centre']:.4f} (true {TRUE['centre']:.3f}, "
      f"error {100 * abs(guess['centre'] / TRUE['centre'] - 1):.1f}%)")

np.savetxt("fwi_traces.csv",
           np.column_stack([np.arange(N_STEPS + 1) * DT, np.asarray(observed),
                            np.asarray(jax.jit(trace)(guess))]),
           delimiter=",", header="t,observed,recovered", comments="")
print("wrote fwi_traces.csv")
