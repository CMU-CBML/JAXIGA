"""Global configuration for JAXIGA.

Importing :mod:`jaxiga` imports this module first, which enables 64-bit floats.
FEM-grade conditioning needs them: at degree 3-5 with several refinements the
element tangents are badly enough conditioned that float32 loses the solution.
"""

import os
import warnings

import jax

_X64_KEY = "jax_enable_x64"


def _enable_x64() -> None:
    """Turn on x64, warning only if the user explicitly asked for it off.

    ``jax.config.values[_X64_KEY]`` defaults to False, so it cannot distinguish
    "user disabled this" from "never touched"; the environment variable can.
    """
    env = os.environ.get("JAX_ENABLE_X64")
    explicitly_off = env is not None and env.lower() in ("0", "false", "no")

    jax.config.update(_X64_KEY, True)

    if explicitly_off:
        warnings.warn(
            "JAX_ENABLE_X64 was set to disable 64-bit floats, but jaxiga requires "
            "them and has re-enabled x64. Single precision is not supported.",
            RuntimeWarning,
            stacklevel=3,
        )


_enable_x64()

# Tolerance for matching points/knots during setup (vertex hashing, interface
# conformity checks). Setup-stage only, never on the traced compute path.
TOL = 1e-10
