import jax
import os
os.environ["JAX_PLATFORMS"] = "cpu"

try:
    from jax.lib import xla_bridge
    print(f"Backend: {xla_bridge.get_backend().platform}")
except Exception as e:
    print(f"Error: {e}")

