import os

# JAX tests are small and run on the CPU, so they never contend with a sweep for GPU memory
# (each JAX process otherwise preallocates 75% of it).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
