import torch
import jax
from jux.env import JuxEnv

# try running with XLA_PYTHON_CLIENT_PREALLOCATE=false if there are cudnn-related errors
# and make sure the PATH environment variable makes sense

print(jax.devices())
print(torch.cuda.is_available())

env = JuxEnv()
