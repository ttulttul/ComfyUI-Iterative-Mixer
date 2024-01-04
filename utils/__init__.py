"Utilities for the Iterative Mixing Sampler node pack."

import inspect
from typing import Dict
import torch

def generate_class_map(cls: type, name_attribute: str) -> Dict[str, type]:
    """
    Generate a dict mapping from strings to classes. Each class should
    have a name_attribute (e.g. 'name') that provides a friendly name
    for the class.
    """
    caller_frame = inspect.currentframe().f_back
    caller_module_dict = caller_frame.f_globals

    subclasses_dict = {}
    for _, subclass in caller_module_dict.items():
        if inspect.isclass(subclass) and issubclass(subclass, cls) and subclass is not cls:
            theName = getattr(subclass, name_attribute, None)
            if theName is None:
                raise ValueError(f"missing {name_attribute} in {subclass}")
            subclasses_dict[theName] = subclass
    return subclasses_dict

@torch.no_grad
def generate_noised_latents(x, sigmas, normalize=False):
    """
    Generate all noised latents for a given initial latent image and sigmas in parallel.

    :param x: Original latent image batch as a PyTorch tensor.
    :param sigmas: Array of sigma values for each timestep as a PyTorch tensor.
    :return: A tensor containing all noised latents for each timestep.
    """
    # Ensure that x and sigmas are on the same device (e.g., CPU or CUDA)
    device = x.device
    sigmas = sigmas.to(device) # ignore the first sigma
    d = x.shape[0]
    num_sigmas = len(sigmas)

    if d > 1:
        raise ValueError("Iterative Mixing currently only works with single latent batches")

    # Expand x and sigmas to match each other in the first dimension
    # x_expanded shape will be:
    # [batch_size * num_sigmas, channels, height, width]
    x_expanded = x.repeat(num_sigmas, 1, 1, 1)
    sigmas_expanded = sigmas.repeat_interleave(d)

    # Create a noise tensor with the same shape as x_expanded
    noise = torch.randn_like(x_expanded)

    # Multiply noise by sigmas, reshaped for broadcasting
    noised_latents = x_expanded + noise * sigmas_expanded.view(-1, 1, 1, 1)

    # Unscientifically normalize the batch based on the mean of the first
    # latent in the batch (i.e. the original latent image).
    if normalize:
        source_mean = noised_latents[0].mean()
        noised_latents = (noised_latents - source_mean)

    return noised_latents

# Borrowed from BlenderNeko: https://github.com/BlenderNeko/ComfyUI_Noise/blob/master/nodes.py
@torch.no_grad
def slerp(low: torch.Tensor, high: torch.Tensor, val: float):
    """
    SLERP two latents.
    """
    dims = low.shape

    # We need at least two dimensions (batch plus whatever else)
    assert len(dims) >= 2

    # Flatten to batches.
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)