"Utilities for the Iterative Mixing Sampler node pack."

import inspect
import logging
from typing import Dict, List, Tuple
import torch

def _trace(msg):
    "Print out a message to the log facility; using warning makes sure it prints."
    #logging.warning(msg)
    pass

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

@torch.no_grad()
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

    # Expand x and sigmas to match each other in the first dimension
    # x_expanded shape will be:
    # [batch_size * num_sigmas, channels, height, width]
    x_expanded = x.repeat(d * num_sigmas, 1, 1, 1)
    sigmas_expanded = sigmas.repeat_interleave(d)

    # Create a noise tensor with the same shape as x_expanded
    noise = torch.randn_like(x_expanded)

    # Multiply noise by sigmas, reshaped for broadcasting
    noised_latents = x_expanded + noise * sigmas_expanded.view(-1, 1, 1, 1)

    # Unscientifically normalize the batch based on the mean of the first
    # latent in the batch (i.e. the original latent image).
    # XXX: I'm going to deprecate this because it's not doing anything useful.
    if normalize:
        logging.warning("the normalize option is being deprecated")
        source_mean = noised_latents[0].mean()
        noised_latents = (noised_latents - source_mean)

    return noised_latents

# Borrowed from BlenderNeko: https://github.com/BlenderNeko/ComfyUI_Noise/blob/master/nodes.py
@torch.no_grad()
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

@torch.no_grad()
def geometric_ranges(start: int, end: int, rewind: float=0.5, max_start: int=None) -> List[Tuple]:
    """
    Generates a list of tuples demarcating the start and end point of a range,
    with the start value in each tuple rising by 50% each time. This function
    is useful for generating a set of step ranges for implementing rewind
    sampling.
    """
    ranges = []
    ranges.append((start, end))

    if max_start is None:
        max_start = end - 1
    
    while True:
        new_start = start + int((end - start) * rewind)
        if new_start >= end or new_start <= start or new_start > max_start:
            break
        ranges.append((new_start, end))
        start = new_start
    
    return ranges

@torch.no_grad()
def match_normalize(target_tensor, source_tensor, dimensions=4):
    "Adjust target_tensor based on source_tensor's mean and stddev"   
    if len(target_tensor.shape) != dimensions:
        raise ValueError("source_latent must have four dimensions")
    if len(source_tensor.shape) != dimensions:
        raise ValueError("target_latent must have four dimensions")

    # Put everything on the same device
    device = target_tensor.device

    # Calculate the mean and std of target tensor
    tgt_mean = target_tensor.mean(dim=[2, 3], keepdim=True).to(device)
    tgt_std = target_tensor.std(dim=[2, 3], keepdim=True).to(device)
    
    # Calculate the mean and std of source tensor
    src_mean = source_tensor.mean(dim=[2, 3], keepdim=True).to(device)
    src_std = source_tensor.std(dim=[2, 3], keepdim=True).to(device)
    
    # Normalize target tensor to have mean=0 and std=1, then rescale
    normalized_tensor = (target_tensor.clone() - tgt_mean) / tgt_std * src_std + src_mean
    
    return normalized_tensor
