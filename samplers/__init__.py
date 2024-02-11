from abc import ABC
import logging
from typing import Optional
import torch
from tqdm.auto import trange

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.samplers
import comfy.k_diffusion.sampling

from ..utils import generate_class_map, generate_noised_latents, slerp, geometric_ranges, _trace
from ..utils.noise import perlin_masks

def _safe_get(theDict, key, theType):
    """
    Safely gives you a value of the requested type from a dict given a dict key.
    The return type is guaranteed to be of type theType. If the value is None,
    then we return None rather than generating a conversion error.
    """
    value = theDict.get(key)

    if value is None:
        return None

    try:
        return theType(theDict.get(key))
    except (ValueError, TypeError) as e:
        raise ValueError(f'must provide a {theType} value for {key}') from e
    

class ModularKSamplerX0Inpaint(comfy.samplers.KSamplerX0Inpaint):
    """
    A version of the KSamplerX0Inpaint that breaks out the application of
    the denoise mask so that we can do fancier things with masking separate
    from denoising.
    """
    def __init__(self, model: comfy.samplers.KSamplerX0Inpaint):
        super().__init__(model.inner_model)
        self.latent_image = model.latent_image
        self.noise = model.noise

    @torch.no_grad()
    def before(self, x: torch.tensor, z_prime: torch.tensor, sigma: float, denoise_mask: Optional[torch.Tensor]) -> torch.tensor:
        """
        Apply the denoise mask appropriately to the x and z_prime tensors.
        Call this method before running denoise steps in the iterative mixing sampler.
        """

        def apply_mask(x, denoise_mask, latent_mask, sigma_reshaped):
            return x * denoise_mask + (self.latent_image + self.noise * sigma_reshaped) * latent_mask

        sigma = torch.tensor([sigma], device=x.device)
        if denoise_mask is not None:
            latent_mask = 1. - denoise_mask
            sigma_reshaped = sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1))

            x = apply_mask(x, denoise_mask, latent_mask, sigma_reshaped)
            z_prime = apply_mask(z_prime.unsqueeze(0), denoise_mask, latent_mask, sigma_reshaped)
            z_prime = z_prime.squeeze(0)

        return x, z_prime

    @torch.no_grad()
    def after(self, denoised_x: torch.Tensor, denoise_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        After the model is called, call this function to apply the latent mask
        again so that your denoising effectively applied only to the part that
        was masked.
        """
        if denoise_mask is not None:
            latent_mask = 1. - denoise_mask
            denoised_x = denoised_x * denoise_mask + self.latent_image * latent_mask
        else:
            # nothing to do
            pass

        return denoised_x

    @torch.no_grad()
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, model_options={}, seed=None):
        return self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, model_options=model_options, seed=seed)

def get_blending_schedule(indices: list[int], blending_schedule: str, **kwargs):
    """
    Return a blending schedule for the given list of step indices and the named
    blending schedule (e.g. "cosine").
    """
    # Check whether the provided schedule exists.
    if blending_schedule not in BLENDING_SCHEDULE_MAP:
        raise ValueError(f"invalid blending_schedule: {blending_schedule}")

    # Generate a blending schedule by running the __call__ method of
    # the correct blending schedule class.
    blending_schedule_generator = BLENDING_SCHEDULE_MAP[blending_schedule]()
    schedule = blending_schedule_generator(indices, **kwargs)
    
    return schedule

def get_blending_function(blending_function):
    return BLENDING_FUNCTION_MAP[blending_function]()

class IterativeMixingSampler(ABC):
    """
    Base class for iterative mixing versions of the usual samplers (Euler, LCM, etc.).
    """

    @torch.no_grad()
    def run(self, model, x, sigmas, c1=None, blend=None, extra_args=None, callback=None,
                 disable=None,
                 start_sigma=0,
                 end_sigma=None,
                 s_churn=0., s_tmin=0., s_tmax=float('inf'),
                 **kwargs):
        """
        Prepare the iterative mixing variables the derived classes will need.
        Each sampler's implementation is in a sample function that is not implemented
        in the base class.
        """

        # Set the final step to accommodate rewind.
        # start_sigma:end_sigma specify the slice of the sigmas
        # that we will use to generate the noised latents. So if we are
        # only denoising part of the way, we will be given a start_sigma
        # index value that is greater than 0.
        end_sigma = end_sigma or len(sigmas)

        if end_sigma > len(sigmas):
            raise ValueError("end_sigma is out of range")
        if start_sigma < 0:
            raise ValueError("start_sigma is out of range")

        # Generate a batch of progressively noised latents according
        # to the sigmas (noise schedule). Note that if we are doing rewind,
        # the sigmas may not start at step 0, so z_prime[0] won't necessarily be
        # a fully noised latent.
        _trace(f"len(sigmas)={len(sigmas)}, start_sigma:end_sigma={start_sigma}:{end_sigma}")
        z_prime = generate_noised_latents(x, sigmas[start_sigma:end_sigma])

        # Select the correct slice of the c1 blending schedule as well in case we
        # are doing rewind (and not starting at step 0).
        _trace(f"len(c1)={len(c1)}")
        c1 = c1[start_sigma:end_sigma]

        # If there is a denoising mask, mask the z_primes so that they are zero outside
        # of the masked area. This ensures that we are never mixing anything into that part
        # of the latent image.
        if extra_args and extra_args['denoise_mask'] is not None:
            denoise_mask = extra_args['denoise_mask']
            z_prime = z_prime * denoise_mask

        # Initialize x with the first z_prime (i.e. the noisiest z_prime).
        x[0] = z_prime[0]

        extra_args = {} if extra_args is None else extra_args
        
        return self.sample(model, x, sigmas[start_sigma:end_sigma],
                           z_prime, c1, blend, extra_args=extra_args,
                           callback=callback,
                           disable=disable, s_churn=s_churn,
                           s_tmin=s_tmin, s_tmax=s_tmax, **kwargs)
    
    @torch.no_grad()
    def __call__(self, model, x, sigmas, extra_args=None, callback=None,
                 disable=None, noise_sampler=None,
                 model_node=None,
                 alpha_1=None,
                 blending_schedule=None,
                 blending_function=None,
                 normalize_on_mean=None,
                 clamp_blending_at_pct=None,
                 start_blending_at_pct=None,
                 stop_blending_at_pct=None,
                 blend_min=0.0,
                 blend_max=1.0,
                 rewind=False,
                 rewind_min=0.0,
                 rewind_max=0.999,
                 s_churn=0., s_tmin=0., s_tmax=float('inf'),
                 **kwargs):
        """
        Prepare the iterative mixing variables the derived classes will need.
        Each sampler's implementation is in a sample function that is not implemented
        in the base class.
        """

        # Deprecation warning:
        if normalize_on_mean:
            logging.warning("normalize_on_mean is deprecated (and does nothing)")

        # Fail if the batch size of x is greater than 1. We currently
        # cannot handle that situation.
        if x.shape[0] > 1:
            raise ValueError("cannot handle batches of latents currently; sorry")

        # Calculate some values we will need to generate the blending schedule
        # and the noised latent sequence later.
        steps = len(sigmas) - 1
        start_blending_at_step = 0 if start_blending_at_pct is None else int(start_blending_at_pct * steps)
        stop_blending_at_step = steps if stop_blending_at_pct is None else int(stop_blending_at_pct * steps)

        # Get the c1 blending schedule as per DemoFusion paper.
        c1 = get_blending_schedule(range(steps),
                                blending_schedule, blending_function=blending_function,
                                alpha_1=alpha_1,
                                clamp_blending_at_pct=clamp_blending_at_pct,
                                start_blending_at_step=start_blending_at_step,
                                stop_blending_at_step=stop_blending_at_step,
                                blend_min=blend_min,
                                blend_max=blend_max)

        # Get the right latent blending function (slerp, additive, etc.)
        blend = get_blending_function(blending_function)

        # If rewind mode is enabled, loop through a list of step ranges
        # denoising, noising again, denoising, etc..
        if rewind == True:
            if rewind_min >= rewind_max:
                raise ValueError("rewind_min must be less than rewind_max")
            elif rewind_max > 1.0 or rewind_min > 1.0:
                raise ValueError("rewind_max and rewind_min cannot exceed 1.0")
            elif rewind_min < 0.0 or rewind_max < 0.0:
                raise ValueError("rewind_min and rewind_min cannot be less than 0.0")
            ranges = geometric_ranges(0, steps, rewind_min, max_start=int(steps * rewind_max))
        else:
            ranges = [(0, steps)]

        # Loop over the step ranges, sampling, rewinding, etc..
        # If rewind is not enabled, the ranges just contain the usual
        # range of steps called for.
        for start_sigma, end_sigma in ranges:
            _trace(f"rewind: {start_sigma}:{end_sigma}, len(c1)={len(c1)}")
            x = self.run(model, x, sigmas, c1=c1,
                                blend=blend, extra_args=extra_args, callback=callback,
                                disable=disable,
                                normalize_on_mean=None,
                                start_sigma=start_sigma,
                                end_sigma=end_sigma + 1,
                                s_churn=0., s_tmin=0., s_tmax=float('inf'),
                                **kwargs)

        return x
    
    @torch.no_grad()
    def sample(self, model, x, sigmas, z_prime, c1, blend, *args, **kwargs):
        raise NotImplementedError("this method must be implemented in derived classes")

class IterativeMixingEulerSamplerImpl(IterativeMixingSampler):
    """
    Implements Algorithm 2 (Euler steps) from Karras et al. (2022)
    but with the "skip residuals" mixing of noised latents suggested
    in the DemoFusion paper from Du, Chang et al. (2023).
    """

    _argname = "euler"

    @torch.no_grad()
    def sample(self, model, x, sigmas, z_prime, c1, blend, extra_args=None, callback=None,
               disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
               *args, **kwargs):
        # Graft the model onto our custom object that can track things.
        our_model = ModularKSamplerX0Inpaint(model)
        denoise_mask = extra_args.get('denoise_mask')

        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)

            # Prepare x for diffusion with the denoise_mask.
            # This applies the denoise mask to x.
            masked_x, masked_zp = our_model.before(x, z_prime[i], sigmas[i], denoise_mask)

            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                masked_x = masked_x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            
            denoised = our_model(masked_x, sigma_hat * s_in, **extra_args)
            d = comfy.k_diffusion.sampling.to_d(x, sigma_hat, denoised)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            dt = sigmas[i + 1] - sigma_hat

            # Euler method
            x = masked_x + d * dt

            # Mix x with z_prime being sure to give z_prime[i] a batch dimension.
            _trace(f"i={i},masked_zp={masked_zp.shape}, x={x.shape}, len(c1)={len(c1)}")
            x = blend(masked_zp.unsqueeze(0), x, c1[i])

            # Reapply the denoise mask now that we are done.
            x = our_model.after(x, denoise_mask)

        return x
    
def mask_to_latent(mask: torch.Tensor) -> torch.Tensor:
    """
    Masks have shape [B, H, W, 1] whereas latents have shape
    [B, 4, H, W]. Let's convert a mask batch into a latent batch.
    """

    # Make a copy because we'll be changing this baby.
    latent = mask.clone()

    if len(latent.shape) < 3:
        raise ValueError("expecting a mask tensor with at least 3 dimensions (H, W, C)")

    # If the mask lacks a batch dimension, add a single batch dimension.
    if len(latent.shape) == 3:
        latent = latent.unsqueeze(0)

    # Repeat the intensity channel across all the channels of the latent channel dim.
    # This reshapes the mask to [B, H, W, 4]
    latent = latent.repeat(1, 1, 1, 4)

    # Transpose to match x2 shape: [B, 4, H, W]
    latent = latent.permute(0, 3, 1, 2)  # Shape becomes [B, 4, H, W]

    return latent

@torch.no_grad()
def reshape_denoise_mask(denoise_mask: torch.Tensor):
    "Reshape a denoise_mask into the correct shape for use with the model."

    if denoise_mask.dim() == 3:
        if denoise_mask.shape[2] == 1:
            # Convert the singleton mask dimension into
            # a 4-wide dimension by repeating the channel magnitudes 4 times.
            denoise_mask = denoise_mask.unsqueeze(3) 
            denoise_mask = denoise_mask.repeat(1, 1, 4, 1)
            denoise_mask = denoise_mask.permute(3, 2, 0, 1).\
                reshape(1, 4, denoise_mask.shape[0], denoise_mask.shape[1])
        else:
            raise ValueError("denoise mask has 3 dimensions but shape[2] != 1")
    elif denoise_mask.dim() != 4:
        raise ValueError(f"denoise_mask has invalid shape {denoise_mask.shape}")

    return denoise_mask

@torch.no_grad()
def merge_denoise_masks(denoise_mask, extra_mask):
    "Reshape a denoise mask to make it easier to work with our Perlin masks that have a funny shape"

    # If the extra_mask was provided, reshape it.
    extra_mask = reshape_denoise_mask(extra_mask) if extra_mask is not None else None

    # If there is no default denoise_mask, but there is an extra_mask, then
    # reshape it and return it. No merging required.
    if denoise_mask is None:
        if extra_mask is not None:
            return extra_mask
        else:
            return None
    else:
        if extra_mask is not None:
            # Multiply the masks element-wise along just one of the four
            # channels (all channels are the same anyway).
            combined_mask = denoise_mask[:, 0, :, :] * extra_mask[:, 0, :, :]

            # Clamp the values to be between 0.0 and 1.0
            combined_mask_clamped = torch.clamp(combined_mask, 0.0, 1.0)

            # If needed, expand the result back to [B, C, H, W]
            combined_mask_expanded = combined_mask_clamped.unsqueeze(1).expand_as(denoise_mask)
            return combined_mask_expanded
        else:
            # Otherwise, just return the denoise_mask by itself.
            return denoise_mask

    

class IterativeMixingPerlinEulerSamplerImpl(IterativeMixingSampler):
    """
    Implements Algorithm 2 (Euler steps) from Karras et al. (2022)
    but with the "skip residuals" mixing of noised latents suggested
    in the DemoFusion paper from Du, Chang et al. (2023). Also, we
    add some perlin noise masked sampling with mixing at each step.
    """

    _argname = "euler_perlin"

    @torch.no_grad()
    def normalized_noise_batch(self, x, noise, dimensions=[0, 1, 2, 3]):
        # Now match the noise by channel based on the statistics of x.
        x_means = x.mean(dim=dimensions, keepdim=True)
        x_stds = x.std(dim=dimensions, keepdim=True)
        latents_means = noise.mean(dim=dimensions, keepdim=True)
        latents_stds  = noise.std(dim=dimensions, keepdim=True)

        normalized_latents = (noise - latents_means) / (latents_stds + 0.00001)
        normalized_latents = normalized_latents * x_stds + x_means

        return normalized_latents

    @torch.no_grad()
    def get_masks(self, mode: str, loop_count: int, C: int, W: int, H: int, x: torch.Tensor, seed: int, scale: float):
        # In direct mode, we generate Perlin latents to merge.
        if mode == "latents":
            # Generate a batch of grayscale perlin noise and rearrange the tensor
            # to have the same dimension structure as Comfy latents.
            latents = perlin_masks(loop_count, W, H, device=x.device,
                                   seed=seed, scale=scale).view(loop_count, 1, H, W).\
                                   repeat_interleave(4, dim=1)

            return latents
        elif mode == "masks" or mode == "matched_noise":
            masks = perlin_masks(loop_count, W, H, device=x.device,
                                   seed=seed, scale=scale)
            return masks
        else:
            raise ValueError(f"Invalid mode {mode}; expecting 'latents' or 'masks'")

    @torch.no_grad()
    def sample(self, model, x, sigmas, z_prime, c1, blend, extra_args=None, callback=None,
               disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
               seed: int=None, perlin_mode="latents", perlin_strength=0.75, perlin_scale=10.,
               *args, **kwargs):
        if len(x.shape) != 4:
            raise ValueError("latent x must be 4D")
        
        [B, C, H, W] = x.shape

        # Extreme hackery: We use our own monkey-patched model object derived from
        # the KSamplerX0Inpaint class so that we can do things differently w.r.t masking.
        our_model = ModularKSamplerX0Inpaint(model)

        # denoise_mask is in extra_args and it has the typical shape of a latent image
        # including 4 channels (not one, as you might expect):
        # denoise_mask=torch.Size([1, 4, 128, 128])

        if B != 1:
            raise ValueError("this sampler cannot deal with batches of more than 1 latent yet; sorry")

        s_in = x.new_ones([x.shape[0]])
        loop_count = len(sigmas) - 1
        seed = extra_args.get("seed") or 0

        z_prime_std = z_prime.std(dim=(1, 2, 3), unbiased=True)
        z_prime_mean = z_prime.mean(dim=(1, 2, 3))

        @torch.no_grad()
        def denoise_step(inner_x, inner_i, inner_s_in, extra_denoise_mask=None):
            "Turn one crank of Euler against an arbitrary latent or partial latent represented by inner_x"
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[inner_i] <= s_tmax else 0.
            sigma_hat = sigmas[inner_i] * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(inner_x) * s_noise
                inner_x = inner_x + eps * (sigma_hat ** 2 - sigmas[inner_i] ** 2) ** 0.5
            denoised = our_model(inner_x, sigma_hat * inner_s_in, **extra_args)
            d = comfy.k_diffusion.sampling.to_d(inner_x, sigma_hat, denoised)
            dt = sigmas[inner_i + 1] - sigma_hat

            return inner_x + d * dt, sigma_hat, denoised

        # Generate a batch of perlin noise. Depending on the perlin_mode,
        # this function either gives us a batch of perlin masks or a batch
        # of perlin latents that we will apply below.        
        perlin_tensors = self.get_masks(perlin_mode, loop_count, C, W, H, x,
                                        seed=seed, scale=perlin_scale)

        # Create a blending schedule for the perlin noise that is scaled
        # via the perlin_strength parameter. c1_perlin will range between
        # perlin_strength and 1.0.
        c1_perlin = 1.0 + perlin_strength * (c1 - 1.0)

        for i in trange(loop_count, disable=disable):
            # Save any existing denoise mask and fetch the masked versions of x and z_prime
            # based on that denoise mask.
            denoise_mask = extra_args.get("denoise_mask")
            masked_x, masked_zp = our_model.before(x, z_prime[i], sigmas[i], denoise_mask)
            
            # Do a single denoise step of the (potentially masked) latent.
            x, sigma_hat, denoised = denoise_step(masked_x, i, s_in)

            # Blend x with the z_prime for this step (masked if needed).
            x = blend(masked_zp.unsqueeze(0), x, c1[i])
            
            # Do an additional step of denoising while applying a perlin mask if desired.
            if perlin_mode == "latents":
                # Instead of being greyscale, the latents option fills all four channels
                # for a remarkably different effect.
                perlin_noise = perlin_tensors[i].unsqueeze(0)

                # Mix in constant perlin noise at every step.
                x = x * c1_perlin[i] + perlin_noise * (1. - c1_perlin[i]) * x
            elif perlin_mode == "masks" and i < (loop_count - 1):
                # Apply a denoise mask to x based on perlin noise. All our perlin noise samples
                # are random, so it doesn't matter which perlin batch index we draw from.
                # Then denoise x with the perlin mask applied and blend that masked denoise
                # back with the original x.
                perlin_denoise_mask = perlin_tensors[i].permute(2, 0, 1).unsqueeze(0)

                # We denoise at the _next_ timestep. I'm not sure whether this is correct.
                x_perlin, sigma_hat, denoised = denoise_step(x, i + 1, s_in)

                # Take this denoised latent and blend it in with the previously denoised latent
                # applying the perlin mask so that a declining share of the denoised part
                # gets blended in as c1_perlin rises from 0.0 to 1.0 * perlin_strength as per its schedule.
                x = x * (1. - perlin_denoise_mask) + (x * c1_perlin[i+1] + x_perlin * (1. - c1_perlin[i+1])) * perlin_denoise_mask
            elif perlin_mode == "matched_noise":
                # Blend in the perlin noise directly.
                perlin_noise = perlin_tensors[i].permute(2, 0, 1).unsqueeze(0)
                x = (x * c1_perlin[i] + perlin_noise * (1. - c1_perlin[i]))

            # Apply the denoise mask after sampling. This restores the original latent
            # to the area outside the denoising mask.
            x = our_model.after(x, denoise_mask)

            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        return x
    
class IterativeMixingLCMSamplerImpl(IterativeMixingSampler):
    """
    Implements Latent Consistency Model sampling.
    """

    _argname = "lcm"

    @torch.no_grad()
    def sample(self, model, x, sigmas, z_prime, c1, blend, extra_args=None, callback=None,
               disable=None, noise_sampler=None, *args, **kwargs):
        extra_args = {} if extra_args is None else extra_args
        noise_sampler = comfy.k_diffusion.sampling.default_noise_sampler(x) if noise_sampler is None else noise_sampler
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

            x = denoised

            # Mix x with z_prime being sure to give z_prime[i] a batch dimension.
            # I have no idea whether this is the right place to do the mixing.
            # So far, LCM results are bad.
            x = blend(z_prime[i].unsqueeze(0), x, c1[i])

            if sigmas[i + 1] > 0:
                x += sigmas[i + 1] * noise_sampler(sigmas[i], sigmas[i + 1])
            
        return x
    
class IterativeMixingDPM2MImpl(IterativeMixingSampler):
    """
    Implements sample_dpmpp_2m with iterative latent mixing.
    """

    _argname = "dpmpp_2m"

    @torch.no_grad()
    def sample(self, model, x, sigmas, z_prime, c1, blend, extra_args=None, callback=None,
                        disable=None, *args, **kwargs):
        """DPM-Solver++(2M)."""
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        old_denoised = None

        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t
            if old_denoised is None or sigmas[i + 1] == 0:
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d

            # Run latent mixing. Is this the right spot?
            x = blend(z_prime[i].unsqueeze(0), x, c1[i])

            old_denoised = denoised
        return x
    
class BlendingFunction(ABC):
    """
    Base class for blending functions. The __call__method dispatches latent blending
    to an appropriate subclass.
    """

    @torch.no_grad()
    def __call__(self, l1: torch.Tensor, l2: torch.Tensor, weight: float)  -> torch.Tensor:
        if l1.device != l2.device:
            raise ValueError(f"BlendingFunction: l1 and l2 on different devices ({l1.device} != {l2.device})")
        return self.blend(l1, l2, weight)

    def blend(self, l1: torch.Tensor, l2: torch.Tensor, weight: float)  -> torch.Tensor:
        raise NotImplementedError("this method must be implemented in derived classes")
    
class AdditiveBlendingFunction(BlendingFunction):
    """
    The simplest latent blending strategy is to just add them together.
    This is the method used in the DemoFusion paper. Assuming c1 represents
    a value from a rising blending schedule, we expect that l1 is z_prime and
    l2 is z_hat.
    """
    _argname = "addition"

    @torch.no_grad()
    def blend(self, l1: torch.Tensor, l2: torch.Tensor, weight: float, bias: float=0.0)  -> torch.Tensor:
        weight += bias
        return (1 - weight) * l1 + weight * l2
    
class NormOnlyBlendingFunction(BlendingFunction):
    """
    This function computes a new vector that is a blend of two latents, where the direction of l1
    is partially aligned towards the direction of l2. The extent of this alignment is controlled
    by the weight parameter.
    """
    _argname = "norm_only"

    @torch.no_grad()
    def blend(self, l1: torch.Tensor, l2: torch.Tensor, weight: float)  -> torch.Tensor:
        dims = l1.shape

        # We need at least two dimensions (batch plus whatever else)
        assert len(dims) >= 2

        # Reshape to flatten everything but the batch dimension.
        l1 = l1.reshape(dims[0], -1)
        l2 = l2.reshape(dims[0], -1)

        # Normalize l2 to get its direction
        l2_normalized = l2 / l2.norm(dim=1, keepdim=True)

        # Calculate the magnitude of v1
        l1_magnitude = l1.norm(dim=1, keepdim=True)

        # Scale the normalized v2 by the magnitude of v1 and by alpha
        result = l1 + weight * (l2_normalized * l1_magnitude - l1)

        # Reshape back to the original dimensions.
        result = result.reshape(dims)
        return result

class SLERPBlendingFunction(BlendingFunction):
    """
    The simplest latent blending strategy is to just add them together.
    This is the method used in the DemoFusion paper. Assuming c1 represents
    a value from a rising blending schedule, we expect that l1 is z_prime and
    l2 is z_hat.
    """
    _argname = "slerp"

    @torch.no_grad()
    def blend(self, l1: torch.Tensor, l2: torch.Tensor, weight: float)  -> torch.Tensor:
        """
        Perform Spherical Linear Interpolation (SLERP) between two tensors.
        
        Args:
        l1 (torch.Tensor): The first latent tensor.
        l2 (torch.Tensor): The second latent tensor.
        weight (float): The interpolation factor, typically between 0 and 1.

        Returns:
        torch.Tensor: The interpolated tensor.
        """
        if l1.shape != l2.shape:
            raise ValueError(f"cannot slerp two latents with different shapes: {l1.shape} != {l2.shape}")

        result = slerp(l1, l2, weight)
        return result
    
class BlendingSchedule(ABC):
    """
    Base class for blending schedules. The __call__method runs various centralized
    things and then dispatches to a derived class to generate the blending schedule.
    """

    @torch.no_grad()
    def __call__(self, indices: list[int], blend_min=0.0, blend_max=1.0, **kwargs) -> torch.Tensor:
        # Intercept the clamp_above_pct parameter here and delete it
        # because the individual schedule functions do not need it.
        schedule = self.blend(indices, **kwargs)

        # If configured, clamp the schedule to 1.0 for steps above
        # the given index.
        clamp_above_pct = _safe_get(kwargs, "clamp_blending_at_pct", float)
        if clamp_above_pct is not None:
            clamp_above = int(clamp_above_pct * len(schedule))
            schedule[clamp_above:] = 1.0

        # Rescale the y-extents of the schedule between the min and max range.
        schedule = torch.lerp(torch.full_like(schedule, blend_min),
                              torch.full_like(schedule, blend_max),
                              schedule)

        return schedule

    def blend(self, indices: list[int], **kwargs) -> torch.Tensor:
        raise NotImplementedError("this method must be implemented in derived classes")

class CosineBlendingSchedule(BlendingSchedule):
    _argname = "cosine"

    @torch.no_grad()
    def blend(self, indices: list[int], **kwargs) -> torch.Tensor:
        """
        Define a tensor representing the constant c1 from the DemoFusion paper.
        """

        stop_blending_at_step = _safe_get(kwargs, "stop_blending_at_step", int)
        alpha_1 = _safe_get(kwargs, "alpha_1", float)

        steps = indices[-1] + 1

        if stop_blending_at_step is not None:
            steps = stop_blending_at_step
        
        t = torch.tensor(indices)

        # Calculate cosine_factor with adjusted step size
        cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (steps - t) / steps))

        # Apply the alpha exponentiation
        c1 = cosine_factor ** alpha_1

        # If you set stop_blending_at_step, then we have to clamp
        # the remaining values to 1.0. This line has no effect if
        # we are blending out to the full number of steps in the indices
        # parameter.
        c1 = torch.where(t > steps, torch.tensor(1.0), c1)

        return c1
    
class LinearBlendingSchedule(BlendingSchedule):
    _argname = "linear"

    @torch.no_grad()
    def blend(self, indices: list[int], **kwargs) -> torch.Tensor:
        """
        Define a linear tensor from 0 to 1 across the given indices.
        Yes this could just be a one-liner, but I'm putting it in a
        function. This is an alternative blending schedule to the cosine
        schedule to see what differs in the output.
        """

        steps = indices[-1] + 1
        start_blending_at_step = _safe_get(kwargs, "start_blending_at_step", int) or 0
        stop_blending_at_step = _safe_get(kwargs, "stop_blending_at_step", int) or steps

        if stop_blending_at_step is not None:
            steps = stop_blending_at_step        

        t = torch.tensor(indices)
        linear_schedule = t / steps

        linear_schedule = torch.where(t > steps,                  torch.tensor(1.0), linear_schedule)
        linear_schedule = torch.where(t < start_blending_at_step, torch.tensor(0.0), linear_schedule)
        
        return linear_schedule

class LogisticBlendingSchedule(BlendingSchedule):
    _argname = "logistic"

    @torch.no_grad()
    def blend(self, indices: list[int], **kwargs) -> torch.Tensor:
        """
        Generate a logistic function ranging between x=[0,1] given step
        indices that range between any two extreems. The logistic will be guaranteed
        to hit x=1 at the max step value in the indices.

        Parameters:
        indices (torch.Tensor): 1D tensor of x-coordinates.
        start_blending_at_step (int): the starting step for the blending curve.
        stop_blending_at_step (int): the ending step for the blending curve.

        Returns:
        torch.Tensor: Output of the logistic function.
        """
        indices = torch.tensor(indices)
        max_step = torch.max(indices)
        alpha_1 = _safe_get(kwargs, "alpha_1", float)
        steepness = alpha_1 * 5 # arbitrary
        start_blending_at_step = _safe_get(kwargs, "start_blending_at_step", int) or 0
        stop_blending_at_step = _safe_get(kwargs, "stop_blending_at_step", int) or steps

        if max_step == 0:
            raise ValueError("the maximum blending step cannot be zero")
        
        midpoint = (((stop_blending_at_step or max_step) + (start_blending_at_step)) / 2) / max_step

        # Adjust the step indices such that the curve ranges over x from 0.0 to 1.0.
        indices = indices.float() / max_step

        # Apply the logistic function
        logistic_output = 1 / (1 + torch.exp(-steepness * (indices - midpoint)))

        # Clamp the output to 1.0 for x-values above the specified threshold
        if stop_blending_at_step is not None:
            x = stop_blending_at_step / max_step
            logistic_output = torch.where(indices > x, torch.tensor(1.0), logistic_output)

        if start_blending_at_step is not None:
            x = start_blending_at_step / max_step
            logistic_output = torch.where(indices < x, torch.tensor(0.0), logistic_output)

        return logistic_output

SAMPLERS_MAP = generate_class_map(IterativeMixingSampler, "_argname")
BLENDING_FUNCTION_MAP = generate_class_map(BlendingFunction, "_argname")
BLENDING_SCHEDULE_MAP = generate_class_map(BlendingSchedule, "_argname")
