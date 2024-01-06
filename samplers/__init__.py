from abc import ABC
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.samplers
import comfy.k_diffusion.sampling
import torch
from tqdm.auto import trange

from ..utils import generate_class_map, generate_noised_latents, slerp
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

    @torch.no_grad
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
                 s_churn=0., s_tmin=0., s_tmax=float('inf'),
                 **kwargs):
        """
        Prepare the iterative mixing variables the derived classes will need.
        Each sampler's implementation is in a sample function that is not implemented
        in the base class.
        """

        # Fail if the batch size of x is greater than 1. We currently
        # cannot handle that situation.
        if x.shape[0] > 1:
            raise ValueError("cannot handle batches of latents currently; sorry")

        # Generate a batch of progressively noised latents according
        # to the sigmas (noise schedule).
        z_prime = generate_noised_latents(x, sigmas, normalize=normalize_on_mean)

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

        # Get a latent blending function (slerp, additive, etc.)
        blend = get_blending_function(blending_function)

        # Initialize x with the first z_prime (i.e. the noisiest z_prime).
        # Note, we are assuming that X has batch count 1.
        x[0] = z_prime[0]

        extra_args = {} if extra_args is None else extra_args
        
        return self.sample(model, x, sigmas, z_prime, c1, blend, extra_args=extra_args, callback=callback,
                    disable=disable, s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, **kwargs)
    
    @torch.no_grad
    def sample(self, model, x, sigmas, z_prime, c1, blend, *args, **kwargs):
        raise NotImplementedError("this method must be implemented in derived classes")

class IterativeMixingEulerSamplerImpl(IterativeMixingSampler):
    """
    Implements Algorithm 2 (Euler steps) from Karras et al. (2022)
    but with the "skip residuals" mixing of noised latents suggested
    in the DemoFusion paper from Du, Chang et al. (2023).
    """

    _argname = "euler"

    @torch.no_grad
    def sample(self, model, x, sigmas, z_prime, c1, blend, extra_args=None, callback=None,
               disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
               *args, **kwargs):
        s_in = x.new_ones([x.shape[0]])
        for i in trange(len(sigmas) - 1, disable=disable):
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
            denoised = model(x, sigma_hat * s_in, **extra_args)
            d = comfy.k_diffusion.sampling.to_d(x, sigma_hat, denoised)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
            dt = sigmas[i + 1] - sigma_hat
            # Euler method
            x = x + d * dt

            # Mix x with z_prime being sure to give z_prime[i] a batch dimension.
            x = blend(z_prime[i].unsqueeze(0), x, c1[i])

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

class IterativeMixingPerlinEulerSamplerImpl(IterativeMixingSampler):
    """
    Implements Algorithm 2 (Euler steps) from Karras et al. (2022)
    but with the "skip residuals" mixing of noised latents suggested
    in the DemoFusion paper from Du, Chang et al. (2023). Also, we
    add some perlin noise masked sampling with mixing at each step.
    """

    _argname = "euler_perlin"

    @torch.no_grad
    def get_masks(self, mode: str, loop_count: int, C: int, W: int, H: int, device, seed: int, scale: float):
        # In direct mode, we generate Perlin latents to merge.
        if mode == "latents":
            # Generate C x loop_count greyscale perlin masks and then permute and rearrange
            # this into a batch of loop_count C channel latents.
            # noise.perlin_masks returns [B*C, H, W, 1]
            latentsx4 = perlin_masks(loop_count * C, W, H, device=device,
                                           seed=seed, scale=scale)

            # But we need [B, C, H, W], which we can accomplish with view()
            latents = latentsx4.view(loop_count, 4, H, W)
            return latents
        elif mode == "masks":
            return perlin_masks(loop_count, W, H, device=device,
                                   seed=seed, scale=scale)
        elif mode == "matched_noise":
            # XXX: Not the best.
            return None
        else:
            raise ValueError(f"Invalid mode {mode}; expecting 'latents' or 'masks'")

    @torch.no_grad
    def sample(self, model, x, sigmas, z_prime, c1, blend, extra_args=None, callback=None,
               disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.,
               seed: int=None, perlin_mode="latents", perlin_strength=0.75, perlin_scale=10.,
               mixing_masks=None,
               *args, **kwargs):
        if len(x.shape) != 4:
            raise ValueError("latent x must be 4D")
        
        [B, C, H, W] = x.shape

        if B != 1:
            raise ValueError("this sampler cannot deal with batches of more than 1 latent; sorry")

        s_in = x.new_ones([x.shape[0]])
        loop_count = len(sigmas) - 1
        seed = extra_args.get("seed") or 0

        z_prime_std = z_prime.std(dim=(1, 2, 3), unbiased=True)
        z_prime_mean = z_prime.mean(dim=(1, 2, 3))

        @torch.no_grad
        def denoise_step(inner_x, inner_i, inner_s_in, denoise_mask=None):
            "Turn one crank of Euler against an arbitrary latent or partial latent represented by inner_x"
            gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[inner_i] <= s_tmax else 0.
            sigma_hat = sigmas[inner_i] * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(inner_x) * s_noise
                inner_x = inner_x + eps * (sigma_hat ** 2 - sigmas[inner_i] ** 2) ** 0.5
            denoised = model(inner_x, sigma_hat * inner_s_in, **extra_args)
            d = comfy.k_diffusion.sampling.to_d(inner_x, sigma_hat, denoised)
            dt = sigmas[inner_i + 1] - sigma_hat
            return inner_x + d * dt, sigma_hat, denoised

        if mixing_masks is not None:
            perlin_tensors = mixing_masks
        else:
            # Generate a batch of perlin noise. Depending on the perlin_mode,
            # this function either gives us a batch of perlin masks or a batch
            # of perlin latents that we will apply below.        
            perlin_tensors = self.get_masks(perlin_mode, loop_count, C, W, H, x.device,
                                            seed=seed, scale=perlin_scale)

        # Create a blending schedule for the perlin noise that is scaled
        # via the perlin_strength parameter. c1_perlin will range between
        # c1_perlin and 1.0.
        c1_perlin = 1.0 + perlin_strength * (c1 - 1.0)

        for i in trange(loop_count, disable=disable):
            x, sigma_hat, denoised = denoise_step(x, i, s_in)
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

            # Mix x with z_prime being sure to give z_prime[i] a batch dimension.
            x = blend(z_prime[i].unsqueeze(0), x, c1[i])

            # Now apply the perlin mask for this step and get the model to denoise again
            # but without the blending of z_prime.
            #x_2, sigma_hat, denoised = denoise_step(x, i, s_in, denoise_mask=perlin_masks[i])
            
            if perlin_mode == "latents":
                x = perlin_tensors[i].unsqueeze(0) * perlin_strength + (1. - perlin_strength) * x
            elif perlin_mode == "masks":
                x, sigma_hat, denoised = denoise_step(x, i, s_in, denoise_mask=perlin_tensors[i] * (float(1.0) - c1_perlin[i]))
            elif perlin_mode == "matched_noise":
                matched_noise = torch.randn(C, H, W, device=z_prime_std.device) * z_prime_std[i] + z_prime_mean[i]
                x = matched_noise * (1. - c1_perlin[i]) + x * c1_perlin[i]

        if perlin_mode == "matched_noise":
            x = (x - x.mean())
        return x
    
class IterativeMixingLCMSamplerImpl(IterativeMixingSampler):
    """
    Implements Latent Consistency Model sampling.
    """

    _argname = "lcm"

    @torch.no_grad
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

    @torch.no_grad
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

    @torch.no_grad
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

    @torch.no_grad
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

    @torch.no_grad
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

    @torch.no_grad
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

    @torch.no_grad
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

    @torch.no_grad
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

    @torch.no_grad
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
