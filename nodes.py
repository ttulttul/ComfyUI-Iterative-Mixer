import math
import torch
import comfy.model_management
import comfy.sample
import comfy.utils
import comfy.samplers
import logging as logger

def calc_sigmas(model, sampler_name, scheduler, steps, start_at_step, end_at_step):
        """
        Copied with gratitude from [ComfyUI_Noise](https://github.com/BlenderNeko/ComfyUI_Noise/blob/master/nodes.py)
        """
        device = comfy.model_management.get_torch_device()
        end_at_step = min(steps, end_at_step)
        start_at_step = min(start_at_step, end_at_step)
        real_model = None
        comfy.model_management.load_model_gpu(model)
        real_model = model.model
        sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas[start_at_step:end_at_step] \
            / model.model.latent_format.scale_factor
        
        return sigmas

def generate_noised_latents(x, sigmas):
    """
    Generate all noised latents for a given initial latent image and sigmas in parallel.

    :param x: Original latent image as a PyTorch tensor.
    :param sigmas: Array of sigma values for each timestep as a PyTorch tensor.
    :return: A tensor containing all noised latents for each timestep.
    """
    # Ensure that x and sigmas are on the same device (e.g., CPU or CUDA)
    device = x.device
    sigmas = sigmas.to(device) # ignore the first sigma
    batch_size = x.shape[0]
    num_sigmas = len(sigmas)

    # Expand x and sigmas to match each other in the first dimension
    # x_expanded shape will be:
    # [batch_size * num_sigmas, channels, height, width]
    x_expanded = x.repeat(num_sigmas, 1, 1, 1)
    sigmas_expanded = sigmas.repeat_interleave(batch_size)

    # Create a noise tensor with the same shape as x_expanded
    noise = torch.randn_like(x_expanded)

    # Multiply noise by sigmas, reshaped for broadcasting
    noised_latents = x_expanded + noise * sigmas_expanded.view(-1, 1, 1, 1)

    return noised_latents


class BatchUnsampler:
    """
    Unsample a latent step by step back to the start of the noise schedule.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "steps": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 1, "max": 10000}),
                    "latent_image": ("LATENT", ),
                }}
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    FUNCTION = "unsampler"

    CATEGORY = "tests"
    
    def unsampler(self, model, sampler_name, scheduler, steps,
                  start_at_step, end_at_step, latent_image):
        """
        Generate a batch of latents representing each z[i] in the
        progressively noised sequence of latents stemming from the
        source latent_image, using the model's noising schedule (sigma)
        in reverse and applying normal noise at each step in the manner
        prescribed by the original latent diffusion paper.
        """
        latent = latent_image
        latent_image = latent["samples"]

        # Calculate the sigmas for this model, sampler, scheduler, etc.
        sigmas = calc_sigmas(model, sampler_name, scheduler, steps,
                            start_at_step, end_at_step)

        # Flip the sigmas (the sampling schedule) in reverse so that the sampler
        # will instead "unsample" the latent, adding noise rather than
        # removing noise. I think we add a small epsilon value to prevent
        # division by zero, but I'm not really sure. I got this code from
        # BlenderNeko's noise node.
        sigmas = sigmas.flip(0)

        # Generate a batch of progressively noised latents according
        # to the reversed sigmas (noise schedule).
        z = generate_noised_latents(latent_image, sigmas)

        # Return the batch of progressively noised latents.
        out = {"samples": z}
        return (out,)

def get_blending_schedule(indices, alpha_1):
    """
    Define a tensor representing the constant c1 from the DemoFusion paper.
    I have found that values between 0.1 and 1.0 deliver reasonable results.
    If you go above 1.0 things get noisy.
    """
    # The final item in indices is the last index of a latent batch.
    # We add one to this to get the number of steps.
    steps = indices[-1] + 1
    t = torch.tensor(indices)

    logger.warning(f"get_blending_schedule: steps={steps}, t.shape={t.shape}")

    # Calculate c1 using the code borrowed from the author's repository.
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * (steps - t) / steps))

    c1 = cosine_factor ** alpha_1
    return c1

def iterative_mixing_ksampler(model, seed, cfg, sampler_name, scheduler, step_increment, positive, negative, latent_image_batch, denoise=1.0, disable_noise=False, force_full_denoise=False, c1=None, alpha_1=0.5, reverse_input_batch=True):
    # The step_increment cannot be zero.
    assert(step_increment != 0)

    z_primes = latent_image_batch["samples"]

    # You will almost always want to reverse the input batch, which
    # we assume is a set of progressively noisier latents. For de-noising,
    # this batch order needs to be reversed so that we're going from
    # the noisiest latent back to the start.
    if reverse_input_batch == True:
        z_primes = torch.flip(z_primes, [0])

    if disable_noise:
        noise = torch.zeros(z_primes.size(), dtype=z_primes.dtype, layout=z_primes.layout, device="cpu")
    else:
        batch_inds = latent_image_batch["batch_index"] if "batch_index" in latent_image_batch else None
        noise = comfy.sample.prepare_noise(z_primes, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent_image_batch:
        noise_mask = latent_image_batch["noise_mask"]

    # Assume that the first image in the batch is the most noised,
    # representing z_prime[T] from the DemoFusion paper. You may
    # need to reverse the batch of latents to ensure that the first
    # latent in the batch is the most noised.

    # Set up a tensor to receive the samples from the de-noising process.
    # It has the same shape as z_primes and the first element is
    # the first element of z_primes. We do some fancy footwork here
    # to give z_out the right batch count given the step_increment
    # parameter.

    # Define the indices of z_prime that we will be visiting.
    # We start with z_prime[0 + step_increment] and go up in steps of
    # step_increment. We make sure we visit z_prime[-1] if the step_increment
    # is such that we would otherwise end short of it.
    steps = z_primes.shape[0] # batch size gives us the de-noising step count
    zp_indices = list(range(step_increment - 1, steps, step_increment))
    if zp_indices[-1] != steps - 1:
        zp_indices.append(steps - 1)

    loop_count = len(zp_indices)

    # z_out is going to store z_prime[0] plus all the de-noised
    # weighted mixtures of denoise(z_prime[i], z_hat[i-1]).
    # Therefore, we give it a length that is one longer than
    # the loop count because the first entry is set to z_prime[0].
    # z_out has the same channels, width, and height as z_prime.
    z_out = torch.zeros(loop_count + 1, *z_primes.size()[1:])
    z_out = z_out.to(z_primes.device)

    # We also return the intermediate samples for analysis.
    # This tensor has the same shape exactly as z_out but the
    # first entry is left all zeroes.
    samples_out = torch.zeros_like(z_out)
    samples_out = samples_out.to(z_primes.device)

    # The first output value is the first z_prime latent.
    # Implicitly, samples_out[0] is just zeroes.
    z_out[0] = z_primes[0]

    # The first z_prime that we will use in the de-noising loop below
    # is z_primes[0] unsqueezed to remove the batch dimension.
    z_i = z_primes[0].unsqueeze(0)

    # Get the blending parameter from the DemoFusion paper.
    if c1 is None:
        c1 = get_blending_schedule(zp_indices, alpha_1=alpha_1)

    # Move the blending schedule tensor to the same device as our
    # latents.
    c1 = c1.to(z_primes.device)

    # The paper suggests that we de-noise the image step by step, blending
    # in samples from the noised z_hat tensor along the way according to 
    # a blending schedule given by the c1 tensor. Each successively de-noised
    # sample z[i] is given by this formula:
    #
    # z[i] = denoise(z[i-1]) * (1 - c1[i]) + z_prime[i-1] * c1
    #
    # Thus we have to do the following:
    # a) latent_image contains the z_primes for all steps.
    # b) Iterate through all the steps, denoising from z_prime[0] initially.
    # c) Blend the denoised z[i] with z_prime[i] to get a new z[i].

    pbar = comfy.utils.ProgressBar(steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    logger.warning(f"zp_indices={zp_indices}, c1={c1}")

    for zp_idx, i in enumerate(zp_indices):
        out_i = zp_idx + 1
        logger.warning(f"zp_idx={zp_idx}, out_i={out_i}, i={i}")
        
        # Grab the i-th z_prime and i-th noise tensor from their batches.
        # Unsqueezing replaces the batch dimension with 1, so it transforms
        # [i, channel, width, height] into [1, channel, width, height]
        z_prime_i = z_primes[i].unsqueeze(0)
        noise_i = noise[i].unsqueeze(0)

        # The paper tells us to de-noise z[i-1] from step
        # T to T-1; in ComfyUI lingo, that means going from
        # step i-1 to step i because we iterate in the reverse
        # direction.
        z_start_step = i - 1
        z_last_step = i
        z_i_minus_1 = z_out[out_i - 1]

        # Define a callback function for the sampler that will
        # correctly indicate our progress across the whole batch.
        def inner_callback (step, x0, x, total_steps):
            pbar.update_absolute(i + step + 1, total_steps * steps)

        # De-noise z[i-1] from step i-1 to step i. Recall that since we
        # start this loop from i=1, z[i-1] is initialized with z_prime[0].
        # After we have the de-noised latent, we will mix it with z_prime[i]
        # according to the paper's cosine blending function. The blended
        # latent will then become z[i] and we will head to the next iteration.
        samples_i = comfy.sample.sample(
                        model, noise_i, steps, cfg,
                        sampler_name, scheduler, positive, negative, z_i_minus_1,
                        denoise=denoise,
                        disable_noise=disable_noise,
                        start_step=z_start_step,
                        last_step=z_last_step,
                        force_full_denoise=force_full_denoise,
                        noise_mask=noise_mask,
                        disable_pbar=disable_pbar, seed=seed,
                        callback=inner_callback)

        # Move samples to the same device as z_prime_i so that we can
        # work with them both to mix below.
        samples_i = samples_i.to(z_prime_i.device)

        # Store the de-noised samples in our output tensor.
        samples_out[out_i] = samples_i

        # Find z_hat (as per the paper) by applying the c1 blending schedule
        # to the samples and the prior z_prime latent. The paper suggests 
        # following this formula, which will mix in a declining fraction of
        # z_prime as de-noising continues:
        #
        # z_hat[i] = denoise(z[i-1]) * (1 - c1[i]) + z_prime[i-1] * c1

        c1_i = c1[zp_idx]
        z_i = c1_i * z_prime_i + (1 - c1_i) * samples_i

        z_out[out_i] = z_i
        out_i += 1

    out = latent_image_batch.copy()
    out["samples"] = z_out

    # We output three latent batches so that you can see how the process
    # works step by step if you wish:
    # 1. The de-noised latents.
    # 2. The noised latents that were provided at the input.
    # 3. The intermediate samples before mixing.
    return (out, {"samples":z_primes}, {"samples": samples_out},)


class IterativeMixingKSamplerAdv:
    """
    Take a batch of latents, z_prime, and progressively de-noise them
    step by step from z_prime[0] to z_prime[steps], mixing in a weighted
    fraction of z_prime[i] at each step so that de-noising is guided by
    the z_prime latents. This batch sampler assumes that the number of steps
    is just the length of z_prime, so there is no steps parameter. The parameter
    latent_image_batch should come from the Batch Unsampler node. The parameter
    alpha_1 controls an exponential cosine function that schedules how much
    of the noised latents to mix with the de-noised latents at each step.
    Small values cause more of the noised latents to be mixed in at each step,
    which provides more guidance to the diffusion, but which may result in more
    artifacts. Large values (i.e. >1.0) can cause output to be grainy. Your
    mileage may vary.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "step_increment": ("INT", {"default": 1, "min": 1, "max": 10000}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image_batch": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "alpha_1": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 100.0, "step": 0.05}),
                    "reverse_input_batch": ("BOOLEAN", {"default": True})
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT","LATENT")
    FUNCTION = "sample"

    CATEGORY = "test"

    def sample(self, model, seed, cfg, sampler_name, scheduler, step_increment, positive, negative, latent_image_batch, denoise=1.0, alpha_1=0.1, reverse_input_batch=True):
        return iterative_mixing_ksampler(model, seed, cfg, sampler_name, scheduler, step_increment, positive, negative, latent_image_batch, denoise=denoise, alpha_1=alpha_1, reverse_input_batch=True)
    
class IterativeMixingKSamplerSimple:
    """
    A simplified version of IterativeMixingKSamplerAdv, this node
    does the noising (unsampling) and de-noising (sampling) all within
    one node with easy settings.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0,
                                      "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "alpha_1": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 100.0, "step": 0.05}),

                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 1, "max": 10000}),
                    "step_increment": ("INT", {"default": 1, "min": 1, "max": 10000}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "test"

    def __init__(self):
        self.batch_unsampler = BatchUnsampler()

    def sample(self, model, positive, negative, latent_image,
               seed, steps, cfg, sampler_name, scheduler,
               denoise, alpha_1,
               start_at_step, end_at_step, step_increment):
        (z_primes,) = self.batch_unsampler.unsampler(model, sampler_name,
                                                 scheduler, steps,
                                                 start_at_step, end_at_step,
                                                 latent_image)
        (z_out, _, _) = iterative_mixing_ksampler(model, seed, cfg, sampler_name, scheduler, step_increment, positive, negative, z_primes, denoise=denoise, alpha_1=alpha_1, reverse_input_batch=True)

        # Return just the final z_out (as a 4D tensor)
        return ({"samples": z_out["samples"][-1:]},)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Batch Unsampler": BatchUnsampler,
    "Iterative Mixing KSampler Advanced": IterativeMixingKSamplerAdv,
    "Iterative Mixing KSampler": IterativeMixingKSamplerSimple
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Batch Unsampler": "Batch Unsampler",
    "Iterative Mixing KSampler Advanced": "Iterative Mixing KSampler Advanced",
    "Iterative Mixing KSampler": "Iterative Mixing KSampler"
}