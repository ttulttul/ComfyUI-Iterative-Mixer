import functools
import io

from abc import ABC
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.samplers
import comfy.k_diffusion.sampling
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import trange, tqdm

from .samplers import SAMPLERS_MAP, BLENDING_FUNCTION_MAP, BLENDING_SCHEDULE_MAP, get_blending_schedule, get_blending_function
from .utils.noise import perlin_masks
from .utils import generate_noised_latents

@torch.no_grad()
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



@torch.no_grad()
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class LatentBatchStatisticsPlot:
    """
    Generate a plot of the statistics of a batch of latents for analysis.
    Outputs an image.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "batch": ("LATENT",)
                }}
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot_image",)
    FUNCTION = "statistics"

    CATEGORY = "tests"

    @torch.no_grad()
    def statistics(self, batch):
        """
        Run a statistical test on each latent in a batch to see how
        close to normal each latent is.
        """

        from scipy import stats

        batch = batch["samples"]
        batch_size = batch.shape[0]
        p_values = []
        means = []
        std_devs = []

        for i in trange(batch.shape[0]):
            # Flatten the tensor
            tensor_1d = batch[i].flatten()

            # Convert to NumPy array
            numpy_array = tensor_1d.numpy()

            # Perform Shapiro-Wilk test
            _, p = stats.shapiro(numpy_array)

            # Store the statistics for this latent
            p_values.append(p)
            means.append(numpy_array.mean())
            std_devs.append(numpy_array.std())
        
        # Assuming 'p_values' is the array of p-values
        # Create a figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Shapiro-Wilk test results
        axs[0].plot(p_values, label="p-values", marker='o', linestyle='-')
        axs[0].set_title('Shapiro-Wilk Test P-Values')
        axs[0].set_xlabel('Batch Number')
        axs[0].set_ylabel('P-Value')
        axs[0].axhline(y=0.05, color='r', linestyle='--', label='Normal Threshold')
        axs[0].legend()

        # Mean
        axs[1].plot(means, marker='o', linestyle='-')
        axs[1].set_title('Mean of Each Batch Latent')
        axs[1].set_xlabel('Batch Number')
        axs[1].set_ylabel('Mean')

        # Standard Deviation
        axs[2].plot(std_devs, marker='o', linestyle='-')
        axs[2].set_title('Standard Deviation of Each Batch Latent')
        axs[2].set_xlabel('Batch Number')
        axs[2].set_ylabel('Standard Deviation')

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Create a PIL image from the buffer and move channels to the end.
        pil_image = Image.open(buf)
        image_tensor = pil2tensor(pil_image)

        # Make this into a batch of one as required by Comfy.
        batch_output = image_tensor.unsqueeze(0)

        # Add a single batch dimension and we're done.
        # The channel dimension has to go on the end.
        return batch_output

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
                    "normalize": ("BOOLEAN", {"default": False})
                }}
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_batch",)
    FUNCTION = "unsampler"

    CATEGORY = "tests"

    @torch.no_grad()
    def unsampler(self, model, sampler_name, scheduler, steps,
                  start_at_step, end_at_step, latent_image, normalize=False):
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
        z = generate_noised_latents(latent_image, sigmas, normalize=normalize)

        # Return the batch of progressively noised latents.
        out = {"samples": z}
        return (out,)

def plot_blending_schedule(schedule: torch.Tensor) -> torch.Tensor:
    """
    Generate a plot of the blending schedule as an image to output from
    the advanced sampler node.
    """

    schedule_size = schedule.shape[0]
    
    # Assuming 'p_values' is the array of p-values
    # Create a figure with subplots
    plt.figure(figsize=(12, 8))
    plt.plot(schedule, label="Schedule", marker='o', linestyle='-')
    plt.title('Blending Schedule')
    plt.xlabel('Step')
    plt.ylabel('Fraction of z_prime to blend')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Create a PIL image from the buffer and move channels to the end.
    pil_image = Image.open(buf)
    image_tensor = pil2tensor(pil_image)

    # Add a single batch dimension and we're done.
    # The channel dimension has to go on the end.
    return image_tensor

class IterativeMixingKSampler:
    """
    A class to manage all the pre-processing of arguments that we need before we
    can conduct sampling. This class exists to make it easier for us to develop
    derivative algorithms and play with ideas without rewriting everything each time.
    """

    def preamble(self, model, seed, cfg, sampler_name, scheduler, positive, negative,
                 latent_image_batch, denoise=1.0, disable_noise=False,
                 force_full_denoise=False, c1=None, alpha_1=0.5, reverse_input_batch=True,
                 blending_schedule="cosine",
                 start_blending_at_pct=0.0,
                 stop_blending_at_pct=1.0,
                 clamp_blending_at_pct=1.0,
                 blending_function="addition"):
        # Store basic parameters.
        self.model = model
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.positive = positive
        self.negative = negative
        self.denoise = denoise
        self.force_full_denoise = force_full_denoise
        self.disable_noise = disable_noise
        self.seed = seed

        # I originally had a step_increment argument but there is no purpose
        # to it, because you can just set the step count and of course
        # the sigmas and everything will adjust accordingly.
        self.step_increment = 1

        # Get the z_primes and move them to the GPU.
        self.z_primes = latent_image_batch["samples"].to(comfy.model_management.get_torch_device())

        # You will almost always want to reverse the input batch, which
        # we assume is a set of progressively noisier latents. For de-noising,
        # this batch order needs to be reversed so that we're going from
        # the noisiest latent back to the start.
        if reverse_input_batch == True:
            self.z_primes = torch.flip(self.z_primes, [0])

        if disable_noise:
            self.noise = torch.zeros(self.z_primes.size(),
                                     dtype=self.z_primes.dtype,
                                     layout=self.z_primes.layout, device="cpu")
        else:
            self.batch_inds = latent_image_batch["batch_index"] if "batch_index" in latent_image_batch else None
            self.noise = comfy.sample.prepare_noise(self.z_primes, seed, self.batch_inds)

        self.noise_mask = None
        if "noise_mask" in latent_image_batch:
            self.noise_mask = latent_image_batch["noise_mask"]

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
        self.steps = self.z_primes.shape[0] # batch size gives us the de-noising step count
        self.zp_indices = list(range(self.step_increment - 1, self.steps, self.step_increment))
        if self.zp_indices[-1] != self.steps - 1:
            self.zp_indices.append(self.steps - 1)

        self.loop_count = len(self.zp_indices)

        # z_out is going to store z_prime[0] plus all the de-noised
        # weighted mixtures of denoise(z_prime[i], z_hat[i-1]).
        # Therefore, we give it a length that is one longer than
        # the loop count because the first entry is set to z_prime[0].
        # z_out has the same channels, width, and height as z_prime.
        self.z_out = torch.zeros(self.loop_count + 1, *self.z_primes.size()[1:])
        self.z_out = self.z_out.to(self.z_primes.device)

        # We also return the intermediate samples for analysis.
        # This tensor has the same shape exactly as z_out but the
        # first entry is left all zeroes.
        self.samples_out = torch.zeros_like(self.z_out)
        self.samples_out = self.samples_out.to(self.z_primes.device)

        # The first output value is the first z_prime latent.
        # Implicitly, samples_out[0] is just zeroes.
        self.z_out[0] = self.z_primes[0]

        # The first z_prime that we will use in the de-noising loop below
        # is z_primes[0] unsqueezed to remove the batch dimension.
        self.z_i = self.z_primes[0].unsqueeze(0)


        # If desired, start blending at a certain step count by setting
        # the blending curve to 0.0
        self.start_blending_at_step = int(self.steps * start_blending_at_pct)

        # If desired, stop blending at a certain fraction of steps by
        # setting the last few values of c1 to 1.0, which will cause
        # 0% of the z_prime latents to be mixed at each following step.
        # This setting can help to reduce the noisiness of the output at
        # the expense of reducing guidance slightly.
        self.stop_blending_at_step = int(self.steps * stop_blending_at_pct)

        # Get the blending parameter from the DemoFusion paper.
        self.c1 = get_blending_schedule(self.zp_indices, blending_schedule, alpha_1=alpha_1,
                                        start_blending_at_step=self.start_blending_at_step,
                                        stop_blending_at_step=self.stop_blending_at_step,
                                        clamp_blending_at_pct=clamp_blending_at_pct)

        # Plot the blending schedule so we can return it in the advanced module.
        self.blending_plot = plot_blending_schedule(self.c1)

        # Move the blending schedule tensor to the same device as our
        # latents.
        self.c1 = self.c1.to(self.z_primes.device)

        # Get a blending function.
        self.blend = get_blending_function(blending_function)

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

        self.pbar = comfy.utils.ProgressBar(self.steps)
        self.disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Set up our output variables.
        self.out = latent_image_batch.copy()
        self.z_primes_out = latent_image_batch.copy()
        self.intermediates_out = latent_image_batch.copy()

    def __call__(self, *args, **kwargs):
        """
        This method is meant to be called from the Node class and takes all the parameters
        consumed by the preamble() method above. We don't consume those parameters directly;
        instead we rely on preamble() to turn them into relevant instance attributes.
        """
        self.preamble(*args, **kwargs)
        
        for zp_idx, i in tqdm(enumerate(self.zp_indices)):
            out_i = zp_idx + 1
            
            # Grab the i-th z_prime and i-th noise tensor from their batches.
            # Unsqueezing replaces the batch dimension with 1, so it transforms
            # [i, channel, width, height] into [1, channel, width, height]
            z_prime_i = self.z_primes[i].unsqueeze(0)
            noise_i = self.noise[i].unsqueeze(0)

            # The paper tells us to de-noise z[i-1] from step
            # T to T-1; in ComfyUI lingo, that means going from
            # step i-1 to step i because we iterate in the reverse
            # direction.
            z_start_step = i - 1
            z_last_step = i
            z_i_minus_1 = self.z_out[out_i - 1]

            # Define a callback function for the sampler that will
            # correctly indicate our progress across the whole batch.
            def inner_callback (step, x0, x, total_steps):
                self.pbar.update_absolute(i + step + 1, total_steps * self.steps)

            # De-noise z[i-1] from step i-1 to step i. Recall that since we
            # start this loop from i=1, z[i-1] is initialized with z_prime[0].
            # After we have the de-noised latent, we will mix it with z_prime[i]
            # according to the paper's cosine blending function. The blended
            # latent will then become z[i] and we will head to the next iteration.
            samples_i = comfy.sample.sample(
                            self.model, noise_i, self.steps, self.cfg,
                            self.sampler_name, self.scheduler, self.positive, self.negative, z_i_minus_1,
                            denoise=self.denoise,
                            disable_noise=self.disable_noise,
                            start_step=z_start_step,
                            last_step=z_last_step,
                            force_full_denoise=self.force_full_denoise,
                            noise_mask=self.noise_mask,
                            disable_pbar=True, seed=self.seed,
                            callback=inner_callback)

            # Move samples to the same device as z_prime_i so that we can
            # work with them both to mix below.
            samples_i = samples_i.to(z_prime_i.device)

            # Store the de-noised samples in our output tensor.
            self.samples_out[out_i] = samples_i

            # Find z_hat (as per the paper) by applying the c1 blending schedule
            # to the samples and the prior z_prime latent. The paper suggests 
            # following this formula, which will mix in a declining fraction of
            # z_prime as de-noising continues:
            #
            # z_hat[i] = denoise(z[i-1]) * (1 - c1[i]) + z_prime[i-1] * c1

            c1_i = self.c1[zp_idx]
            z_i = self.blend(z_prime_i, samples_i, c1_i)

            self.z_out[out_i] = z_i

        
        self.out["samples"] = self.z_out
        self.z_primes_out["samples"] = self.z_primes
        self.intermediates_out["samples"] = self.samples_out

        # We output three latent batches so that you can see how the process
        # works step by step if you wish:
        # 1. The de-noised latents.
        # 2. The noised latents that were provided at the input.
        # 3. The intermediate samples before mixing.
        return (self.out, self.z_primes_out, self.intermediates_out, self.blending_plot)

def iterative_mixing_ksampler(model, seed, cfg, sampler_name, scheduler, positive, negative,
                              latent_image_batch, denoise=1.0, disable_noise=False,
                              force_full_denoise=False, c1=None, alpha_1=0.5, reverse_input_batch=True,
                              blending_schedule="cosine",
                              start_blending_at_pct=0.0,
                              stop_blending_at_pct=1.0,
                              clamp_blending_at_pct=1.0,
                              blending_function="addition"):
    # I originally had a step_increment argument but there is no purpose
    # to it, because you can just set the step count and of course
    # the sigmas and everything will adjust accordingly.
    step_increment = 1

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

    # If desired, start blending at a certain step count by setting
    # the blending curve to 0.0
    start_blending_at_step = int(steps * start_blending_at_pct)

    # If desired, stop blending at a certain fraction of steps by
    # setting the last few values of c1 to 1.0, which will cause
    # 0% of the z_prime latents to be mixed at each following step.
    # This setting can help to reduce the noisiness of the output at
    # the expense of reducing guidance slightly.
    stop_blending_at_step = int(steps * stop_blending_at_pct)

    # Get the blending parameter from the DemoFusion paper.
    if c1 is None:
        c1 = get_blending_schedule(zp_indices, blending_schedule, alpha_1=alpha_1,
                                   start_blending_at_step=start_blending_at_step,
                                   stop_blending_at_step=stop_blending_at_step,
                                   clamp_blending_at_pct=clamp_blending_at_pct)

    # Plot the blending schedule so we can return it in the advanced module.
    blending_plot = plot_blending_schedule(c1)

    # Move the blending schedule tensor to the same device as our
    # latents.
    c1 = c1.to(z_primes.device)

    # Get a blending function.
    blend = get_blending_function(blending_function)

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

    for zp_idx, i in tqdm(enumerate(zp_indices)):
        out_i = zp_idx + 1
        
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
                        disable_pbar=True, seed=seed,
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
        z_i = blend(z_prime_i, samples_i, c1_i)

        z_out[out_i] = z_i

    out = latent_image_batch.copy()
    out["samples"] = z_out

    z_primes_out = latent_image_batch.copy()
    z_primes_out["samples"] = z_primes

    intermediates_out = latent_image_batch.copy()
    intermediates_out["samples"] = samples_out

    # We output three latent batches so that you can see how the process
    # works step by step if you wish:
    # 1. The de-noised latents.
    # 2. The noised latents that were provided at the input.
    # 3. The intermediate samples before mixing.
    return (out, z_primes_out, intermediates_out, blending_plot)

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
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image_batch": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "alpha_1": ("FLOAT", {"default": 2.4, "min": 0.05, "max": 100.0, "step": 0.05}),
                    "reverse_input_batch": ("BOOLEAN", {"default": True}),
                    "blending_schedule": (list(BLENDING_SCHEDULE_MAP.keys()), {"default": "cosine"}),
                    "stop_blending_at_pct": ("FLOAT", {"default": 1.0}),
                    "clamp_blending_at_pct": ("FLOAT", {"default": 1.0}),
                    "blending_function": (list(BLENDING_FUNCTION_MAP.keys()), {"default": "addition"})
                    }
                }

    RETURN_TYPES = ("LATENT","LATENT","LATENT", "IMAGE",)
    RETURN_NAMES = ("mixed_latents", "noised_latents", "intermediate_latents", "plot_image",)
    FUNCTION = "sample"

    CATEGORY = "test"

    def sample(self, model, seed, cfg, sampler_name, scheduler, positive, negative,
               latent_image_batch, denoise=1.0, alpha_1=0.1, reverse_input_batch=True,
               blending_schedule="cosine",
               stop_blending_at_pct=1.0, clamp_blending_at_pct=1.0,
               blending_function=list(BLENDING_FUNCTION_MAP.keys())[0]):
        sampler = IterativeMixingKSampler()
        return sampler(model, seed, cfg, sampler_name, scheduler, positive, negative,
                                         latent_image_batch, denoise=denoise, alpha_1=alpha_1, reverse_input_batch=True,
                                         blending_schedule=blending_schedule,
                                         stop_blending_at_pct=stop_blending_at_pct,
                                         clamp_blending_at_pct=clamp_blending_at_pct,
                                         blending_function=blending_function)

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
                    "steps": ("INT", {"default": 40, "min": 0, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0,
                                      "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "alpha_1": ("FLOAT", {"default": 2.4, "min": 0.05, "max": 100.0, "step": 0.05}),

                    "blending_schedule": (list(BLENDING_SCHEDULE_MAP.keys()), {"default": "cosine"}),
                    "blending_function": (list(BLENDING_FUNCTION_MAP.keys()), {"default": "addition"}),

                    "normalize_on_mean": ("BOOLEAN", {"default": False})
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
               blending_schedule,
               blending_function,
               normalize_on_mean):
        (z_primes,) = self.batch_unsampler.unsampler(model, sampler_name,
                                                 scheduler, steps,
                                                 0, steps,
                                                 latent_image, normalize=normalize_on_mean)
        
        sampler = IterativeMixingKSampler()
        (z_out, _, _, _) = sampler(model, seed, cfg, sampler_name, scheduler, positive, negative,
                                                  z_primes, denoise=denoise, alpha_1=alpha_1, reverse_input_batch=True,
                                                  blending_schedule=blending_schedule,
                                                  blending_function=blending_function,
                                                  stop_blending_at_pct=1.0)

        # Return just the final z_out (as a 4D tensor)
        return ({"samples": z_out["samples"][-1:]},)
    

class LatentBatchComparator:
    """
    Generate plots showing the differences between two batches of latents.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "latent_batch_1": ("LATENT", ),
                    "latent_batch_2": ("LATENT", )
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot_image",)

    CATEGORY = "test"

    FUNCTION = "plot_latent_differences"

    def plot_latent_differences(self, latent_batch_1, latent_batch_2):
        """
        Generate a plot of the differences between two batches of latents.
        """

        import torch.nn.functional as F

        (tensor1, tensor2) = [x["samples"] for x in (latent_batch_1, latent_batch_2)]
        
        # We cannot compare latent batches if their dimensions are different.
        if tensor1.shape != tensor2.shape:
            raise ValueError("Latent batches must have the same shape: %s != %s" %\
                            (tensor1.shape, tensor2.shape))

        # Grab the shape
        B, C, H, W = tensor1.shape

        # Vectorized calculation of pairwise Cosine Similarity
        tensor1_flat = tensor1.view(B, -1)
        tensor2_flat = tensor2.view(B, -1)

        # Add an extra dimension to tensor1_flat for broadcasting
        tensor1_flat_expanded = tensor1_flat.unsqueeze(1)

        # Compute cosine similarity along the last dimension (C*H*W)
        cosine_similarities_vectorized = F.cosine_similarity(tensor1_flat_expanded, tensor2_flat.unsqueeze(0), dim=2)

        # Plotting the vectorized distances
        plt.figure(figsize=(15, 10))

        # Plot Cosine Similarities
        plt.imshow(cosine_similarities_vectorized, cmap='viridis')
        plt.title('Cosine Similarity Matrix')
        plt.xlabel('Batch 1 Index')
        plt.ylabel('Batch 2 Index')

        plt.tight_layout()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Create a PIL image from the buffer and move channels to the end.
        pil_image = Image.open(buf)
        image_tensor = pil2tensor(pil_image)

        # Make this into a batch of one as required by Comfy.
        batch_output = image_tensor.unsqueeze(0)
        return batch_output

class MixingMaskGeneratorNode:
    """
    A node that can generate different kinds of noise mask batches for
    iterative mixing purposes.
    """

    MASK_TYPES = ["perlin", "random"]
    MAX_RESOLUTION=8192 # copied and pasted from ComfyUI/nodes.py; there is no library way to get this number

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_type": (s.MASK_TYPES, {"default": "perlin"}),
                "perlin_scale": ("FLOAT", {"default": 10., "min": 0.1, "max": 400.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 16, "max": s.MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": s.MAX_RESOLUTION, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    
    RETURN_TYPES = ("MASK",)
    CATEGORY = "mask/generation"

    FUNCTION = "get_masks"

    def get_masks(self, mask_type, perlin_scale, seed, width, height, batch_size):
        mask_height = height // 8
        mask_width = width // 8
        
        if mask_type == "perlin":
            perlin_tensors = perlin_masks(batch_size, mask_width, mask_height, device=self.device, seed=seed, scale=perlin_scale)
            masks = perlin_tensors.view(batch_size, 1, mask_height, mask_width)
        elif mask_type == "random":
            masks = torch.randn([batch_size, width // 8, height // 8])
        else:
            raise ValueError("invalid mask_type")
        
        return (masks,)


class IterativeMixingSamplerNode:
    """
    A sampler implementing iterative mixing of latents.
    Use this with the SamplerCustom node.
    """
    PERLIN_MODES = ["masks", "latents", "matched_noise"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "sampler": (list(SAMPLERS_MAP.keys()), {"default": "euler"}),
                    "alpha_1": ("FLOAT", {"default": 2.4, "min": 0.05, "max": 100.0, "step": 0.05}),
                    "blending_schedule": (list(BLENDING_SCHEDULE_MAP.keys()), {"default": "cosine"}),
                    "blending_function": (list(BLENDING_FUNCTION_MAP.keys()), {"default": "addition"}),
                    "normalize_on_mean": ("BOOLEAN", {"default": False}),
                    "start_blending_at_pct": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                    "stop_blending_at_pct": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.01}),
                    "clamp_blending_at_pct": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "blend_min": ("FLOAT", {"default": 0.0, "step": 0.01}),
                    "blend_max": ("FLOAT", {"default": 1.0, "step": 0.01}),
                    "perlin_mode": (s.PERLIN_MODES, {"default": "masks"}),
                    "perlin_strength": ("FLOAT", {"default": 0.75, "step": 0.001}),
                    "perlin_scale": ("FLOAT", {"default": 10., "min": 0.1, "max": 400.0})
        },
        "optional": {
            "mixing_masks": ("MASK",)
        }}
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def get_sampler(self, model, sampler, alpha_1,
                    blending_schedule, blending_function, normalize_on_mean,
                    start_blending_at_pct, stop_blending_at_pct,
                    clamp_blending_at_pct, blend_min, blend_max,
                    perlin_mode, perlin_strength, perlin_scale, mixing_masks=None):
        extras = {k: v for k, v in locals().items() if k != 'self'}

        # We cannot have an extra arg called "model" as this will conflict with
        # the model positional argument passed into the sampler, which is the underlying
        # neural network model rather than the comfy "model" node object.
        extras['model_node'] = extras['model']
        del extras['model']

        # Create a sampler object based on the selected sampler.
        # We use partial() to bind the self param onto the front.
        if sampler not in SAMPLERS_MAP:
            raise ValueError(f"invalid sampler: {sampler}")

        sampler_obj = SAMPLERS_MAP[sampler]()
        sampler_fn = functools.partial(sampler_obj.__call__)
        del extras['sampler']

        # Call Comfy's sampler object, passing in our sampler function.
        sampler = comfy.samplers.KSAMPLER(sampler_fn, extra_options=extras)
        return (sampler, )
    
class IterativeMixingScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    @torch.no_grad()
    def get_sigmas(self, model, scheduler, steps, denoise):
        # Determine the sigmas based on the specified denoising strength.
        # This is mostly copied from comfy_extras/nodes_custom_sampler.py.
        sigmas = None
        cs = comfy.samplers.calculate_sigmas_scheduler

        if denoise is None or denoise > 0.9999:
            sigmas = cs(model.model, scheduler, steps).cpu()
        else:
            new_steps = int(steps/denoise)
            sigmas = cs(model.model, scheduler, new_steps).cpu()
            sigmas = sigmas[-(steps + 1):]
        
        return (sigmas, )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Batch Unsampler": BatchUnsampler,
    "Latent Batch Statistics Plot": LatentBatchStatisticsPlot,
    "Latent Batch Comparison Plot": LatentBatchComparator,
    "Iterative Mixing KSampler Advanced": IterativeMixingKSamplerAdv,
    "Iterative Mixing KSampler": IterativeMixingKSamplerSimple,
    "IterativeMixingSampler": IterativeMixingSamplerNode,
    "IterativeMixingScheduler": IterativeMixingScheduler,
    "MixingMaskGenerator": MixingMaskGeneratorNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Batch Unsampler": "Batch Unsampler",
    "Latent Batch Statistics Plot": "Latent Batch Statistics Plot",
    "Iterative Mixing KSampler Advanced": "Iterative Mixing KSampler Advanced",
    "Iterative Mixing KSampler": "Iterative Mixing KSampler",
    "Latent Batch Comparison Plot": "Latent Batch Comparison Plot"
}