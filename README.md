# ComfyUI Iterative Mixing Nodes

This repo contains 2 nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that combine to implement a strategy I'm calling Iterative Mixing of Latents.
The technique is stolen from the [DemoFusion](https://arxiv.org/abs/2311.16973) paper with gratitude. I also acknowledge [BlenderNeko](https://github.com/BlenderNeko) for the inspiration that led to the Batch Unsampler node included in this pack.

## Nodes

### Iterative Mixing KSampler:
This node de-noises a latent image while mixing a bit of a noised sequence of latents at each step.
- **model**: a diffusion model
- **positive**: positive conditioning
- **negative**: negative conditioning
- **latent_image_batch**: the batch from the Batch Unsampler
- **seed**: noise generation seed
- **steps**: the number of steps of de-noising to perform
- **cfg**: classifier-free-guidance scale for de-noising
- **sampler_name**: the name of the sampler you wish to use
- **scheduler**: the name of the scheduler to use
- **denoise**: the denoising strength - note I'm not sure if this does anything
- **alpha_1**: a parameter to specify how latents are mixed; try values between 0.1 and 2.0
- **reverse_input_batch**: should always be True because the Batch Unsampler produces latents in the wrong order
- **blending_schedule**: choose between cosine and linear to get a different effect; alpha_1 is ignored with linear

### Batch Unsampler:
This node takes a latent image as input, adding noise to it in the manner described in the original [Latent Diffusion Paper](https://arxiv.org/abs/2112.10752).
- **model**: a diffusion model
- **latent_image**: the latent that you want to unsample into a series of progressively noisier latents
- **sampler_name**: the sampler that will give us the correct sigmas for the model
- **scheduler**: the scheduler that will give us the correct sigmas for the model
- **steps**: the number of steps of noising; the latent will be noised all the way across this many steps
- **start_at_step**: if you want to start part way (untested)
- ***end_at_step**: if you want to end part way (untested)

### Iterative Mixing KSampler Advanced:
This node de-noises a latent image while mixing a bit of the noised latents in from the Batch Unsampler at each step. Note that the number of steps is inferred from the size of the input latent batch from the Batch Unsampler, which is why this parameter is missing.
- **model**: a diffusion model
- **positive**: positive conditioning
- **negative**: negative conditioning
- **latent_image_batch**: the batch from the Batch Unsampler
- **seed**: noise generation seed
- **cfg**: classifier-free-guidance scale for de-noising
- **sampler_name**: the name of the sampler you wish to use
- **scheduler**: the name of the scheduler to use
- **denoise**: the denoising strength - note I'm not sure if this does anything
- **alpha_1**: a parameter to specify how latents are mixed; try values between 0.1 and 2.0
- **reverse_input_batch**: should always be True because the Batch Unsampler produces latents in the wrong order
- **blending_schedule**: choose between cosine and linear to get a different effect; alpha_1 is ignored with linear

## What does "Iterative Mixing" mean?

I made up the term. In the [DemoFusion](https://arxiv.org/abs/2311.16973) paper, they use the term "skip residual" (see section 3.3):

> For each generation phase $ s $, we have already obtained a series of noise-inversed versions of $ \( z_t^{s} \) as \( z_t^{'} \) $ with $ \( t \) in \( [1, T] \) $. During the denoising process, we introduce the corresponding noise-inversed versions as skip residuals. In other words, we modify $ \( p_{\theta}(z_{t-1}|z_t) \) $ to $ \( p_{\theta}(z_{t-1}|\hat{z_t}) \) $ with

> $$
\hat{z_t}^{s} = c_1 \times z_t^{'s} + (1 - c_1) \times z_t^{s},
$$

> where $ \( c_1 = \left(\frac{1 + \cos(\frac{2\pi t}{T})}{2}\right)^{\alpha_1} \) $ is a scaled cosine decay factor with a scaling factor $ \( \alpha_1 \) $. This essentially utilizes the results from the previous phase to guide the generated image's global structure during the initial steps of the denoising process. Meanwhile, we gradually reduce the impact of the noise residual, allowing the local denoising paths to optimize the finer details more effectively in the later steps.

