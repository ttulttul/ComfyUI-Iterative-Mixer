import comfy.model_management
import torch
import torch.nn as nn

# normalize() and PerlinPowerFractal are borrowed from the PowerNoiseSuite with credit to @WASasquatch.
# https://github.com/WASasquatch/PowerNoiseSuite

@torch.no_grad
def perlin_masks(batch_size: int, width: int, height: int, device=None, seed: int=None, scale: float=10., **kwargs) -> torch.Tensor:
    """
    Generate a batch of Perlin noise masks with some nice defaults chosen
    to work well with iterative mixing. Credit to WASasquatch and their
    PowerNoiseSuite node for the inspiration and code.
    """

    # By default, we'll put the tensor onto Comfy's preferred device.
    if device is None:
        device = comfy.model_management.get_torch_device()
    
    c = PerlinPowerFractal(width, height)

    # Not sure why, but it seems we have to use the CPU device here and then move it to the device
    # desired by the caller or some platforms generate an error deep inside Intel code.
    masks = c.forward(batch_size, 0, 0, 0, 0, device=torch.device("cpu"), seed=seed, scale=scale, **kwargs)
    masks = masks.to(device)

    # return shape is [B, H, W, 1]
    return masks

@torch.no_grad
def range_normalize(x: torch.Tensor, target_min: float=None, target_max:float =None) -> torch.Tensor:
    """
    Linearly normalize a tensor `x` so that it ranges between 0.0 to 1.0.

    Args:
        x (torch.Tensor): The input tensor to be normalized.
        target_min (float, optional): The minimum value after normalization. 
            - When `None` min will be tensor min range value.
        target_max (float, optional): The maximum value after normalization. 
            - When `None` max will be tensor max range value.

    Returns:
        torch.Tensor: The normalized tensor
    """
    min_val = x.min()
    max_val = x.max()
    
    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val
        
    normalized = (x - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled

class PerlinPowerFractal(nn.Module):
    """
    Generate a batch of images with a Perlin power fractal effect.
    Generating a batch of 1024 images at 64x64 scale with one channel takes only 12ms on an A100.

    Args:
        width (int): Width of each tensor in pixels. Specifies the width of the generated image. Range: [64, 8192].
        height (int): Height of each image in pixels. Specifies the height of the generated image. Range: [64, 8192].
        batch_size (int): Number of noisy tensors to generate in the batch. Determines the number of images generated simultaneously. Range: [1, 64].
        X (float): X-coordinate offset for noise sampling. Shifts the noise pattern along the X-axis. Range: [-99999999, 99999999].
        Y (float): Y-coordinate offset for noise sampling. Shifts the noise pattern along the Y-axis. Range: [-99999999, 99999999].
        Z (float): Z-coordinate offset for noise sampling. Shifts the noise pattern along the Z-axis for time evolution. Range: [-99999999, 99999999].
        frame (int): The current frame number for time evolution. Controls how the noise pattern evolves over time. Range: [0, 99999999].
        evolution_factor (float): Factor controlling time evolution. Determines how much the noise evolves over time based on the batch index. Range: [0.0, 1.0].
        octaves (int): Number of octaves for fractal generation. Controls the level of detail and complexity in the output. More octaves create finer details. Range: [1, 8].
        persistence (float): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave. Higher values amplify the effect of each octave. Range: [0.01, 23.0].
        lacunarity (float): Lacunarity parameter for fractal generation. Controls the increase in frequency from one octave to the next. Higher values result in more variation between octaves. Range: [0.01, 99.0].
        exponent (float): Exponent applied to the noise values. Adjusting this parameter controls the overall intensity and contrast of the output. Higher values increase intensity and contrast. Range: [0.01, 38.0].
        scale (float): Scaling factor for frequency of noise. Larger values produce smaller, more detailed patterns, while smaller values create larger patterns. Range: [2, 2048].
        brightness (float): Adjusts the overall brightness of the generated noise.
            - -1.0 makes the noise completely black.
            - 0.0 has no effect on brightness.
            - 1.0 makes the noise completely white. Range: [-1.0, 1.0].
        contrast (float): Adjusts the contrast of the generated noise.
            - -1.0 reduces contrast, enhancing the difference between dark and light areas.
            - 0.0 has no effect on contrast.
            - 1.0 increases contrast, enhancing the difference between dark and light areas. Range: [-1.0, 1.0].
        seed (int, optional): Seed for random number generation. If None, uses random seeds for each batch. Controls the reproducibility of the generated noise. Range: [0, 0xffffffffffffffff].

    Methods:
        forward(batch_size, X, Y, Z, frame, device='cpu', evolution_factor=0.1, octaves=4, persistence=0.5, lacunarity=2.0, exponent=4.0, scale=100, brightness=0.0, contrast=0.0, seed=None, min_clamp=0.0, max_clamp=1.0):
            Generate the batch of images with Perlin power fractal effect.

    Returns:
        torch.Tensor: A tensor containing the generated images in the shape (batch_size, height, width, 1).
    """
    def __init__(self, width: int, height: int):
        """
        Initialize the PerlinPowerFractal.

        Args:
            width (int): Width of each tensor in pixels.
            height (int): Height of each image in pixels.
        """
        super(PerlinPowerFractal, self).__init__()
        self.width = width
        self.height = height

    @torch.no_grad
    def forward(self, batch_size, X, Y, Z, frame, device='cpu', evolution_factor=0.1,
                octaves=4, persistence=0.5, lacunarity=2.0, exponent=4.0, scale=100,
                brightness=0.0, contrast=0.0, seed=None, min_clamp=0.0, max_clamp=1.0):
        """
        Generate a batch of images with Perlin power fractal effect.

        Args:
            batch_size (int): Number of noisy tensors to generate in the batch.
            X (float): X-coordinate offset for noise sampling.
            Y (float): Y-coordinate offset for noise sampling.
            Z (float): Z-coordinate offset for noise sampling.
            frame (int): The current frame number for time evolution.
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            evolution_factor (float, optional): Factor controlling time evolution. Default is 0.1.
            octaves (int, optional): Number of octaves for fractal generation. Default is 4.
            persistence (float, optional): Persistence parameter for fractal generation. Default is 0.5.
            lacunarity (float, optional): Lacunarity parameter for fractal generation. Default is 2.0.
            exponent (float, optional): Exponent applied to the noise values. Default is 4.0.
            scale (float, optional): Scaling factor for frequency of noise. Default is 100.
            brightness (float, optional): Adjusts the overall brightness of the generated noise. Default is 0.0.
            contrast (float, optional): Adjusts the contrast of the generated noise. Default is 0.0.
            seed (int, optional): Seed for random number generation. If None, uses random seeds for each batch. Default is None.
            min_clamp (float, optional): Minimum value to clamp the pixel values to. Default is 0.0.
            max_clamp (float, optional): Maximum value to clamp the pixel values to. Default is 1.0.

        Returns:
            torch.Tensor: A tensor containing the generated images in the shape (batch_size, height, width, 1).
        """

        def fade(t):
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        def lerp(t, a, b):
            return a + t * (b - a)

        def grad(hash, x, y, z):
            h = hash & 15
            u = torch.where(h < 8, x, y)
            v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
            return torch.where(h & 1 == 0, u, -u) + torch.where(h & 2 == 0, v, -v)

        def noise(x, y, z, p):
            X = (x.floor() % 255).to(torch.int32)
            Y = (y.floor() % 255).to(torch.int32)
            Z = (z.floor() % 255).to(torch.int32)

            x -= x.floor()
            y -= y.floor()
            z -= z.floor()

            u = fade(x)
            v = fade(y)
            w = fade(z)

            A = p[X] + Y
            AA = p[A] + Z
            AB = p[A + 1] + Z
            B = p[X + 1] + Y
            BA = p[B] + Z
            BB = p[B + 1] + Z

            r = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                              lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                     lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                              lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))))

            return r

        unique_seed = seed if seed is not None else torch.randint(0, 10000, (1,)).item()
        torch.manual_seed(unique_seed)

        # Generate batch_count random permutations - enough to fill every mask with random indices.
        p_count = max(self.width, self.height) ** 2
        p_all = torch.stack([torch.randperm(p_count, dtype=torch.int32, device=device).repeat(2) for _ in range(batch_size)])
        
        # Initialize an empty latent batch that we will put the masks into.
        # This gives us a tensor with shape [batch_size, height, width].
        latent = torch.zeros(batch_size, self.height, self.width, dtype=torch.float32, device=device)

        # Generate unique perlin noise in each batch entry.
        for batch_idx in range(batch_size):
            # Get our slice of randomness.
            p = p_all[batch_idx]

            # Create noise_map for each item in the batch
            noise_map_single = latent[batch_idx]

            X_ = torch.arange(self.width, dtype=torch.float32, device=device).unsqueeze(0) + X
            Y_ = torch.arange(self.height, dtype=torch.float32, device=device).unsqueeze(1) + Y
            Z_ = evolution_factor * batch_idx * torch.ones(1, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(1) + Z + frame

            for octave in range(octaves):
                frequency = lacunarity ** octave
                amplitude = persistence ** octave

                nx = X_ / scale * frequency
                ny = Y_ / scale * frequency
                nz = (Z_ + frame * evolution_factor) / scale * frequency

                noise_values = noise(nx, ny, nz, p) * (amplitude ** exponent)

                noise_map_single += noise_values.squeeze(0) * amplitude

            noise_map_single = range_normalize(noise_map_single, min_clamp, max_clamp)

            latent_single = (noise_map_single + brightness) * (1.0 + contrast)
            latent_single = range_normalize(latent_single)

            # Add this latent to the batch.
            latent[batch_idx] = latent_single

        # Append a singleton final dimension to represent the grayscale channel before returning.
        return latent.unsqueeze(-1)
