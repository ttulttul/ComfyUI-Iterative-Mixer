import torch
import comfy.model_management
import comfy.sample
import comfy.utils
import comfy.samplers
import cv2
import numpy as np


from torchvision.transforms.functional import normalize
from nodes import KSampler, VAEEncode, VAEDecode, LatentFromBatch

from .nodes import BLENDING_SCHEDULE_MAP, BatchUnsampler, IterativeMixingKSamplerAdv
from .facelib.utils.face_restoration_helper import FaceRestoreHelper


# Face enhance
class IterativeMixingFaceEnhance:
    """
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"model": ("MODEL",),
                     "face_detection_model": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                     "image": ("IMAGE",),
                     "vae": ("VAE",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 0, "max": 9999}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "alpha_1": ("FLOAT", {"default": 3.0, "min": 0.05, "max": 100.0, "step": 0.05}),
                     "blending_schedule": (list(BLENDING_SCHEDULE_MAP.keys()), {"default": "cosine"}),
                     "stop_blending_at_pct": ("FLOAT", {"default": 1.0}),
                     "denoise_refine": ("FLOAT", {"default": .15, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT", "LATENT", "IMAGE")
    RETURN_NAMES = ("image_refined", "face_cropped", "unsampled_face_latents", "resampled_face_latent", "face_refined")
    FUNCTION = "enhance"
    CATEGORY = "test"

    def enhance(self, model, face_detection_model, image, vae, seed, steps, cfg, sampler_name, scheduler, positive,
                negative, denoise=1.0, alpha_1=0.1, blending_schedule='cosine',
                stop_blending_at_pct=1.0, denoise_refine=0.15):

        device = comfy.model_management.get_torch_device()

        face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1),
                                        det_model=face_detection_model, save_ext='png', use_parse=True, device=device)

        cropped_faces = self.crop_face(device, face_helper, image)
        cropped_face = cropped_faces
        cropped_face_latents = VAEEncode().encode(vae, cropped_face)
        cropped_face_latent = cropped_face_latents[0]

        unsampled_face_latents = BatchUnsampler().unsampler(
            model, sampler_name, scheduler, steps=steps, start_at_step=0, end_at_step=9999,
            latent_image=cropped_face_latent, normalize=False)
        unsampled_face_latents = unsampled_face_latents[0]

        iterative_mixing_result = IterativeMixingKSamplerAdv().sample(
            model=model, seed=seed, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive,
            negative=negative, latent_image_batch=unsampled_face_latents, denoise=denoise, alpha_1=alpha_1,
            reverse_input_batch=True, blending_schedule=blending_schedule, stop_blending_at_pct=stop_blending_at_pct)
        iterative_mixing_result = iterative_mixing_result[0]  # primary results

        resampled_face_latent = LatentFromBatch().frombatch(iterative_mixing_result, steps+1, 1)
        resampled_face_latent = resampled_face_latent[0]

        refined_resampled_face_latent = KSampler().sample(
            model=model, seed=seed, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
            positive=positive, negative=negative, latent_image=resampled_face_latent, denoise=denoise_refine)

        face_image_restored = VAEDecode().decode(vae=vae, samples=refined_resampled_face_latent[0])
        face_image_restored = face_image_restored[0]

        restored_image = self.paste_face(face_helper, face_image_restored, original_resolution=image.shape[1:3])

        face_helper.clean_all()
        return restored_image, cropped_face, unsampled_face_latents, resampled_face_latent, face_image_restored

    @staticmethod
    def crop_face(device, face_helper, image) -> torch.tensor:
        image_np = 255. * image.cpu().numpy()
        total_images = image_np.shape[0]
        out_images = np.ndarray(shape=(total_images, 512, 512, 3))
        next_idx = 0
        for i in range(total_images):
            cur_image_np = image_np[i, :, :, ::-1]
            face_helper.clean_all()
            face_helper.read_image(cur_image_np)
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()
            faces_found = len(face_helper.cropped_faces)
            if faces_found == 0:
                next_idx += 1  # output black image for no face
            if out_images.shape[0] < next_idx + faces_found:
                print((next_idx + faces_found, 512, 512, 3))
                out_images = np.resize(out_images, (next_idx + faces_found, 512, 512, 3))

            for j in range(faces_found):
                cropped_face_1 = face_helper.cropped_faces[j]
                cropped_face_2 = img2tensor(cropped_face_1 / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_2, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
                cropped_face_3 = cropped_face_2.unsqueeze(0).to(device)
                cropped_face_4 = tensor2img(cropped_face_3, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
                cropped_face_5 = cv2.cvtColor(cropped_face_4, cv2.COLOR_BGR2RGB)
                out_images[next_idx] = cropped_face_5
                next_idx += 1
        cropped_face_6 = np.array(out_images).astype(np.float32) / 255.0
        cropped_face_7 = torch.from_numpy(cropped_face_6)
        return cropped_face_7

    @staticmethod
    def paste_face(face_helper, restored_face_tensor, original_resolution):
        restored_face_np = 255. * restored_face_tensor.cpu().numpy()
        restored_face_np = restored_face_np[0]

        img_r = cv2.cvtColor(restored_face_np, cv2.COLOR_BGR2RGB)
        face_helper.add_restored_face(img_r)
        face_helper.get_inverse_affine()

        restored_img = face_helper.paste_faces_to_input_image()
        restored_img = restored_img[:, :, ::-1]  # reverses the order of elements along the third axis.
        if original_resolution != restored_img.shape[0:2]:
            restored_img = cv2.resize(restored_img, (0, 0),
                                      fx=original_resolution[1] / restored_img.shape[1],
                                      fy=original_resolution[0] / restored_img.shape[0],
                                      interpolation=cv2.INTER_LINEAR)
        restored_img = np.clip(restored_img, 0, 255)

        restored_img_np = restored_img.astype(np.float32) / 255.0
        restored_img_tensor = torch.from_numpy(restored_img_np).unsqueeze(0)
        return restored_img_tensor


def img2tensor(imgs: np.ndarray, bgr2rgb: bool = True, float32: bool = True) -> torch.tensor:
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].
    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
