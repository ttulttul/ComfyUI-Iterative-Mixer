from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .face_enhance_node import IterativeMixingFaceEnhance

NODE_CLASS_MAPPINGS['IterativeMixingFaceEnhance'] = IterativeMixingFaceEnhance
NODE_DISPLAY_NAME_MAPPINGS['IterativeMixingFaceEnhance'] = 'Iterative Mixing Face Enhancer'

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']