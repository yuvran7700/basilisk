# src/experiments/transforms/corruption.py
"""
Implement image corruptions such as Gaussian Noise, Gaussian Blur, Motion Blur, 
Brightness and Contrast
"""
from imagecorruptions import corrupt

from .base import BaseTransform


class ImageCorruptionTransform(BaseTransform):

    def __init__(self, corruption_name):
        super().__init__(name=corruption_name)
        self.corruption_name = corruption_name

    def apply(self, image, boxes, severity):
        corrupted = corrupt(
            image,
            corruption_name=self.corruption_name,
            severity=severity,
        )

        return corrupted, boxes

class GaussianNoise(ImageCorruptionTransform):
    def __init__(self):
        super().__init__("gaussian_noise")


class GaussianBlur(ImageCorruptionTransform):
    def __init__(self):
        super().__init__("gaussian_blur")


class MotionBlur(ImageCorruptionTransform):
    def __init__(self):
        super().__init__("motion_blur")


class Brightness(ImageCorruptionTransform):
    def __init__(self):
        super().__init__("brightness")


class Contrast(ImageCorruptionTransform):
    def __init__(self):
        super().__init__("contrast")