# src/experiments/transforms/corruption.py
"""
Implement image corruptions such as Gaussian Noise, Gaussian Blur, Motion Blur, 
Brightness and Contrast
"""
import cv2
from .base import BaseTransform
import numpy as np
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

    def apply(self, image, boxes, severity):

        sigma = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]

        img = image.astype(np.float32) / 255.0

        noise = np.random.normal(
            loc=0,
            scale=sigma,
            size=img.shape
        )

        corrupted = np.clip(img + noise, 0, 1) * 255

        return corrupted.astype(np.uint8), boxes


class GaussianBlur(ImageCorruptionTransform):
    def __init__(self):
        super().__init__("gaussian_blur")

    def apply(self, image, boxes, severity):
        sigma = [1, 2, 3, 4, 6][severity - 1]

        corrupted = cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=sigma
        )

        return corrupted, boxes

class MotionBlur(ImageCorruptionTransform):

    def __init__(self):
        super().__init__("motion_blur")

    def apply(self, image, boxes, severity):

        kernel_size = [10, 15, 15, 15, 20][severity - 1]

        # Create horizontal motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        # Rotate kernel by a random angle
        angle = np.random.uniform(-45, 45)

        matrix = cv2.getRotationMatrix2D(
            (kernel_size / 2, kernel_size / 2),
            angle,
            1
        )

        kernel = cv2.warpAffine(
            kernel,
            matrix,
            (kernel_size, kernel_size)
        )

        kernel /= kernel.sum()

        corrupted = cv2.filter2D(image, -1, kernel)

        return corrupted, boxes


class Brightness(ImageCorruptionTransform):

    def __init__(self):
        super().__init__("brightness")

    def apply(self, image, boxes, severity):

        adjustment = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]

        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Increase brightness (V channel)
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] + adjustment * 255,
            0,
            255
        )

        corrupted = cv2.cvtColor(
            hsv.astype(np.uint8),
            cv2.COLOR_HSV2BGR
        )

        return corrupted, boxes


class Contrast(ImageCorruptionTransform):

    def __init__(self):
        super().__init__("contrast")

    def apply(self, image, boxes, severity):

        factor = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]

        img = image.astype(np.float32)

        # Calculate mean for each colour channel
        mean = np.mean(img, axis=(0, 1), keepdims=True)

        corrupted = np.clip(
            (img - mean) * factor + mean,
            0,
            255
        )

        return corrupted.astype(np.uint8), boxes
