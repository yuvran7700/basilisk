# src/experiments/transforms/all_corruptions.py
from .corruptions import GaussianNoise, GaussianBlur, MotionBlur, Brightness, Contrast
from .rotate import Rotate


def all_corruptions():
    return [GaussianNoise(),
             GaussianBlur(),
             MotionBlur(),
             Brightness(),
             Contrast(),
             Rotate()]