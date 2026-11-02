import random

from .base import BaseTransform
from .all_corruptions import all_corruptions


class CompositeTransform(BaseTransform):

    def __init__(self, rng=None):
        super().__init__(name="composite")
        self.transforms = all_corruptions()
        self.rng = rng if rng is not None else random.Random()

    def apply(self, image, boxes, severity):
        count = self.rng.randint(2, 4)
        selected = self.rng.sample(self.transforms, count)

        for transform in self.transforms:
            if transform in selected:
                level = self.rng.randint(1, 3)
                image, boxes = transform.apply(image, boxes, level)

        return image, boxes