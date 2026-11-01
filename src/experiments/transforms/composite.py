import random

from .base import BaseTransform
from .all_corruptions import all_corruptions


class CompositeTransform(BaseTransform):

    def __init__(self):
        super().__init__(name="composite")
        # Fixed order of available corruptions
        self.transforms = all_corruptions()

    def apply(self, image, boxes, severity):
        # Randomly select 2–4 corruptions
        count = random.randint(2, 4)
        selected = random.sample(self.transforms, count)

        # Apply selected corruptions in a fixed order
        for transform in self.transforms:
            if transform in selected:
                level = random.randint(1, 3)
                image, boxes = transform.apply(image, boxes, level)

        return image, boxes
