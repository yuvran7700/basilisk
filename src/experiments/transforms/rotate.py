# src/experiments/transforms/rotate.py
import albumentations as A

from .base import BaseTransform


class Rotate(BaseTransform):

    ANGLE_RANGES = {
        1: (-5, 5),
        2: (-10, 10),
        3: (-15, 15),
        4: (-20, 20),
        5: (-30, 30),
    }

    def __init__(self):
        super().__init__(name="rotate")
        self.pipelines = {
            severity: A.Compose(
                [A.Rotate(limit=angle_range, p=1.0)],
                bbox_params=A.BboxParams(format="pascal_voc"),
            )
            for severity, angle_range in self.ANGLE_RANGES.items()
        }

    def apply(self, image, boxes, severity):
        result = self.pipelines[severity](image=image, bboxes=boxes)
        return result["image"], result["bboxes"]