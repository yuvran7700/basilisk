
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import final

from config import RESULTS_DIR


class BaseDetector(ABC):
    """Base class for all bird detection methods."""
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def predict(self, image):
        """Override to detect birds in one image.

        Args:
            image: OpenCV BGR image (H, W, 3).

        Returns:
            dict: "boxes" (N, 4) in XYXY pixels and
                  "scores" (N,) between 0 and 1.
                  Use empty arrays when no birds are detected.
        """
        pass
