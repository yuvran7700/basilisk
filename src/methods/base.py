
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import final

from config import RESULTS_DIR


class BaseDetector(ABC):
    """Base class for all bird detection methods."""

    def __init__(self, name):
        """Initialise the detector with its method name."""
        self.name = name
        self.training_time = None

    @final
    def train(self, dataset):
        """Run training and log its duration. 
        
        Call this method to train your detector.
        Do NOT override this method. Implement _train() instead.

        """
        # Start training timer
        start = time.perf_counter()

        # Run method-specific training
        self._train(dataset)

        # Record elapsed time
        self.training_time = time.perf_counter() - start

        # Save training time
        self._save_training_time()

        return self.training_time

    def _save_training_time(self):
        """INTERNAL METHOD - Appends training time to a txt file. 
            Called automatically."""
        # Create method-specific results directory
        results_path = RESULTS_DIR / self.name
        results_path.mkdir(parents=True, exist_ok=True)

        # Define training log file
        log_path = results_path / "training_times.txt"

        # Append training time with timestamp
        with open(log_path, "a") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(
                f"{timestamp} - Training time: "
                f"{self.training_time:.4f} seconds\n"
            )

    def _train(self, dataset):
        """Override to implement training. 
           Do NOT call this method directly. Call train() instead.
        """
        pass

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
