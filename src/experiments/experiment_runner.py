"""
Purpose: Applies the requested transformation on a given dataset,
    and runs performance evaluation for the given detectors.

Experiments:
    - CleanExperiment: Evaluates the original dataset.
    - RandomSeverityExperiment: Applies one corruption with a random
      severity (1-5) per image.
    - SeveritySweepExperiment: Applies one corruption at each
      severity level (1-5).
    - RandomCompositeExperiment: Applies a random combination
      of corruptions to each image.

Each experiment generates its datasets once per call to run()
and evaluates every detector against those same datasets.

For repeated runs, call run() multiple times or create new
experiment instances with different seeds.
"""

import random
from abc import ABC, abstractmethod

from src.evaluation.evaluator import Evaluator
from src.experiments.transforms.composite import CompositeTransform

SEVERITIES = range(1, 6)


class BaseExperiment(ABC):
    """Generates datasets and evaluates detectors."""

    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**32)
        self.rng = random.Random(self.seed)

    def run(self, dataset, detectors):
        """Returns {(corruption, severity): {detector_name: result}}."""
        evaluator = Evaluator(detectors)

        # Step 1: Generate datasets
        datasets = self._generate_datasets(dataset)

        # Step 2: Evaluate detectors
        results = {}

        for key, data in datasets.items():
            results[key] = evaluator.evaluate(data)

        return results

    @abstractmethod
    def _generate_datasets(self, dataset):
        """Return {(corruption_name, severity): corrupted dataset}."""
        pass

    @staticmethod
    def _corrupt(transform, dataset, severity):
        return [
            transform.apply(image, boxes, severity)
            for image, boxes in dataset
        ]


class RandomSeverityExperiment(BaseExperiment):
    """Applies one corruption with random severity per image."""

    def __init__(self, transform, seed=None):
        super().__init__(seed)
        self.transform = transform
        self.name = type(transform).__name__

    def _corrupt(self, transform, dataset, severity=None):
        return [
            transform.apply(image, boxes, self.rng.choice(SEVERITIES))
            for image, boxes in dataset
        ]

    def _generate_datasets(self, dataset):
        corrupted = self._corrupt(self.transform, dataset)

        return {
            (self.name, "random"): corrupted
        }


class SeveritySweepExperiment(BaseExperiment):
    """Applies one corruption across severity levels 1-5."""

    def __init__(self, transform, seed=None):
        super().__init__(seed)
        self.transform = transform
        self.name = type(transform).__name__

    def _generate_datasets(self, dataset):
        datasets = {}

        for severity in SEVERITIES:
            datasets[(self.name, severity)] = self._corrupt(
                self.transform, dataset, severity
            )

        return datasets
class RandomCompositeExperiment(BaseExperiment):
    """Applies a random combination of corruptions to each image."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.composite = CompositeTransform(rng=self.rng)

    def _generate_datasets(self, dataset):
        corrupted = self._corrupt(
            self.composite, dataset, severity=None
        )

        return {
            ("composite", "random"): corrupted
        }

class CleanExperiment(BaseExperiment):
    """Evaluates the original dataset without transformations."""

    def _generate_datasets(self, dataset):
        return {
            ("clean", None): dataset
        }