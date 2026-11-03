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
import itertools
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
        """Returns {corruption_name: [evaluation_results]}."""
        evaluator = Evaluator(detectors)

        # Step 1: Generate datasets
        datasets = self._generate_datasets(dataset)

        # Step 2: Evaluate detectors
        results = {}

        for name, corrupted_datasets in datasets.items():
            results[name] = []

            for data in corrupted_datasets:
                results[name].append(evaluator.evaluate(data))

        return results

    @abstractmethod
    def _generate_datasets(self, dataset):
        """Return {corruption_name: [corrupted datasets]}."""

    @staticmethod
    def _corrupt(transform, dataset, severities):
        """Apply transform to each sample, using one severity per sample."""
        return [
            transform.apply(image, boxes, severity)
            for (image, boxes), severity in zip(dataset, severities)
        ]


class RandomSeverityExperiment(BaseExperiment):
    """Applies one corruption with random severity per image."""
    def __init__(self, transform, seed=None):
        super().__init__(seed)
        self.transform = transform

    def _generate_datasets(self, dataset):
        name = type(self.transform).__name__
        severities = (self.rng.choice(SEVERITIES) for _ in dataset)
        return {name: [self._corrupt(self.transform, dataset, severities)]}


class SeveritySweepExperiment(BaseExperiment):
    """Applies one corruption across severity levels 1-5."""

    def __init__(self, transform, seed=None):
        super().__init__(seed)
        self.transform = transform

    def _generate_datasets(self, dataset):
        name = type(self.transform).__name__
        return {
            name: [
                self._corrupt(self.transform, dataset, itertools.repeat(severity))
                for severity in SEVERITIES
            ]
        }


class RandomCompositeExperiment(BaseExperiment):
    """Applies a random combination of corruptions to each image."""
    def __init__(self, seed=None):
        super().__init__(seed)
        self.transform = CompositeTransform(rng=self.rng)

    def _generate_datasets(self, dataset):
        name = type(self.transform).__name__
        severities = itertools.repeat(None)
        return {name: [self._corrupt(self.transform, dataset, severities)]}

class CleanExperiment(BaseExperiment):
    """Evaluates the original dataset without transformations."""
    def _generate_datasets(self, dataset):
        return {type(self).__name__: [dataset]}
