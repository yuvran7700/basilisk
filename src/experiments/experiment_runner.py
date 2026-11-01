"""
Purpose: Applies the requested transformation on given dataset, 
    and run the performance evalution for given model.

    Inputs: 
    - Dataset — The images and ground-truth annotations to evaluate (val or test).
    - Detector — The trained model implementing BaseDetector.predict().
    - Transformation — The experiment to apply (blur, noise, rotation, composite etc.).

Step 1: generate the corrupted dataset(s)
    - no transform: the clean dataset
    - sweep=False:  one dataset, random severity (1-5) per image
    - sweep=True:   five datasets, one per severity 1-5
Step 2: evaluate every detector on each of those datasets
"""
import random
from abc import ABC, abstractmethod

from src.evaluation.evaluator import Evaluator
from src.experiments.transforms.all_corruptions import all_corruptions
from src.experiments.transforms.composite import CompositeTransform

SEVERITIES = range(1, 6)


class BaseExperiment(ABC):
    """Owns the run loop: build ALL datasets for a run, then evaluate them."""

    def __init__(self, runs=1):
        self.runs = runs

    def run(self, dataset, detectors):
        """Returns one dict per run: {(corruption, severity): {detector: result}}"""
        evaluators = {name: Evaluator(d) for name, d in detectors.items()}

        results = []
        for _ in range(self.runs):
            datasets = self._generate_datasets(dataset)             # Step 1: build all
            results.append(self._evaluate(datasets, evaluators))    # Step 2: evaluate
        return results

    @abstractmethod
    def _generate_datasets(self, dataset):
        """Return {(corruption, severity): corrupted dataset}."""

    @staticmethod
    def _evaluate(datasets, evaluators):
        results = {}
        for key, data in datasets.items():
            results[key] = {name: ev.evaluate(data) for name, ev in evaluators.items()}
        return results

    @staticmethod
    def _corrupt(transform, dataset, severity):
        return [transform.apply(image, boxes, severity) for image, boxes in dataset]


class RandomSeverityExperiment(BaseExperiment):
    """1. Each corruption on its own, random severity per image."""

    def __init__(self, corruptions=None, runs=1):
        super().__init__(runs)
        self.corruptions = all_corruptions() if corruptions is None else corruptions

    def _generate_datasets(self, dataset):
        datasets = {}
        for transform in self.corruptions:
            corrupted = []
            for image, boxes in dataset:
                severity = random.choice(SEVERITIES)
                corrupted.append(transform.apply(image, boxes, severity))
            datasets[(type(transform), "random")] = corrupted
        return datasets


class SeveritySweepExperiment(BaseExperiment):
    """2. Severity 1-5 for all corruptions (default) or only the ones given."""

    def __init__(self, corruptions=None):
        super().__init__(runs=1)    # deterministic, one run is enough
        self.corruptions = all_corruptions() if corruptions is None else corruptions

    def _generate_datasets(self, dataset):
        datasets = {}
        for transform in self.corruptions:
            for severity in SEVERITIES:
                datasets[(type(transform), severity)] = self._corrupt(transform, dataset, severity)
        return datasets


class RandomCompositeExperiment(BaseExperiment):
    """3. Each image gets a random stack of corruptions."""

    def __init__(self, runs=1):
        super().__init__(runs)
        self.composite = CompositeTransform()

    def _generate_datasets(self, dataset):
        corrupted = self._corrupt(self.composite, dataset, severity=None) 
        # composite draws its own levels
        return {(CompositeTransform, "random"): corrupted}


class CleanExperiment(BaseExperiment):
    """Baseline: the untouched dataset."""

    def __init__(self):
        super().__init__(runs=1)

    def _generate_datasets(self, dataset):
        return {("clean", None): dataset}