from abc import ABC, abstractmethod

class BaseTransform(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def apply(self, image, boxes, severity):
        pass