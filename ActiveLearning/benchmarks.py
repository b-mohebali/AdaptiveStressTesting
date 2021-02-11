import numpy as np 
from abc import ABC, abstractmethod
from math import *
from enum import Enum
import math

class NotEnoughArguments(Exception):
    pass
class TooManyArguments(Exception):
    pass
class InputDimensionsDoNotMatch(Exception):
    pass

class ComparisonDirectionPositive(Enum):
    LESS_THAN_TH = 0
    MORE_THAN_TH = 1


class Benchmark(ABC):
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.inputDim = 0
        self.direction = None

    def _functionVec(self,data):
        return np.array([self._function(datum) for datum in data])

    # The input is a single point in space in the form of a d-dimensional Numpy vector
    @abstractmethod
    def _function(self, datum):
        pass

    def _checkInputDimensions(self, datum):
        if len(datum) != self.inputDim:
            exept = NotEnoughArguments(f'Not enough arguments for Branin Function. Please pass only {self.inputDim} arguments.') if len(datum) < self.inputDim else TooManyArguments(f'Too many arguments for Branin function. Please pass only {self.inputDim} arguments.')
            raise exept 
        return 

    def _checkInputVecDimensions(self, data):
        [self._checkInputDimensions(datum) for datum in data]
        return 

    def getLabel(self, datum, checkDim = True):
        if checkDim:
            self._checkInputDimensions(datum)
        s = self._function(datum)
        if self.direction == ComparisonDirectionPositive.LESS_THAN_TH:
            return 1 if s < self.threshold else 0
        return 1 if s > self.threshold else 0

    def getLabelVec(self, data):
        self._checkInputVecDimensions(data)
        labels = np.array([self.getLabel(datum, checkDim = False) for datum in data])
        return labels

    def getScore(self, datum):
        self._checkInputDimensions(datum)
        return self._function(datum)

    def getScoreVec(self, data):
        self._checkInputVecDimensions(data)
        return self._functionVec(data)
    

class Branin(Benchmark):
    
    def __init__(self, threshold=8):
        Benchmark.__init__(self, threshold)
        self.inputDim = 2
        self.direction = ComparisonDirectionPositive.LESS_THAN_TH

    def _function(self, datum):
        x1,x2 = datum[0], datum[1]
        return (x2 - 5.1/(4*pi**2)*x1**2 + x1*5/pi -6)**2 + 10 *(1 - 1/(8*pi))*cos(x1) + 10

class Hosaki(Benchmark):
    def __init__(self, threshold=-1):
        Benchmark.__init__(self, threshold)
        self.inputDim = 2
        self.direction = ComparisonDirectionPositive.LESS_THAN_TH

    def _function(self, datum):
        x1,x2 = datum[0], datum[1]
        return  (1 - 8*x1 + 7*x1**2 - 7*x1**3 / 3 + x1**4 / 4) * x2**2 * exp(-x2)
    
class DistanceFromOrigin(Benchmark):
    def __init__(self, threshold = 1, inputDim = 3, center = None):
        Benchmark.__init__(self, threshold = threshold)
        self.direction = ComparisonDirectionPositive.LESS_THAN_TH
        self.center = np.zeros((inputDim,)) if center is None else np.array(center)
        self.inputDim = inputDim
        
    def _function(self, datum):
        return math.sqrt(sum((datum-self.center)**2))
        