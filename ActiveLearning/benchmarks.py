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
        self._functionVec = np.vectorize(self._function)
        self.direction = None
       
    def _checkInputDimensions(self, *args):
        if len(args) != self.inputDim:
            exept = NotEnoughArguments(f'Not enough arguments for Branin Function. Please pass only {self.inputDim} arguments.') if len(args) < self.inputDim else TooManyArguments(f'Too many arguments for Branin function. Please pass only {self.inputDim} arguments.')
            raise exept 
        
    def _checkInputVecDimensions(self, *args):
        self._checkInputDimensions(*args)
        if self.inputDim > 0 and len(args[0]) > 1:
            n = len(args[0])
            for i in range(1,len(args)):
                if n != len(args[i]):
                    raise InputDimensionsDoNotMatch('Input dimensions do not match. Please pass vectors of the same length.')
        return 

    def getLabel(self, *args):
        self._checkInputDimensions(*args)
        s = self.score(*args, checkDim=False)
        if self.direction == ComparisonDirectionPositive.LESS_THAN_TH:
            return 1 if s < self.threshold else 0
        return 1 if s > self.threshold else 0


    def getLabelVec(self, *args):
        self._checkInputDimensions(*args)
        self._checkInputVecDimensions(*args)
        sVec = self._functionVec(*args)
        labels = np.zeros(shape = (len(sVec),), dtype = int)
        labels[sVec < self.threshold] = 1   
        return labels

    def score(self, *args, checkDim = True):
        if checkDim:
            self._checkInputDimensions(args)
        return self._function(args)

    def scoreVec(self,*args, checkDim = True):
        if checkDim:
            self._checkInputVecDimensions(*args)
        return self._functionVec(*args)

    @abstractmethod
    def _function(self, args):
        pass

class Branin(Benchmark):
    
    def __init__(self, threshold=8):
        Benchmark.__init__(self, threshold)
        self.inputDim = 2
        self.direction = ComparisonDirectionPositive.LESS_THAN_TH

    def _function(self, *args):
        x1,x2 = args[0], args[1]
        return (x2 - 5.1/(4*pi**2)*x1**2 + x1*5/pi -6)**2 + 10 *(1 - 1/(8*pi))*cos(x1) + 10

class Hosaki(Benchmark):
    def __init__(self, threshold=-1):
        Benchmark.__init__(self, threshold)
        self.inputDim = 2
        self.direction = ComparisonDirectionPositive.LESS_THAN_TH

    def _function(self, *args):
        x1,x2 = args[0], args[1]
        return  (1 - 8*x1 + 7*x1**2 - 7*x1**3 / 3 + x1**4 / 4) * x2**2 * exp(-x2)
    
class DistanceFromOrigin(Benchmark):
    def __init__(self, threshold = 1, inputDim = 3, root = False):
        Benchmark.__init__(self, threshold = threshold)
        self.direction = ComparisonDirectionPositive.LESS_THAN_TH
        self.root = root
        self.inputDim = inputDim
        
    def _function(self, *args):
        s = sum([arg**2 for arg in args])
        return math.sqrt(s)
        