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
class BadInputDimension(Exception):
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
        notEnough = NotEnoughArguments(f'Not enough arguments for Branin Function. Please pass only {self.inputDim} arguments.')
        tooMany = TooManyArguments(f'Too many arguments for Branin function. Please pass only {self.inputDim} arguments.')
        if len(datum) != self.inputDim:
            exept = notEnough if len(datum) < self.inputDim else tooMany 
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
    
class DistanceFromCenter(Benchmark):
    def __init__(self, threshold = 1, inputDim = 3, center = None):
        Benchmark.__init__(self, threshold = threshold)
        self.direction = ComparisonDirectionPositive.LESS_THAN_TH
        self.center = np.zeros((inputDim,)) if center is None else np.array(center)
        self.inputDim = inputDim
        
    def _function(self, datum):
        return math.sqrt(sum((datum-self.center)**2))

class CorridorBenchmark(Benchmark):
    def __init__(self, threshold=0):
        Benchmark.__init__(self,threshold)
        self.inputDim = 2 
        self.direction = ComparisonDirectionPositive.LESS_THAN_TH

    def _function(self, datum):
        x1,x2 = datum[0], datum[1]
        return x2 - abs(tan(x1/2 + 2)) - 3


class ThreeD1(Benchmark):
    '''
        This is the benchmark based on Equation 5.30 on Mohebali's prospectus. Although the dimension of the boundary can be arbitrary, this implementation chose 3 as the number of dimensions. 
    '''
    def __init__(self, threshold=0):
        Benchmark.__init__(self,threshold)
        self.inputDim = 3
    
    def _function(self, datum):
        x1,x2,x3 = datum[0], datum[1],datum[2]
        return (x1-2)**2 + x2**2 + (x3+2)**2 - 3*x1*x2*x3 + 1

class FourD(Benchmark):
    def __init__(self, threshold=0):
        Benchmark.__init__(self, threshold)
        self.inputDim = 4 

    def _function(self, datum):
        x1,x2,x3,x4 = datum
        g = (x1-2)**2 +x2**2+(x3+2)**2+(x4-2)**2 - 3*(x1*x2*x3 + x2*x3*x4) + 1
        return g 

class ArbitraryDimension(Benchmark):
    beta = [-1,0,1]

    def __init__(self,inputDim, threshold=0):
        if inputDim < 3:
            raise BadInputDimension('The input dimension must be at least 3.')
        Benchmark.__init__(self,threshold)
        self.inputDim = inputDim

    def _function(self, datum):
        A = np.sum([(datum[i] + 2*self.beta[i%3])**2 for i in range(self.inputDim)])
        B = -3 * np.sum([(datum[_]*datum[_+1]*datum[_+2]) for _ in range(self.inputDim-2)])
        g = A + B + 1
        return g

class BumpyFunc(Benchmark):
    def __init__(self, threshold=0):
        super().__init__(threshold)
        self.inputDim = 2 
        self.direction = ComparisonDirectionPositive.MORE_THAN_TH

    def _function(self, datum):
        x1 = datum[0]
        x2 = datum[1]
        return 3*exp(-2 * (x1-4)**2) + exp(-2 * (x1 - 7)**2) + exp(x1/10) - x2 

class SineFunc(Benchmark):
    def __init__(self, threshold = 0):
        Benchmark.__init__(self,threshold)
        self.inputDim = 2 
    
    def _function(self, datum):
        x1,x2 = datum 
        return x2 - 2 * sin(x1) - 5 


class TrainedSvmClassifier(Benchmark):
    """
    This class takes an already trained classifier and uses it as the benchmark for the process. The purpose is for the cases when the actual solution is not available but a classifier can be trained with a high number of samples.

    NOTE: The decision_function() function is specific to the SVM classifier and may no be available for other types of classifiers. 
    
    NOTE: The threshold for this classifier depends on the decision function being the probability or just the output of the decision function.
    """
    def __init__(self, classifier, inputDim, threshold=0):
        Benchmark.__init__(self, threshold)
        self.threshold = 0.5 if classifier.probability else 0
        self.classifier = classifier
        self.inputDim = inputDim

    # NOTE: Needs testing once the mother sample is done.
    def _function(self, datum):
        return self.classifier.decision_function(datum.reshape(1,self.inputDim))
    
    def isProbability(self):
        return self.classifier.probability
    
    def predict(self, data):
        return self.classifier.predict(data)