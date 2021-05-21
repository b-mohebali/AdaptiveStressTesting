from geneticalgorithm import geneticalgorithm as ga
from .Sampling import SampleSpace
import numpy as np
from abc import ABC, abstractmethod
from ActiveLearning.benchmarks import Benchmark
from scipy.linalg import norm
from math import *

# TODO: This is going to be the parent class for all the optimizers used for selecting a point in a 
#   design space. 
class GA_Optimizer(ABC):
    def __init__(self, 
                space: SampleSpace, 
                batchSize: int = 1, 
                convergence_curve = True, 
                progress_bar = True):
        self.space = space
        self.clf = None 
        self.convergence_curve = convergence_curve
        self.progress_bar = progress_bar
        self.batchSize = batchSize
        self.currentSpaceSamples = None
    
    # Batch size getter and setter:
    def setBatchSize(self,batchSize):
        self._batchSize = batchSize
    def getBatchSize(self):
        return self._batchSize
    batchSize = property(getBatchSize, setBatchSize)

    @abstractmethod
    def objFunction(self, X):
        pass
    
    def getModel(self):
        algoParam = {'max_num_iteration': 40,
            'population_size':600,
            'mutation_probability':0.1,
            'elit_ratio': 0,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type':'uniform',
            'max_iteration_without_improv':None}
        gaModel = ga(function = self.objFunction, 
                        dimension = self.space.dNum, 
                        variable_type = 'real', 
                        variable_boundaries= self.space.getAllDimensionBounds(),
                        convergence_curve=self.convergence_curve,
                        algorithm_parameters=algoParam,
                        progress_bar=self.progress_bar)
        return gaModel

    def findNextPoints(self,pointNum=None):
        pointNum = self.batchSize if pointNum is None else pointNum
        newPointsFound = []
        self.currentSpaceSamples = self.space.samples
        for _ in range(pointNum):
            gaModel = self.getModel()
            gaModel.run()
            newPoint = gaModel.output_dict['variable']
            newPointsFound.append(newPoint)
            self.addPointToSampleList(newPoint)
        return np.array(newPointsFound)
    
    def addPointToSampleList(self, point):
        self.currentSpaceSamples = np.append(self.currentSpaceSamples, point.reshape(1,len(point)),axis=0)    


# The wrapper class for the genetic algorithm that solves the exploitation problem:
class GA_Exploiter(GA_Optimizer):
    def __init__(self, 
                space: SampleSpace, 
                epsilon: float, 
                clf,
                batchSize: int = 1, 
                convergence_curve = True, 
                progress_bar = True):
        GA_Optimizer.__init__(self, 
                    space = space,
                    batchSize=batchSize,
                    convergence_curve=convergence_curve, 
                    progress_bar=progress_bar)
        self.epsilon = epsilon                   
        self.clf = clf
    
    def objFunction(self, X):
        dist = self.space.nearestPointDistance(X, self.currentSpaceSamples)
        pen = 0
        df = self.clf.decision_function(X.reshape(1,len(X)))
        if abs(df) > self.epsilon:
            pen = abs(df) *100
        return -1 * dist + pen 

class GA_Explorer(GA_Optimizer):
    def __init__(self, space: SampleSpace, 
                batchSize: int = 1, 
                convergence_curve = True, 
                progress_bar = True,
                beta:float = 100):
        GA_Optimizer.__init__(self, 
                        space = space, 
                        batchSize=batchSize, 
                        convergence_curve=convergence_curve, 
                        progress_bar=progress_bar)
        self.beta = beta 

    def objFunction(self, X):
        return sum([exp(-self.beta * norm(p-X)) for p in self.currentSpaceSamples])


         