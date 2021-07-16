from yamlParseObjects.yamlObjects import simulationConfig
from geneticalgorithm import geneticalgorithm as ga
# from .Sampling import ConvergenceSample, SampleSpace
from ActiveLearning.Sampling import ConvergenceSample, SampleSpace
import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import norm
from math import *
from sklearn import svm
from enum import Enum

class Exploration_Type(Enum):
    VORONOI = 0
    RBF = 1



# TODO: This is going to be the parent class for all the optimizers used for selecting a point in a 
#   design space. 
class GA_Optimizer(ABC):
    def __init__(self, 
                space: SampleSpace, 
                batchSize: int = 1, 
                convergence_curve = True, 
                progress_bar = True,
                constraints = []):
        self.space = space
        self.clf = None 
        self.convergence_curve = convergence_curve
        self.progress_bar = progress_bar
        self.batchSize = batchSize
        self.currentSpaceSamples = None
        self.ranges = space.getAllDimensionsRanges()
        self.constraints = constraints

    # Batch size getter and setter:
    def setBatchSize(self,batchSize):
        self._batchSize = batchSize
    def getBatchSize(self):
        return self._batchSize
    batchSize = property(getBatchSize, setBatchSize)

    @abstractmethod
    def objFunction(self, X):
        pass
    
    def constrainedObjFunction(self, X):
        """
            This function applies a set of constraints, defined as functions of n-dimensional vectors that return boolean values, and penalize the objective function if any of the constraints is True. If the constraints list is empty or none is True the same objective function value is passed.
        """
        initialValue = self.objFunction(X)
        if len(self.constraints)==0:
            return initialValue
        results = [constraint(X) for constraint in self.constraints] if self.constraints else [True]
        if not all(results):
            initialValue += 1e6
        return initialValue

    def getModel(self):
        algoParam = {'max_num_iteration': 40,
            'population_size':600,
            'mutation_probability':0.1,
            'elit_ratio': 0,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type':'uniform',
            'max_iteration_without_improv':None}
        gaModel = ga(function = self.constrainedObjFunction, 
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
                progress_bar = True,
                constraints = []):
        GA_Optimizer.__init__(self, 
                    space = space,
                    batchSize=batchSize,
                    convergence_curve=convergence_curve, 
                    progress_bar=progress_bar,
                    constraints = constraints)
        self.epsilon = epsilon                   
        self.clf = clf
    
    def objFunction(self, X):
        dist = self.space.nearestPointDistance(X, self.currentSpaceSamples, normalize=True)
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
                beta:float = 1,
                constraints = []):
        GA_Optimizer.__init__(self, 
                        space = space, 
                        batchSize=batchSize, 
                        convergence_curve=convergence_curve, 
                        progress_bar=progress_bar,
                        constraints=constraints)
        self.beta = beta 
        
    # The scale of the dimensions are normalized so that it does not affect the 
    # relative distance between the points of the dataset and the prospective 
    # solutions.
    def objFunction(self, X):
        return sum([exp(-self.beta*norm(np.divide(p-X,self.ranges))**2) for p in self.currentSpaceSamples])

class GA_Voronoi_Explorer(GA_Optimizer):
    def __init__(self, 
            space: SampleSpace, 
            batchSize: int = 1, 
            convergence_curve = True, 
            progress_bar = True,
            constraints = []):
        GA_Optimizer.__init__(self, 
                        space = space, 
                        batchSize=batchSize, 
                        convergence_curve=convergence_curve, 
                        progress_bar=progress_bar,
                        constraints = constraints)
        self.ranges = space.getAllDimensionsRanges()
    
    def objFunction(self, X):
        return -1 * np.min(np.linalg.norm(np.divide(self.currentSpaceSamples - X, self.ranges), axis = 1))

class ResourceAllocator:
    def __init__(self,
            space: SampleSpace,
            convSample: ConvergenceSample,
            l:float,
            epsilon: float,
            simConfig:simulationConfig):
        self.space = space
        self.convSample = convSample
        self.l = l
        if epsilon > 0.5:
            raise ValueError('The value of epsilon is invalid.')
        self.epsilon = epsilon
        self.budget = simConfig.sampleBudget
        self.batchSize = simConfig.batchSize
    
    # The function that drives the calculation algorithm:
    def allocateResources(self):
        pass

    

    # Step 3A:
    def _exprBounds(self, currentBudget: int):
        budgetRatio = currentBudget / self.budget
        lowerBound = self.epsilon * (1 - budgetRatio)
        upperBound = (1-self.epsilon) * (1 - budgetRatio)
        return lowerBound, upperBound

    # Step 3B:
    def _exptBounds(self, currentBudget:int):
        budgetRatio = currentBudget / self.budget
        lowerBound = budgetRatio + self.epsilon * (1 - budgetRatio)
        upperBound = 1 - self.epsilon*(1 - budgetRatio)
        return lowerBound, upperBound

    # Step 4:
    def _calTendency(self, dynamicTendency, lowerBound, upperBound):
        return max(min(dynamicTendency, upperBound), lowerBound)

    # Step 5:
    def _calSampleNumbers(self, exprTen, exptTen):
        # Calculating resource allocation coefficients:
        R_expr = exprTen / (exprTen + exptTen)
        R_expt = exptTen / (exprTen + exptTen)

        # Calculating how many samples will be spent on each type:
        expr_samples = round(R_expr * self.batchSize)
        expt_samples = round(R_expt * self.batchSize)

        # Returning the results:
        return expr_samples, expt_samples


def allocateResources(mainSamples,
                    mainLabels,
                    exploitSamples,
                    exploitLabels,
                    exploreSamples,
                    exploreLabels,
                    convSample: ConvergenceSample):
    exploitBudget = 0
    exploreBudget = 0

    # Training the reference classifier:
    mainClf = svm.SVC(kernel = 'rbf', C = 1000)
    mainClf.fit(mainSamples, mainLabels)
    # Constructing two sets of data: 
    #   - Main data + exploitation data
    #   - Main data + exploration data
    exploitData = np.append(np.copy(mainSamples), exploitSamples, axis = 0)
    exploreData = np.append(np.copy(mainSamples), exploreSamples, axis = 0)
    exploitLabels = np.append(np.copy(mainLabels), exploitLabels, axis = 0)
    exploreLabels = np.append(np.copy(mainLabels), exploreLabels, axis = 0)
    
    # training a set of two classifiers on the data that we constructed:
    clf_exploit = svm.SVC(kernel = 'rbf', C = 1000)
    clf_explore = svm.SVC(kernel = 'rbf', C = 1000)
    clf_exploit.fit(exploitData, exploitLabels)
    clf_explore.fit(exploreData, exploreLabels)
    
    # Getting the difference that each group of points make in the boundary:
    exploitDiff = convSample.getDifferenceMeasure(mainClf, clf_exploit)
    exploreDiff = convSample.getDifferenceMeasure(mainClf, clf_explore)

    # Calculating the budgets for each part of the algorithm:

    # Returning the results:
    return exploitBudget, exploreBudget

