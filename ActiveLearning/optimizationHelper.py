from yamlParseObjects.yamlObjects import ResourceAllocationReport, simulationConfig
from geneticalgorithm import geneticalgorithm as ga
# from .Sampling import ConvergenceSample, SampleSpace
from ActiveLearning.Sampling import ConvergenceSample, SampleSpace, StandardClassifier
import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import norm
from math import *
from sklearn import svm
from enum import Enum
import yaml
import heapq

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
            This function applies a set of constraints, defined as functions of n-dimensional vectors that return boolean values, and penalize the objective function if any of the constraints is violated. If the constraints list is empty or none is violated the same objective function value is passed.
        """
        initialValue = self.objFunction(X)
        if len(self.constraints)==0:
            return initialValue
        results = [constraint(X) for constraint in self.constraints]
        # Penalizing the objective function if any of the constraints are violated. 
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
        pointNum = int(pointNum)
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
        self.prevExploitTendency = 1
        self.prevExploreTendency = 1
    
    def objFunction(self, X):
        return -1 * np.min(np.linalg.norm(np.divide(self.currentSpaceSamples - X, self.ranges), axis = 1))

class GA_Convergence_Sampler(GA_Optimizer):
    def __init__(self, 
                space: SampleSpace,
                clf: StandardClassifier,  
                batchSize: int = 1, 
                convergence_curve=True, 
                progress_bar=True, 
                constraints=[]):
        GA_Optimizer.__init__(self, 
                        space, 
                        batchSize=batchSize, 
                        convergence_curve=convergence_curve, 
                        progress_bar=progress_bar, 
                        constraints=constraints)
        self.clf = clf  
        self.ranges = space.getAllDimensionsRanges()

    def getFittingSupportVector(self):
        """
            Finds the list of the support vectors that fit the conditions for sampling in the next iteration.
            TODO: The batch sampling for this type of sample is still under investigation. 
                So the current implementation samples a single point.
        """
        bestPoints = {} 
        SVs = self.clf.getSupportVectors(standard = True)
        labels = self.clf.predict(SVs)
        for idx, sv in enumerate(SVs):
            svLabel = labels[idx]
            svDistance = self._minDistanceFromOppositeClass(sv, svLabel)
            # print(f'Found point for SV # {idx+1}: {foundPoint}, SV itself: {sv}')
            bestPoints[svDistance] = (sv,idx)      
        bestDistance = max(bestPoints.keys())
        # Radius is an instance variable that will be used in the objective function.
        self.R = bestDistance / 2 
        fittingSv, fittingIdx = bestPoints[bestDistance]
        label = labels[fittingIdx]
        return fittingSv, label


    def _minDistanceFromOppositeClass(self, sv, label):
        oppLabel = label ^ 1 
        distances = np.linalg.norm(np.divide(self.space.samples-sv,self.ranges), axis = 1)
        distance = min(distances[self.space._eval_labels==oppLabel])
        return distance 
    
    def findNextPoints(self, pointNum=1):
        sv, label = self.getFittingSupportVector()
        self.sv = sv 
        self.label = label 
        return super().findNextPoints(pointNum=pointNum)
    
    def objFunction(self, X):
        '''
            NOTE: self.sv is normalized. when we want to meassure the distance with the input point, we need to normalize the point to the correct range as well. 
        '''
        # Decision function of the input point:
        df = self.clf.decision_function(X.reshape(1,len(X)))
        # Initial objective function is the decision function times the label of the SV. the reason is that the new point has to be on the opposite side of the boundary from the designated SV:
        f = df * self.label
        # Scaling the input point only for calculating its distance to the designated SV:
        xScaled = self.clf.scaler.transform(X.reshape(1,len(X)))
        distance = np.linalg.norm(xScaled - self.sv)
        # Penalizing the objective function if the distance to the designated SV is more than R or the objective function becomes positive:
        if distance > self.R or f > 0:
            f += 1e6
        return f
    


class ResourceAllocator:
    """
        Resource allocator class. Takes the exploitative and exploratory samples at each iteration and tells the algorithm how many explorative or exploitative samples have to be calculated for the next iteration. 

        Attributes: 
            - space: The sample space where all the samples are and will be.
            - convSample: The large random (or quasi random) sample of points used to measure the change in the hypothesis between the iterations. 
            - simConfig: The simulation configuration object containing information about the total budget allocated for the whole process. 
            - epsilon: Parameter that controls the range of possible variation for the tendencies.  
            - outputLocation: Location of the report that will be saved. 
            - l: The lambda parameter
    """

    def __init__(self,
            convSample: ConvergenceSample,
            simConfig:simulationConfig = None,
            outputLocation = None,
            initSample = None):
        self.convSample = convSample
        self.l = simConfig.resourceLambda
        self.epsilon = simConfig.resourceEpsilon
        if self.epsilon > 0.5:
            raise ValueError('The value of epsilon is invalid.')
        self.budget = simConfig.sampleBudget
        self.batchSize = simConfig.batchSize
        self.budget = simConfig.sampleBudget 
        self.initialSampleSize = simConfig.initialSampleSize if initSample is None else initSample 
        self.reports = [] 
        self.outputLocation = outputLocation
        self.reportName = f'{outputLocation}/ResourceAllocationReport.yaml'
        self.prevExploitTendency = 1 # Placeholder for saving the exploitation tendency in each iteration
        self.prevExploreTendency = 1 # Placeholder for the exploration tendency.
        print('Resource Allocator: budget size is ', self.budget)
        print('Resource Allocator: Initial sample size is', self.initialSampleSize)

    def saveReports(self):
        """
            Saving all the reports in a separate yaml file for debugging.
        """
        print('Saving the resource allocation report...')
        with open(self.reportName, 'w') as yamlFile:
            yaml.dump_all(self.reports, yamlFile)
        

    
    # The function that drives the calculation algorithm:
    def allocateResources(self, 
                        mainSamples, 
                        mainLabels, 
                        exploitSamples, 
                        exploitLabels, 
                        exploreSamples, 
                        exploreLabels,
                        saveReport = False,
                        tendency_bounds = False):
        exploitBudget = 0 
        exploreBudget = 0

        sampleSize = len(mainLabels) + len(exploitLabels) + len(exploreLabels)
        # Training the reference classifier:
        mainClf = svm.SVC(kernel = 'rbf', C = 1000)
        mainClf.fit(mainSamples, mainLabels)

        # Constructing two sets of data: 
        #   - Main data + exploitation data
        #   - Main data + exploration data
        if len(exploitLabels)>0:
            exploitData = np.append(np.copy(mainSamples), exploitSamples, axis = 0)
            exploitLabels = np.append(np.copy(mainLabels), exploitLabels, axis = 0)
            clf_exploit = svm.SVC(kernel = 'rbf', C = 1000)
            clf_exploit.fit(exploitData, exploitLabels)
            exploitDiff = self.convSample.getDifferenceMeasure(mainClf, clf_exploit)
            exploitTendency = exploitDiff * self.l
            exploitBounds = self._exptBounds(sampleSize)
            # exploitTendency = self._calTendency(exploitTendency, exploitBounds)
        else:
            exploitTendency = self.prevExploitTendency    
        
        if len(exploreLabels) > 0:
            exploreLabels = np.append(np.copy(mainLabels), exploreLabels, axis = 0)
            exploreData = np.append(np.copy(mainSamples), exploreSamples, axis = 0)
            clf_explore = svm.SVC(kernel = 'rbf', C = 1000)
            clf_explore.fit(exploreData, exploreLabels)
            exploreDiff = self.convSample.getDifferenceMeasure(mainClf, clf_explore)
            exploreTendency = exploreDiff * self.l
            exploreBounds = self._exprBounds(sampleSize)
            # exploreTendency = self._calTendency(exploreTendency, exploreBounds)
        else: 
            exploreTendency = self.prevExploreTendency        

        exploitBudget,exploreBudget, r_expt, r_expr = self._calSampleNumbers(exploitTendency, exploreTendency, return_Rs=True)

        # Creating the report for this iteration: 
        if saveReport:
            resourceReport = ResourceAllocationReport()
            resourceReport.iteration_number= (sampleSize - self.initialSampleSize)/self.batchSize 
            resourceReport.exploration_tendency = float(exploreTendency)
            resourceReport.exploitation_tendency = float(exploitTendency)
            resourceReport.R_explore = float(r_expr)
            resourceReport.R_exploit = float(r_expt)
            if len(exploitLabels) > 0: 
                resourceReport.exploitBounds = [float(_) for _ in exploitBounds]
                resourceReport.calculated_exploitation_budget = float(exploitBudget)
            
            if len(exploreLabels) > 0: 
                resourceReport.exploreBounds = [float(_) for _ in exploreBounds]
                resourceReport.calculated_exploration_budget = float(exploreBudget)
            
            resourceReport.budgetRatio = float(sampleSize / self.budget)
            self.reports.append(resourceReport)
            self.saveReports()

        self.prevExploitTendency = exploitTendency
        self.prevExploreTendency = exploreTendency
        
        # Returning the results of the calculations:
        return exploitBudget, exploreBudget

    

    def _exprBounds(self, sampleSize: int):
        """
            Contains the logic that controls the bounds on the exploration tendency of the algorithm.
        """
        budgetRatio = sampleSize / self.budget
        print('Resource Allocator: Budget ratio for exploration bounds: ', budgetRatio)
        lowerBound = self.epsilon * (1 - budgetRatio)
        upperBound = (1-self.epsilon) * (1 - budgetRatio)
        return lowerBound, upperBound


    def _exptBounds(self, sampleSize:int):
        """
            Contains the logic that controls the bounds on the exploitation tendency of the algorithm. 
        """
        budgetRatio = sampleSize / self.budget
        print('Resource Allocator: Budget ratio for exploitation bounds', budgetRatio)
        lowerBound = budgetRatio + self.epsilon * (1 - budgetRatio)
        upperBound = 1 - self.epsilon*(1 - budgetRatio)
        return lowerBound, upperBound

    # Step 4:
    def _calTendency(self, dynamicTendency, bounds):
        lowerBound = bounds[0]
        upperBound = bounds[1]
        return max(min(dynamicTendency, upperBound), lowerBound)

    # Step 5:
    def _calSampleNumbers(self, exptTen, exprTen, return_Rs = False):
        # Calculating resource allocation coefficients:
        R_expr = exprTen / (exprTen + exptTen)
        R_expt = exptTen / (exprTen + exptTen)

        # Calculating how many samples will be spent on each type:
        expr_samples = round(R_expr * self.batchSize)
        expt_samples = round(R_expt * self.batchSize)

        # Returning the results:
        if return_Rs:
            return expt_samples, expr_samples, R_expt, R_expr
        return expt_samples, expr_samples

