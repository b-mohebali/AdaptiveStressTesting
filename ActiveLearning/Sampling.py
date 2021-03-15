from yamlParseObjects.yamlObjects import * 
from typing import List
from samply.hypercube import cvt, lhs
from enum import Enum
from ActiveLearning.benchmarks import *
from sklearn import svm


class InsufficientInformation(Exception):
    pass


# This class represents a single 
class Dimension():
    def __init__(self, varConfig: variableConfig):
        self.name = varConfig.name
        self.bounds = [varConfig.lowerLimit, varConfig.upperLimit]
        self.range = self.bounds[1] - self.bounds[0]
        self.description = varConfig.description



"""
    This class represents the sample space (or design space). The functionality 
    is meant to store the information about the samples already taken and evaluated,
    as well as the dimension information like the name of the dimensions and their 
    bounds. Other functionalities facilitate the sampling process.
"""
class InitialSampleMethod(Enum):
    CVT = 0
    LHS = 1
    Sobol = 2

class Space():
    def __init__(self, variableList: List[variableConfig], initialSampleCount = 20, benchmark: Benchmark = None):
        self.initialSampleCount = initialSampleCount
        self.dimensions = []
        for varConfig in variableList:
            self.dimensions.append(Dimension(varConfig = varConfig))
        self.dNum = len(self.dimensions)
        self.convPointsNum = 100 * 5 ** self.dNum
        self.samples = []
        self.eval_labels = []
        self.benchmark = benchmark
        self.clf = None    
        self.convPoints = None
        self.sampleConvPoints()
        self.pastConvLabels = np.zeros(shape = (self.convPointsNum,),dtype = float)
        self.currentConvLabels = None

    def setBenchmark(self, benchmark: Benchmark):
        self._benchmark = benchmark
    def getBenchmark(self):
        return self._benchmark 
    benchmark = property(getBenchmark, setBenchmark)   

    def setClassifier(self, classifier):
        self._clf = classifier
    def getClassifier(self):
        return self._clf
    clf = property(getClassifier, setClassifier)

    def getSamplesCopy(self):
        return np.copy(self.samples)

    """
    This funnction returns a measure of how much the hypothesis has moved since the last time 
    that this measure was calculated. 
    """
    def getChangeMeasure(self, updateConvLabels = False, percent = False):
        self.currentConvLabels = self.clf.predict(self.convPoints)
        diff = np.sum(np.abs(self.currentConvLabels - self.pastConvLabels)) / self.convPointsNum
        if updateConvLabels:
            self.pastConvLabels = self.currentConvLabels
        return diff*100.0 if percent else diff
        
    """
    The accuracy is calculated as the percentage of the convergence points that are 
    correctly classified by the classifier that is trained on the data. 
    It compares the labels coming from the classifier and the benchmark object.
    """
    def getAccuracyMeasure(self, benchmark = None, percent = False):
        if self.convPoints is None:
            raise InsufficientInformation("The convergence points are not sampled yet.")
        if benchmark is not None:
            self.benchmark = benchmark
        pred_labels = self.clf.predict(self.convPoints)
        real_labels = self.benchmark.getLabelVec(self.convPoints)
        diff = 1 - np.sum(np.abs(pred_labels - real_labels)) / self.convPointsNum
        return diff * 100.0 if percent else diff


    def clf_decision_function(self,point):
        return self.clf.decision_function(point)

    # Number of conversion points must rise exponentially by the dimension of the space
    def sampleConvPoints(self):
        if self.convPoints is not None:
            return
        # Sampling the convergence points using LHS to save some time. 
        # Note that the sample is very dense so we do not need the CVT to ensure spread
        convPoints = lhs(count = self.convPointsNum, dimensionality=self.dNum)
        # Scaling each of the dimensions according to the dimension ranges of the space. 
        for dimIndex, dimension in enumerate(self.dimensions):
            convPoints[:,dimIndex] *= dimension.range 
            convPoints[:,dimIndex] += dimension.bounds[0] 
        self.convPoints = convPoints
       

    def _labelConvPoints(self):
        # First we need the sample. IF the sample is already taken it will not change until the end of the process.
        if self.convPoints is None:
            self.sampleConvPoints()

    # def addPointToSampleList(self, point):
    #     self.samples = np.append(self.samples, point.reshape(1,len(point)),axis=0)    

    def getAllDimensionNames(self):
        return [dim.name for dim in self.dimensions]
    
    def getAllDimensionDescriptions(self):
        return [dim.description for dim in self.dimensions]
    
    def getAllDimensionBounds(self):
        return np.array([dim.bounds for dim in self.dimensions])

    def generateInitialSample(self, method = InitialSampleMethod.CVT):
        # First set of samples. All the dimensions are between 0 and 1
        if method == InitialSampleMethod.CVT:
            samples = cvt(count = self.initialSampleCount, dimensionality=self.dNum)
        elif method == InitialSampleMethod.LHS:
            samples = lhs(count = self.initialSampleCount, dimensionality=self.dNum)
        # all the samples are then scaled to their appropriate range:
        # samples = np.array(samples)
        for dimIndex, dimension in enumerate(self.dimensions):
            samples[:,dimIndex] *= dimension.range # Samples times the range of the dimension
            samples[:,dimIndex] += dimension.bounds[0] # Samples shifted by the lower bound.
        self.samples = samples
        return

    def getBenchmarkLabels(self, benchmark:Benchmark = None, updateClassifier = False):
        
        if benchmark is not None:
            self.benchmark = benchmark
        self.eval_labels = self.benchmark.getLabelVec(self.samples)
        # TODO updating the classifier and effectively the decision boundary in case the labels are updated.
        if updateClassifier:
            pass
        return self.eval_labels

    def addPointsToSampleList(self, points):
        self.samples = np.append(self.samples, points, axis=0)

    # Nearest point from the dataset:
    def nearestPointDistance(self, X, samplesList = None):
        if samplesList is None:
            return np.min(np.linalg.norm(self.samples - X, axis=1))
        return np.min(np.linalg.norm(samplesList - X, axis=1))

    # TODO This method has to be able to return the accuracy measure and the measure of change in hypothesis.
    def fit_classifier(self):
        if len(self.samples)==0:
            raise InsufficientInformation('No data points present in the space.')
        if len(self.eval_labels)==0:
            raise InsufficientInformation('The data is not labeled yet.')
        if len(self.eval_labels) != len(self.samples):
            raise ValueError('Number of data points and labels do not match.')
        self.clf = svm.SVC(kernel = 'rbf', C = 1000)
        self.clf.fit(self.samples, self.eval_labels)
        return

    