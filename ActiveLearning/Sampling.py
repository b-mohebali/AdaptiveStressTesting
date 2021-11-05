from yamlParseObjects.yamlObjects import VariableConfig
from typing import List
from samply.hypercube import cvt, lhs, halton
from enum import Enum
from .benchmarks import Benchmark
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score
import numpy as np 

class InsufficientInformation(Exception):
    pass

class SampleNotEmpty(Exception):
    pass

# This class represents a single dimension in the design space:
class Dimension():
    def __init__(self, varConfig: VariableConfig):
        self.name = varConfig.name
        self.bounds = [varConfig.lowerLimit, varConfig.upperLimit]
        self.range = abs(self.bounds[1] - self.bounds[0])
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
    HALTON = 3

'''
    This class determines the type of the performance metric for the current iteration
    of hypothesis.
'''
class PerformanceMeasure(Enum):
    ACCURACY = 0
    PRECISION = 1
    RECALL = 2
    F1_SCORE = 3


"""
    New implementation of the space class with limited functionality. 
    NOTE: The functionality of the Space class was getting too much and so
        it was broken down into several classes.
"""
class SampleSpace():
    def __init__(self, 
                variableList: List[VariableConfig]):
        self.dimensions = []
        for varConfig in variableList:
            self.dimensions.append(Dimension(varConfig=varConfig))
            self._dNum = len(self.dimensions)
            self.convPointsNum = 100 * 5 ** self.dNum 
            self._samples = []
            self._eval_labels = []
        self.ranges = self.getAllDimensionsRanges()
        self.ones = np.ones(shape = (self.dNum,))
    
    # Defining the samples and labels as private fields with the type list 
    # Upon retrive, they are turned into numpy arrays. 
    def getSamples(self):
        return np.array(self._samples)
    samples = property(fget = getSamples)

    def getEvalLabels(self):
        return np.array(self._eval_labels)
    eval_labels = property(fget = getEvalLabels)

    # Defining the dimension number as a private field that can't be set.
    def getDimensionNumber(self):
        return self._dNum
    dNum = property(fget = getDimensionNumber)

    def getAllDimensionNames(self):
        return [dim.name for dim in self.dimensions]
    
    def getAllDimensionDescriptions(self):
        return [dim.description for dim in self.dimensions]
    
    def getAllDimensionsRanges(self):
        # Returns a d-dimensional array of the ranges of the dimensions in the space.
        return np.array([dim.range for dim in self.dimensions])
   
    def getAllDimensionBounds(self):
        # Returns a d x 2 vector of the upper bound and lower bound of each dimension in the space.
        return np.array([dim.bounds for dim in self.dimensions])
    
    def getSamplesNum(self):
        return len(self._samples)
    sampleNum = property(fget = getSamplesNum)

    # TODO: Formatting the stored data: 


    # TODO: Adding samples with labels to the dataset:
    def addSample(self, dataPoint, label):
        self._samples.append(dataPoint)
        self._eval_labels.append(label)

    def addSamples(self, dataPoints, labels):
        if len(labels) != len(dataPoints):
            raise ValueError('The number of samples and labels do not match')
        for idx, dataPoint in enumerate(dataPoints):
            self.addSample(dataPoint, labels[idx])
        return 

    # Nearest point from the dataset:
    # NOTE: This is copied from the old implementation of the space class.
    def nearestPointDistance(self, X, samplesList = None, normalize = False):
        if normalize:
            r = self.ranges
        else:
            r = self.ones
        if samplesList is None:
            result = np.min(np.linalg.norm(np.divide(self.samples-X,r),axis=1))
        else:
            result = np.min(np.linalg.norm(np.divide(samplesList-X,r),axis=1))
        return result
"""
    NOTE: The initial classifier used here is SVM. The necessity for any other type of 
    classifier was not felt at this point.
    - Also it may be shown that the SVM class does not require a simple wrapper like 
        this. 
"""


# TODO: This class holds the sample used for convergence check at each iteration:
class ConvergenceSample():
    '''
    Contains the convergence sample used for checking the amount of 
    change in the hypothesis. It also provides some functionality for measuring 
    the mentioned change.

    NOTE:   The approximation of the accuracy of the current iteration of the 
            hypothesis is only possible if the actual function is a low-cost 
            benchmark and not an expensive computational model.  
    '''
    def __init__(self, 
                space: SampleSpace,
                constraints = [],
                size = None):
        self.size = 100 * 5**space.dNum if size is None else size
        self.samples = generateInitialSample(space, 
                                            sampleSize=self.size, 
                                            method=InitialSampleMethod.HALTON,
                                            constraints=constraints)
        self.size = len(self.samples)
        self.pastLabels = np.zeros(shape=(self.size,), dtype = float)
        self.currentLabels = None

    def getChangeMeasure(self, 
                        classifier,
                        updateLabels = False, 
                        percent = False):
        self.currentLabels = classifier.predict(self.samples)
        diff = np.sum(np.abs(self.currentLabels - self.pastLabels))/ self.size
        if updateLabels:
            self.pastLabels = self.currentLabels
        return (diff * 100.0) if percent else diff 
    
    @classmethod
    def _movingAverage(cls, measure, n=1):
        if n < 1:
            raise ValueError('n cannot be less than 1.')
        startInd = max(0, len(measure)-n)
        return sum(measure[startInd:])/n
    
    @classmethod
    def movingAverageVec(cls, measure, n=1):
        movingAv = [] 
        for ind in range(len(measure)):
            endInd = ind + 1 
            startInd = max(0, endInd - n)
            movingAv.append(sum(measure[startInd: endInd])/n)
        return movingAv
        
    
    def getDifferenceMeasure(self, 
                        clf1,
                        clf2,
                        percent = False):
        '''
        Takes two classifiers and says how much they differ in 
            terms of how they classify a given sample space. 
            
        Inputs: 
            - clf1: First classifier
            - clf2: Second classifier
        
        Outputs: 
            - diff: A measure of difference between their decision boundaries, or
                how much of the space is labeled differently by these two 
                classifiers. 
        '''
        labels1 = clf1.predict(self.samples)
        labels2 = clf2.predict(self.samples)
        diff = np.sum(np.abs(labels1 - labels2))/self.size
        return diff if not percent else diff * 100.0 
        
    # TODO: Get a threshold for the classification. Right now the default threshold 
    #       is 0.5, which may not be the best for the application.
    def getPerformanceMetrics(self,
                            benchmark: Benchmark = None,
                            yTrue = None,
                            classifier = None,
                            percentage = True,
                            metricType: PerformanceMeasure = PerformanceMeasure.ACCURACY):
        """
        Takes a benckmark object and a classifier and compares the 
        prediction of the classifier with the benchmark. In case the classifier 
        is not provided, the last iteration of the labels are assumed to be the 
        predictions of the classifier.
            Inputs:
                - Benchmark: The benchmark object as implemented in the Benchmark.py file.
                - classifier: A classifier object that implement the predict() function
                - percentage: Whether the results is reported as percentage or 
                    a ratio between 0 and 1.
                - Metrics type: Choses what metric is reported between accuracy, 
                    precision, or recall
            outputs:
                - The performance metric as selected by the user. 
        """

        # If the classifier is not entered, the last set of labels are used for the 
        #   evaluation. 
        yPred = self.currentLabels if classifier is None else classifier.predict(self.samples)
        if yTrue is None: 
            yTrue = benchmark.getLabelVec(self.samples)
        if metricType == PerformanceMeasure.ACCURACY:
            metric = accuracy_score(yTrue, yPred)
        elif metricType == PerformanceMeasure.PRECISION:
            metric = precision_score(yTrue, yPred)
        elif metricType == PerformanceMeasure.RECALL:
            metric = recall_score(yTrue, yPred)
        elif metricType == PerformanceMeasure.F1_SCORE:
            percentage = False
            metric = f1_score(yTrue, yPred)
        return metric * 100 if percentage else metric
    

def getSamplePointsAsDict(dimNames, sampleList):
    """
    Returns a list of dictionaries containing the values of the 
    design variables for each experiment (one dict per experiment)
    
    Inputs:
        - dimNames: List of the name of the dimensions in the space        
        - sampleList: A List of lists containing the values for all the 
            sample points. The order is the same as the dimensions of the
            space passed to the function
        
    Outputs:
        - Samples dictionary: A list of dictionaries each haveing the name 
            and the value of the dimensions for each of the sample points.    
    """
    output = []
    for sample in sampleList:
        sampleDict = {}
        for idx, dimName in enumerate(dimNames):
            sampleDict[dimName] = sample[idx]
        output.append(sampleDict)
    return output 

def getAccuracyMeasure( convSample:ConvergenceSample,
                        measure: PerformanceMeasure,
                        classifier, 
                        benchmark: Benchmark, 
                        percent = False):
    '''
        Calculates the performance measure of the current iteration of the 
        hypothesis that is stored in the classifier argument.
    '''
    predLabels = classifier.predict(convSample.samples)
    realLabels = benchmark.getLabelVec(convSample.samples)
    if measure == PerformanceMeasure.ACCURACY:
        return accuracy_score(realLabels, predLabels) * 100 if percent else 1
    # TODO: Other types of performance measures (scores) 

class ActiveClassifier():

    def __init__(self, C=1000, kernel = 'rbf'):
        self.kernel = kernel
        self.C = C
        self.clf = None
    
    def fit_classifier(self, data, labels):
        if len(data) != len(labels):
            raise ValueError('Number of data points and labels do not match.')
        classifier = svm.SVC(kernel = self.kernel,C = self.C)
        classifier.fit(data, labels)
        self._clf = classifier
        return classifier

    # Defining the classifier as a private field:
    # NOTE: The only way to change the classifier is through training it with new data.
    #   This may change in the future.
    def getClassifier(self):
        return self._clf
    clf = property(fget = getClassifier)





def generateInitialSample(space: SampleSpace, 
                        sampleSize: int, 
                        method: InitialSampleMethod = InitialSampleMethod.CVT,
                        checkForEmptiness = False,
                        constraints = [],
                        resample = False):
    ''' This function samples from the entire space. 
        Can be used for the convergence samples as well without the check for 
        the emptiness of the space.

        Inputs: 
            - space:  Contains information about the number of dimensions and their range of variations.

            - method: CVT (Centroid Voronoi Tesselation)
                LHS (Latin Hypercube)

            - CheckForEmptiness: (True/False) Raises error if the sample list of the space is not empty, meaning that the initial sample is most likely taken. 

            - Constraints : List of the constraint functions that act on the sampled points and return boolean values showing whether the constraint is respected or violated. 

            - resample: Boolean input indicating whether the sample is going to be retaken to compensate for the rejected samples.

    '''
    if checkForEmptiness and len(space.samples) > 0:
        raise SampleNotEmpty('The space already contains samples.')
    if method == InitialSampleMethod.CVT:
        print('Generating the samples using CVT method. This may take a while...')
        samples = cvt(count = sampleSize, dimensionality= space.dNum, epsilon=0.0000001)
    elif method == InitialSampleMethod.LHS:
        print('Generating the samples using LHS method. This may take a while...')
        samples = lhs(count = sampleSize, dimensionality=space.dNum)
    elif method == InitialSampleMethod.HALTON:
        print('Generating the samples using Halton sequences method. This may take a while...')
        samples = halton(count = sampleSize, dimensionality=space.dNum)
    
    for dimIndex, dimension in enumerate(space.dimensions):
        samples[:,dimIndex] *= dimension.range
        samples[:,dimIndex] += dimension.bounds[0]
    # Chekcing the points for the constraints:
    if len(constraints)>0:
        results = np.array([np.apply_along_axis(cons,axis = 1, arr = samples) for cons in constraints]).T
        taking = np.apply_along_axis(all, axis=1,arr=results)
        na = sum(taking.astype(int))
        nr = sampleSize - na
        # If resample is activated then the sample is taken again to compensate for all the rejected samples.
        if resample and na < sampleSize : 
            newSampleSize = int((sampleSize**2)/na) + 1
            print(f'{nr} Samples were rejected due to violating constraints. The samples will be retaken to compensate for the rejected ones. New sample size will be {newSampleSize}.')
            return generateInitialSample(
                space = space,
                sampleSize = newSampleSize,
                method=method,
                checkForEmptiness=checkForEmptiness,
                constraints=constraints,
                resample = False)
        else: 
            samples = samples[taking,:]
    return samples
    

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class StandardClassifier(SVC,Benchmark):
    """
        A custom classifier based on Support vector classifier class that trains a standard scaler to the same data that it uses to train the SVM. The data is then standardized before fitting the fitting function is called. 
    """
    def __init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False,
                 random_state=None):
        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state)
        self.threshold = 0.5 if probability else 0

    def fit(self, X, y, sample_weight=None):
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler(feature_range=(0,1))
        normalX = self.scaler.fit_transform(X)
        self.inputDim = len(normalX[0,:])
        return super().fit(normalX, y, sample_weight=sample_weight)
    
    def predict(self, X):
        # TODO: If X is just one sample then the scaler.transform will fail. It needs to be detected. 
        normalX = self.scaler.transform(X)
        return super().predict(normalX)
    
    def decision_function(self, X, standardize= True):
        return super().decision_function(self.scaler.transform(X) if standardize else X)

    def getSupportVectors(self, standard = True):
        """
            Returns the suppor vectors of the wrapped classifier.

            Inputs: 
                - standard: Boolean. If True, the actual value of the support vectors are passed. If False, the suppor vectors are normalized along all their axes before being passed. 

            Outputs:
                - supportVectors: List of the support vectors of the trained classifier
        """
        if not standard:
            return self.support_vectors_
        else:
            return self.scaler.inverse_transform(self.support_vectors_)

    def _function(self, datum):
        return self.decision_function(datum.reshape(1,self.inputDim))

def testImport():
    print('The sampling script is imported.')