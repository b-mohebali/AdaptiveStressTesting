from yamlParseObjects.yamlObjects import variableConfig
from typing import List
from samply.hypercube import cvt, lhs
from enum import Enum
from .benchmarks import Benchmark
from sklearn import svm
from sklearn.metrics import accuracy_score,precision_score, recall_score
import numpy as np 

class InsufficientInformation(Exception):
    pass

class SampleNotEmpty(Exception):
    pass

# This class represents a single dimension in the design space:
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

'''
    This class determines the type of the performance metric for the current iteration
    of hypothesis.
'''
class PerformanceMeasure(Enum):
    ACCURACY = 0
    PRECISION = 1
    RECALL = 2

class Space():
    def __init__(self, variableList: List[variableConfig], 
                initialSampleCount = 20, 
                benchmark: Benchmark = None):
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

    """
        This function returns a list of dictionaries containing the values of the 
        design variables for each experiment (one dict per experiment)
    """
    def getSamplePointsAsDict(self):
        output = []
        for sample in self.samples:
            sampleDict = {}
            for idx, dim in enumerate(self.dimensions):
                sampleDict[dim.name] = sample[idx]
            output.append(sampleDict)
        return output

    """
        This function gets a samples dictionary and loads it into the samples
        list of the space. This is for the times when we want to load an already
        sampled list into the space without sampling from scratch. 
    """
    def loadSamples(self,samplesDictList):
        for sampleDict in samplesDictList:
            sample = []
            for dim in self.dimensions:
                sample.append(sampleDict[dim.name])
            self.samples.append(sample)
    


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


def generateInitialSample(space: Space, 
                        sampleSize: int, 
                        method: InitialSampleMethod = InitialSampleMethod.CVT,
                        checkForEmptiness = False):
    ''' This function samples from the entire space. 
        Can be used for the convergence samples as well without the check for 
        the emptiness of the space.

        space:  Contains information about the number of dimensions and their range 
                of variations.

        method: CVT (Centroid Voronoi Tesselation)
                LHS (Latin Hypercube)

        CheckForEmptiness: (True/False) 
                Raises error if the sample list of the space is not empty, meaning 
                that the initial sample is most likely taken. 
    '''
    if checkForEmptiness and len(space.samples) > 0:
        raise SampleNotEmpty('The space already contains samples.')
    if method == InitialSampleMethod.CVT:
        samples = cvt(count = sampleSize, dimensionality= space.dNum)
    elif method == InitialSampleMethod.LHS:
        samples = lhs(count = sampleSize, dimensionality=space.dNum)
    for dimIndex, dimension in enumerate(space.dimensions):
        samples[:,dimIndex] *= dimension.range
        samples[:,dimIndex] += dimension.bounds[0]
    return samples




# TODO: This class holds the sample used for convergence check at each iteration:
class ConvergenceSample():
    '''
    This class contains the convergence sample used for checking the amount of 
    change in the hypothesis. It also provides some functionality for measuring 
    the mentioned change.

    NOTE:   The approximation of the accuracy of the current iteration of the 
            hypothesis is only possible if the actual function is a low-cost 
            benchmark and not an expensive computational model.  
    '''
    def __init__(self, 
                space: Space):
        self.size = 100 * 5**space.dNum
        self.samples = generateInitialSample(space, 
                                            sampleSize=self.size, 
                                            method=InitialSampleMethod.LHS)
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
    
    # TODO
    """
    This function takes a benckmark object and a classifier and compares the 
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
    # TODO: Get a threshold for the classification. Right now the default threshold 
    #       is 0.5, which may not be the best for the application.
    def getPerformanceMetrics(self,
                            benchmark: Benchmark,
                            classifier = None,
                            percentage = True,
                            metricType: PerformanceMeasure = PerformanceMeasure.ACCURACY):
        
        # If the classifier is not entered, the last set of labels are used for the 
        #   evaluation. 
        yPred = self.currentLabels if classifier is None else classifier.predict(self.samples)
        yTrue = benchmark.getLabelVec(self.samples)
        if metricType == PerformanceMeasure.ACCURACY:
            metric = accuracy_score(yTrue, yPred)
        elif metricType == PerformanceMeasure.PRECISION:
            metric = precision_score(yTrue, yPred)
        elif metricType == PerformanceMeasure.RECALL:
            metric = recall_score(yTrue, yPred)
        return metric * 100 if percentage else metric
    
"""
    This function returns a list of dictionaries containing the values of the 
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
def getSamplePointsAsDict(dimNames, sampleList):
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

"""
    New implementation of the space class with limited functionality. 
    NOTE: The functionality of the Space class was getting too much and so
        it was broken down into several classes.
"""
class Space2():
    def __init__(self, 
                variableList: List[variableConfig]):
        self.dimensions = []
        for varConfig in variableList:
            self.dimensions.append(Dimension(varConfig=varConfig))
            self._dNum = len(self.dimensions)
            self.convPointsNum = 100 * 5 ** self.dNum 
            self._samples = []
            self._eval_labels = []
    
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
    
    def getAllDimensionBounds(self):
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
    def nearestPointDistance(self, X, samplesList = None):
        if samplesList is None:
            return np.min(np.linalg.norm(self.samples - X, axis=1))
        return np.min(np.linalg.norm(samplesList - X, axis=1))


"""
    NOTE: The initial classifier used here is SVM. The necessity for any other type of 
    classifier was not felt at this point.
    - Also it may be shown that the SVM class does not require a simple wrapper like 
        this. 
"""
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



