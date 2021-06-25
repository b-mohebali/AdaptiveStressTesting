from os.path import isfile
import numpy as np 
import os
import yaml
import string
from typing import List
from yamlParseObjects.yamlObjects import *
from repositories import * 
import pickle
from enum import Enum
import string

class BadExtention(Exception):
    pass

resultFileName = 'finalReport.yaml'


def readDataset(repoLoc, dimNames, includeTimes = False, sampleRange = None):
    """
        Reading a dataset from a group of evaluated samples.
        
        Input: 
            - Location of the samples
            - List of the variables configurations for the order of their names
            - IncludeTimes: Boolean determining whether the process time of each is needed and should be included in the output or not. 
            - sampleRange: If not None, only the samples in the entered range
                will be read for dataset loading. If None, all samples in the repository will be read for dataset loading. 

        Output:
            - dataset: Dataset in the form of data points as a list of lists
            - labels: Label of each sample in the same order as the samples in the dataset
            - times: (Optional) The time each sample took to be evaluated.


        Note: Assumes that the samples are in folders with numerical names. Ignores
        all the other types of names (alphabetical or symbolic characters)

    """
    sampleFolders = [name for name in os.listdir(repoLoc) if name.isdigit()]
    # This makes sure that only the samples that actually exist in the 
    #   repo will be loaded into the dataset.
    if sampleRange is not None:
        sampleFolders  = [_ for _ in sampleFolders if _ in sampleRange]
    labels = []
    dataset = []
    elapsed_times = []
    for sampleFolder in sampleFolders: 
        d,l,t = readSingleSample(repoLoc, dimNames, sampleFolder)
        labels.append(l)
        dataset.append(d)
        elapsed_times.append(t)
    if includeTimes:
        return dataset, labels, elapsed_times
    return dataset, labels

def readSingleSample(repoLoc,dimNames, sampleNumber):
    """
        This function read the results of a single sample from the repo.
        NOTE: The final report coming from the factor screening may have more dimensions than needed for the adaptive sampling. The extra dimensions will be ignored by this function as the list of the desirable dimensions is passed to it. 
        Inputs:
            - Location of the samples (repoLoc)
            - List of the dimension names (dimNames)
            - Sample Number (sampleNumber)

        Output:
            - list of the values for the dimensions of the sample (varList)
            - Label of the sample as it was evaluated by the metrics (label)
            - The time the sample took to be evaluated.
    """
    yamlFile = repoLoc + f'/{sampleNumber}/{resultFileName}'
    reportObj = FinalReport(yamlFile)
    varList = []
    for dimName in dimNames:
        varList.append(reportObj.variables[dimName])
    return varList, reportObj.label, reportObj.elapsed_time


def getNextSampleNumber(repoLoc, createFolder:bool = False, count = 1):
    """
        Gets the location of the repository and determines what is the number of the next sample. 

        Returns a list of sample numbers if count > 1
    """
    sampleFolders = [int(name) for name in os.listdir(repoLoc) if name.isdigit()]
    lastSampleNumber = max(sampleFolders)
    nextSampleNumber = lastSampleNumber + 1
    nextSamples = [_ for _ in range(nextSampleNumber, nextSampleNumber+count)]
    if createFolder:
        [os.mkdir(repoLoc + f'/{_}') for _ in nextSamples]
    return nextSamples

def getNextEvalSample(repoLoc):
    """
        Gets the location of the repository and determines the first sample that needs MATLAB metrics evaluation. This means all the samples before this sample must be evaluated using the MATLAB metrics implementation. 

        Returns the number of the next sample that needs MATLAB metrics evaluation.
        If all the samples in the repo are evaluated (meaning the report files are present in all the sample folders) returns None.
    """
    sampleFolders = [int(name) for name in os.listdir(repoLoc) if name.isdigit()]
    sampleFolders.sort()
    for sf in sampleFolders:
        resultFileLoc = f'{repoLoc}/{sf}/{resultFileName}'
        if not os.path.isfile(resultFileLoc):
            return sf
    return None

def saveClassifierAsPickle(cls, pickleName: str):
    """
        Takes a classifier and a name for the saved file. Saves the classifier object as a pickle in the directory allocated to the trained classifiers.

        Inputs:
            - cls: Classifier that is going to be saved.
            - pickleName: The name of the file being used to save the classifier
    """
    # Checking if the file has the right extension:
    if not pickleName.endswith('.pickle'):
        raise BadExtention("The file name has to end with .pickle")
    # Building the absolute path of the saved file:
    fileName = picklesLoc + pickleName
    # Saving the pickle:
    with open(fileName, 'wb'):
        pickle.dump(cls, fileName)


class ProcessLocator:
    """
        This class contains the answers to some questions regarding where we are in the process of adaptive sampling. Its main application is when the proces is interrupted midway and needs to be restarted from the middle without the need to discard the already gathered samples. These are the questions and potential answers:

        1- Is the initial sample taking done?
            - No -> What is the # of the next sample to be taken?
            - Yes -> Are all the metrics evaluated?
                - Yes -> Go to Question2.
                - No -> What is the # of the next sample to be evaluated by the MATLAB code?
        2- Is the adaptive sampling phase done completed?
            - Yes -> 3- Are all the samples evaluated by MATLAB code? 
                - Yes -> train the classifier and report the final results.
                - No -> What is the # of the next sample to be evaluated by the MATLAB code? 
            - No -> What is the # of the next sample to be taken? 
    """
    def __init__(self):
        self.initialSampleDone = False          # Are initial samples taken?
        self.initialSamplesEvaluated = False    # Are initial samples evaluated?
        self.adaptiveSamplingDone = False       # Are adaptive samples taken?
        self.adaptiveSamplesEvaluated = False   # Are adaptive samples evaluated?
        self.nextSampleNum = None               # Next sample to be simulated.
        self.nextEvalSampleNum = None           # Next sample to be evaluated.

    def locateProcess(self, repoLoc, simConfig: simulationConfig):
        """ 
            The logic of finding the answers to those questions is here:
        """
        initSampleSize = simConfig.initialSampleSize
        budget = simConfig.sampleBudget
        nextEval = getNextEvalSample(repoLoc=repoLoc)
        currentSample = getNextSampleNumber(repoLoc)[0]
        # Checking to see if the initial sample is done:
        if currentSample <= initSampleSize:
            self.nextSampleNum = currentSample
            self.nextEvalSampleNum = nextEval
            return self
        else:
            self.initialSampleDone = True 
            if nextEval <= initSampleSize:
                # If we are here it means the adaptive phase is not started yet. But the initial samples are all taken but not evaluated. 
                self.nextEvalSampleNum = nextEval
                return self 
            self.initialSamplesEvaluated = True 
        # Check to see if the adaptive sampling phase is done:
        if currentSample <= budget: 
            # If we are here it means that the adaptive samples are not done but the initial samples are fully taken and evaluted.
            self.adaptiveSamplesEvaluated = (nextEval > budget)
            self.nextEvalSampleNum = nextEval
            self.nextSampleNum= currentSample
            return self
        else: 
            # If we are here the adaptive samples are all taken according to the assigned budget. 
            self.adaptiveSamplingDone = True 
            self.adaptiveSamplesEvaluated = nextEval > budget
            self.nextEvalSampleNum = nextEval if not self.adaptiveSamplesEvaluated else None
        return self


def loadChangeMeasure(reportFile: str):
    """
        This function takes the location of an iteration report yaml file and returns a list of iteration numbers as well as the change measure in each iteration. 

        Inputs:
            - iterationReport: The absolute location of the yaml file containing the report from an adaptive sampling process. 
        
        Outputs: 
            - iterationNumbers: The list of the iteration numbers. Used as the x-axis data for plotting the change measure.
            - changeMeasures: The list of the change measure values throughout the iterations of the process. Used as the y-axis data for plotting.
    """
    with open(reportFile, 'rt') as rFile:
        yamlString = rFile.read()
    reports = yaml.load_all(yamlString, Loader = yaml.Loader)
    iterationNumbers = []
    changeMeasures = []
    for report in reports:
        iterationNumbers.append(report.iterationNumber)
        changeMeasures.append(report.changeMeasure)
    return iterationNumbers, changeMeasures 

def loadAccuracy(reportFile: str):
    """
        This function takes the location of an iteration report yaml file and returns a list of iteration numbers as well as the accuracy in each iteration.

        NOTE: The accuracy may not have been recorded through the process for reasons such as lack of benchmark. In that case nothing will be returned. In case some of the reports have recorded accuracy, only the accuracy and iteration number of the reports with recorded accuracy will be returned.

        Inputs:
            - iterationReport: The absolute location of the yaml file containing the report from an adaptive sampling process. 
        
        Outputs: 
            - iterationNumbers: The list of the iteration numbers. Used as the x-axis data for plotting accuracy.
            - accuracies: The list of the accuracies values throughout the iterations of the process if they exist. Used as the y-axis data for plotting.
    """
    with open(reportFile, 'rt') as rFile:
        yamlString = rFile.read()
    reports = yaml.load_all(yamlString, Loader = yaml.Loader)
    iterationNumbers = []
    accuracies = []
    for report in reports:
        if hasattr(report, 'accuracy'):
            iterationNumbers.append(report.iterationNumber)
            accuracies.append(report.changeMeasure)
    return iterationNumbers, accuracies


# ------------------------------------------------------------------------
# This part is related to the factor screening analysis and data handling:
class OutputType(Enum):
    DICT = 0
    NPARRAY = 1
    LIST = 2

def loadMetricValues(dataLoc, metricNames, outType : OutputType = OutputType.DICT):
    '''
        This function takes the location of a data repo and loads the metric values for all the evaluated samples into a dictionary.
        Inputs: 
            - dataLoc: Location of the repository where the sample folders are.
            - metricNames: List of strings containing the name by which the values are saved in the report yaml file. 
            TODO:
            - outType: The desired type for the output. 
    '''
    samples = [int(name) for name in os.listdir(dataLoc) if name.isdigit()]
    samples.sort()
    metricVals = {metName:[] for metName in metricNames}

    for sample in samples:
        reportPath = f'{dataLoc}/{sample}/{resultFileName}'
        with open(reportPath, 'r') as fp:
            yamlString = fp.read()
        yamlObj = yaml.load(yamlString, Loader = yaml.SafeLoader)
        for metName in metricNames:
            metricVals[metName].append(yamlObj[metName])
    return metricVals

def loadVariableValues(dataLoc, varNames):
    '''
        This function takes the location of the data repository and the name of the variables in the analysis and loads the values of the variables for all the samples. The values are then used for reconstructing the design matrix.
        Inputs:
            - dataLoc: Location of the repository where the sample folders are.
            - varNames: List of strings containing the name by which the variable values are saved within the report yaml file.
        
        Output:
            - Dictionary with variable names as the keys and the list of their values as the values. The order of the values in each list is the same as the order of the samples.
    '''
    samples = [int(name) for name in os.listdir(dataLoc) if name.isdigit()]
    samples.sort()
    varValues = {varName: [] for varName in varNames}

    for sample in samples:
        reportPath = f'{dataLoc}/{sample}/{resultFileName}'
        with open(reportPath, 'r') as fp:
            yamlString = fp.read()
        yamlObj = yaml.load(yamlString, Loader = yaml.SafeLoader)
        variables = yamlObj['variables']
        for varName in varNames: 
            varValues[varName].append(variables[varName])
    return varValues

def reconstructDesignMatrix(variableValues):
    '''
        This function takes the dict of lists (coming from the loadVariableValues function) and normalizes it to make the design matrix consisting only of 1 and -1 elements. 
        Inputs: 
            - variableVAlues: Dict(string, List[float]) containing the values for the variables used in the analysis.
        
        Output:
            - Design matrix consisting of 1 and -1 elements where 1 is related to the upper limit of the variable and -1 to its lower limit. 
    '''
    # Taking the name of the varriables:
    varNames = list(variableValues.keys())
    # Taking the number of the samples in the repo:
    N_s = len(variableValues[varNames[0]])
    # Number of variables:
    N_v = len(varNames)
    # Initializing the design matrix:
    H = np.zeros(shape = (N_s, N_v + 1))
    H[:,0] = np.ones(shape = (N_s,))
    # Filling out the matrix:
    for idx, var in enumerate(varNames):
        varVals = np.array(variableValues[var])
        varMax = max(varVals)
        varMin = min(varVals)
        print(varVals, var, varMin,varMax)
        col = np.zeros(shape = (N_s,), dtype=float)
        col[varVals==varMax] = 1
        col[varVals==varMin] = -1
        print(col)
        H[:,idx+1] = col
    return H
