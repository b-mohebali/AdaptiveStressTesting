#! /usr/bin/python

from multiprocessing import process
from yamlParseObjects.yamlObjects import *
from yamlParseObjects.variablesUtil import *
import logging 
import os, sys
import subprocess
from profileExample.profileBuilder import * 
import platform
from eventManager.eventsLogger import * 
import csv
import platform
import shutil
import numpy as np
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from enum import Enum
import time
from ActiveLearning.Sampling import *
from ActiveLearning.dataHandling import *
from ActiveLearning.visualization import * 
from ActiveLearning.optimizationHelper import GeneticAlgorithmExploiter
from ActiveLearning.benchmarks import TrainedSvmClassifier
from sklearn import svm
from copy import copy
simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
print(simConfig.name)
for p in simConfig.codeBase:
    sys.path.insert(0,p)
    print(p + ' has been added to the path.')


from ActiveLearning.simulationHelper import * 
from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup
from rscad import rtds
from repositories import *
import simulation

from metricsRunTest import * 

"""
    Setting up the matlab engine. 
    This cannot be done in another file as a global object. So it will be 
    instantiated here in the driver script and will be passed to the function
    that runs the matlab metrics.
"""
matlabEngine = setUpMatlab(simConfig=simConfig)
"""
    Defining the PGM model object as a global variable so that it does not
        have to be instantiated every time.
"""
# Model under test:
mut = PGM_control('','./', configFile=simConfig)

"""
Steps of checks for correctness: 
    DONE 1- Run an FFD sample with the 4 variables in a new location with the control object from simulationHelper.py script. -> DONE
    DONE 2- Implement the initial sampling using the combination of the control objects and the developed active learning code. 
    DONE 3- Implement the exploitation part. Save the change measures in each step in case the process is interrupted for any reason.
    DONE 4- Implement the loading of the benchmark classifier trained on the Monte-Carlo data. 
    DONE 5- Run a sample with Visualiation and the benchmark and compare the results.
    6- Calculate the metrics of the classifier such as precision, recall, accuracy, measure of change vs the number of iterations.
    7- Empirically show that the active learner can reach comparable performance with the Monte-Carlo sampling method using a fraction of the process time. 
    8- Improve on the process time using exploration, prallelization, batch sampling.

NOTE 1: Use the currentDir variable from repositories to point to the AdaptiveStressTesting folder. The automation codebase tends to change the working directory during the process and it has to be switched back to use the assets.
"""

print('This is the AC PGM sampling test file. ')
variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'


# Extracting the hyperparameters of the analysis:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

# Setting the main files and locations:
descriptionFile = currentDir + '/assets/yamlFiles/varDescription.yaml'
sampleSaveFile = currentDir + '/assets/experiments/adaptive_sample.txt'
repoLoc = adaptRepo3

# Defining the location of the output files:
outputFolder = f'{repoLoc}/outputs'
figFolder = setFigureFolder(outputFolder)

# Defining the design space and the handler for the name of the dimensions. 
designSpace = SampleSpace(variableList=variables)
dimNames = designSpace.getAllDimensionNames()
initialReport = IterationReport(dimNames)
initialReport.setStart()
# # Taking the initial sample based on the parameters of the process. 
initialSamples = generateInitialSample(space = designSpace,
                                        sampleSize=initialSampleSize,
                                        method = InitialSampleMethod.CVT,
                                        checkForEmptiness=False)

### Preparing and running the initial sample: 
# formattedSample = getSamplePointsAsDict(dimNames, initialSamples)
# saveSampleToTxtFile(formattedSample, sampleSaveFile)
# runSample(sampleDictList=formattedSample, 
#         dFolder = dataFolder,
#         remoteRepo=repoLoc,
#         simConfig=simConfig)



## Loading sample from a pregenerated file in case of interruption:
# print(currentDir)
# formattedSample = loadSampleFromTxtFile(sampleSaveFile)

# runSample(formattedSample, dFolder = dataFolder, 
#                 remoteRepo=repoLoc,
#                 simConfig=simConfig,
#                 sampleGroup=[80])


#### Running the metrics on the first sample: 

setUpMatlab(simConfig=simConfig)
# # Forming the sample list which includes all the initial samples:
samplesList = list(range(1, initialSampleSize+1))
### Calling the metrics function on all the samples:
# Using the parallelized metrics evaluation part. 
runMetricsBatch(dataLocation=repoLoc,
                sampleGroup=samplesList,
                configFile=simConfig,
                figureFolder=figFolder,
                processNumber=4)


#### Load the mother sample for comparison:
"""
This part loads a pickled classifier that is trained on the Monte-Carlo sample taken from the 
    model. The purpose for this classifier is to act as a benchmark for the active classifier
    that we are trying to make. 
"""
motherClfPickle = picklesLoc + 'mother_clf.pickle'
classifierBench = None
if os.path.exists(motherClfPickle) and os.path.isfile(motherClfPickle):
    with open(motherClfPickle,'rb') as pickleIn:
        motherClf = pickle.load(pickleIn)
    threshold = 0.5 if motherClf.probability else 0
    classifierBench = TrainedSvmClassifier(motherClf, len(variables), threshold)


#### Load the results into the dataset and train the initial classifier:
dataset, labels = readDataset(repoLoc, dimNames=dimNames)


# updating the space:
designSpace._samples, designSpace._eval_labels = dataset, labels
# Stopping the time measurement for the iteration report:
initialReport.setStop()

#### Iterations of exploitative sampling:
clf = svm.SVC(kernel = 'rbf', C = 1000)
clf.fit(dataset, labels)

# Updating the budget:
currentBudget = budget - initialSampleSize


convergenceSample = ConvergenceSample(designSpace)
changeMeasure = [convergenceSample.getChangeMeasure(percent = True,
                            classifier = clf,
                            updateLabels=True)]
samplesNumber = [initialSampleSize]

# Defining the optimizer object:
optimizer = GeneticAlgorithmExploiter(space = designSpace,
                                    epsilon = 0.03,
                                    batchSize = batchSize,
                                    convergence_curve = False,
                                    progress_bar = True)

iterationReports = []
# Creating the report object:

iterationReportsFile = f'{outputFolder}/iterationReport.yaml'
iterationNum = 0
# Saving the initial iteration report.
initialReport.iterationNumber = iterationNum 
initialReport.budgetRemaining = currentBudget
initialReport.setChangeMeasure(changeMeasure[0])
initialReport.batchSize = initialSampleSize
initialReport.setMetricResults(labels)
initialReport.setSamples(dataset)

iterationReports.append(initialReport)
saveIterationReport(iterationReports, iterationReportsFile)


## -----------------------------------
# # Setting up the parameters for visualization: 
insigDims = [0,2]
figSize = (12,10)
gridRes = (4,4)
meshRes = 100
sInfo = SaveInformation(fileName = f'{figFolder}/initial_plot', 
                        savePDF=True, 
                        savePNG=True)
"""
TODO: Implementation of the benchmark for this visualizer. 
    The correct way is to use a pickle that contains the classifier 
    trained on the mother sample. Since the results of the evaluation 
    of the mother sample are in the local system.

    NOTE: Done but not tested yet.
"""
plotSpace(designSpace,
            figsize = figSize,
            meshRes = 100,
            classifier = clf,
            gridRes = gridRes,
            showPlot=False,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = True,
            benchmark = classifierBench) 
plt.close()

## Adaptive Sampling loop:
while currentBudget > 0:
    # Setting up the iteration number:
    iterationNum += 1
    iterReport = IterationReport(dimNames, batchSize = batchSize)
    iterReport.setStart()
    print('Current budget: ', currentBudget, ' samples.')
    # Finding new points using the optimizer
    newPointsFound = optimizer.findNextPoints(clf,
                                        min(currentBudget, batchSize))
    # Updating the remaining budget:
    currentBudget -= len(newPointsFound)
    # formatting the samples for simulation:
    # NOTE: this is due to the old setting used for the DOE code in the past.
    formattedFoundPoints = getSamplePointsAsDict(dimNames, newPointsFound)
    # Getting the number of next samples:
    nextSamples = getNextSampleNumber(repoLoc, 
        createFolder=False, 
        count = len(newPointsFound))
    # running the simulation at the points that were just found:
    """
    TODO: Run all the matlab processes simultaneously. The simulation is done 
            on a point by point basis for now. But the bottleneck of the 
            timing is in the MATLAB metrics calculations. 
    """
    for idx, sample in enumerate(formattedFoundPoints):
        runSinglePoint(sampleDict = sample,
                        dFolder = dataFolder,
                        remoteRepo = repoLoc,
                        simConfig= simConfig,
                        sampleNumber = nextSamples[idx],
                        modelUnderTest=mut)
    # Evaluating the newly simulated samples using MATLAB engine:
        getMetricsResults(dataLocation = repoLoc, 
                        eng = matlabEngine,
                        sampleNumber = nextSamples[idx],
                        metricNames = simConfig.metricNames,
                        figFolderLoc=figFolder)
    # Updating the classifier and checking the change measure:
    dataset,labels = readDataset(repoLoc, dimNames= dimNames)
    designSpace._samples, designSpace._eval_labels = dataset, labels
    prevClf = clf
    clf = svm.SVC(kernel = 'rbf', C = 1000)
    clf.fit(dataset, labels)
    newChangeMeasure = convergenceSample.getChangeMeasure(percent = True,
                        classifier = clf, 
                        updateLabels = True)
    
    # Saving the change measure vector vs the number of samples in each iteration. 
    changeMeasure.append(newChangeMeasure)
    samplesNumber.append(len(labels))
    print('Hypothesis change estimate: ', changeMeasure[-1:], ' %')

    # Visualization of the current state of the space and the classifier
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_Labeled'
    plotSpace(designSpace,
            figsize = figSize,
            meshRes = 100,
            classifier = clf,
            gridRes = gridRes,
            showPlot=False,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = True,
            prev_classifier = prevClf,
            benchmark = classifierBench) 
    plt.close() # Just in case. 
    # Saving the iteration report:
    # TODO: Reduce the lines of code that does this job:
    iterReport.setStop()
    iterReport.budgetRemaining = currentBudget
    iterReport.iterationNumber = iterationNum
    iterReport.setMetricResults(labels[-len(newPointsFound):])
    iterReport.setSamples(newPointsFound)
    iterReport.setChangeMeasure(newChangeMeasure)
    iterationReports.append(iterReport)
    saveIterationReport(iterationReports,iterationReportsFile)

# Plotting the change measure throughout the process.
plt.figure(figsize = (8,5))
plt.plot(samplesNumber, changeMeasure)
plt.grid(True)
sInfo = SaveInformation(fileName = f'{figFolder}/change_measure', savePDF = True, savePNG = True)
saveFigures(sInfo)
plt.close()



    
    











