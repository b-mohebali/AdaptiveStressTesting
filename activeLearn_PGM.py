#! /usr/bin/python

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
from ActiveLearning.optimizationHelper import GeneticAlgorithmSolver
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
    DONE 1- Run an FFD sample with the 4 variables in a new location with the control object
        from simulationHelper.py script. -> DONE
    2- Implement the initial sampling using the combination of the control objects and 
        the developed active learning code. 
    3- Implement the exploitation part. Save the change measures in each step in case the 
        process is interrupted for any reason.
        
"""
"""
NOTE 1: Use the currentDir variable from repositories to point to the AdaptiveStressTesting
    folder. The automation codebase tends to change the working directory during 
    the process and it has to be switched back to use the assets 

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
sampleSaveFile = currentDir + '/assets/experiments/test_sample.txt'
repoLoc = adaptRepo3

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

# # Preparing and running the initial sample: 
formattedSample = getSamplePointsAsDict(dimNames, initialSamples)
saveSampleToTxtFile(formattedSample, sampleSaveFile)
runSample(sampleDictList=formattedSample, 
        dFolder = dataFolder,
        remoteRepo=repoLoc,
        simConfig=simConfig)



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
# # Calling the metrics function on all the samples:
getMetricsResults(dataLocation=repoLoc,
                eng = matlabEngine,
                sampleNumber = samplesList,
                metricNames = simConfig.metricNames)

# TODO: The mother sample results is not loaded into the caps servers
#### Load the mother sample for comparison:


#### Load the results into the dataset and train the initial classifier:
dataset, labels = readDataset(repoLoc, variables)


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

# Defining the optimizer object:
optimizer = GeneticAlgorithmSolver(space = designSpace,
                                    epsilon = 0.03,
                                    batchSize = batchSize,
                                    convergence_curve = False,
                                    progress_bar = True)

iterationReports = []
# Creating the report object:
outputFolder = f'{repoLoc}/outputs'
figFolder = setFigureFolder(outputFolder)
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
sInfo = SaveInformation(fileName = f'{figFolder}/initial_plot', savePDF=True, savePNG=False)
"""
TODO: Implementation of the benchmark for this visualizer. 
    The correct way is to use a pickle that contains the classifier 
    trained on the mother sample. Since the results of the evaluation 
    of the mother sample are in the local system.
"""
plotSpace(designSpace,
            figsize = figSize,
            meshRes = 100,
            classifier = clf,
            gridRes = gridRes,
            showPlot=False,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = False) 
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
    formattedFoundPoints = getSamplePointsAsDict(dimNames, newPointsFound)
    # Getting the number of next samples:
    nextSamples = getNextSampleNumber(repoLoc, createFolder=False, count = len(newPointsFound))
    # running the simulation at the points that were just found:
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
    dataset,labels = readDataset(repoLoc, variables)
    designSpace._samples, designSpace._eval_labels = dataset, labels
    prevClf = clf
    clf = svm.SVC(kernel = 'rbf', C = 1000)
    clf.fit(dataset, labels)
    newChangeMeasure = convergenceSample.getChangeMeasure(percent = True,
                        classifier = clf, 
                        updateLabels = True)
    changeMeasure.append(newChangeMeasure)
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
            legend = False,
            prev_classifier = prevClf) 
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




    
    











