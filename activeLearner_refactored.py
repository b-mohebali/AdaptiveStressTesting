from ActiveLearning.dataHandling import getFirstEmptyFolder
from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import *
import logging
import os,sys
import subprocess
from ActiveLearning.benchmarks import DistanceFromCenter, Branin, Benchmark, Hosaki
from ActiveLearning.Sampling import *
import platform
import shutil
import matplotlib.pyplot as plt 
from enum import Enum
from sklearn import svm
from ActiveLearning.optimizationHelper import GA_Exploiter, GA_Voronoi_Explorer, ResourceAllocator, allocateResources
from copy import copy, deepcopy

from ActiveLearning.visualization import *
import time 
from datetime import datetime 
import numpy as np 

def constraint1(X):
    x1 = X[0]
    x2 = X[1]
    cons = x1-x2 < 2.5
    return cons

def constraint2(X):
    x2=X[1]
    cons = x2 < 3.5
    return cons
consVector = [constraint1, constraint2]

# Loading the config files of the process:
simConfig = simulationConfig('./assets/yamlFiles/adaptiveTesting.yaml')
variableFiles = './assets/yamlFiles/varAdaptTest.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variableFiles, scalingScheme=Scale.LINEAR)

# Loading the parameters of the process:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize

# Individual budgets. Will be replaced by dynamic resource allocator:
exploitationBudget = 2
explorationBudget = batchSize - exploitationBudget

# Defining the design space based on the variables config file: 
mySpace = SampleSpace(variableList=variables)
dimNames = mySpace.getAllDimensionNames()
initialReport = IterationReport(dimNames)
# Defining the benchmark:
# myBench = DistanceFromCenter(threshold=1.5, inputDim=mySpace.dNum, center = [4] * mySpace.dNum)
# myBench = Branin(threshold=8)
myBench = Hosaki(threshold = -1)
# Generating the initial sample. This step is pure exploration MC sampling:

# Starting time:
initialReport.startTime = datetime.now()
initialReport.setStart()
initialSamples = generateInitialSample(space = mySpace, 
                                        sampleSize = initialSampleSize,
                                        method = InitialSampleMethod.CVT,
                                        checkForEmptiness=False,
                                        constraints=consVector)
initialLabels = myBench.getLabelVec(initialSamples)

# Initial iteration of the classifier trained on the initial samples and their labels:
clf = StandardClassifier(kernel = 'rbf', C = 1000)
clf.fit(initialSamples, initialLabels)
# Adding the samples and their labels to the space: 
mySpace.addSamples(initialSamples, initialLabels)

# # Setting up the location of the output of the process:
outputFolder = f'{simConfig.outputFolder}/{getFirstEmptyFolder(simConfig.outputFolder)}'
print('Output folder for figures: ', outputFolder)
iterationReportFile = f'{outputFolder}/iterationReport.yaml'
figFolder = setFigureFolder(outputFolder)
sInfo = SaveInformation(fileName = f'{figFolder}/InitialPlot', savePDF=True, savePNG=True)

# Visualization of the first iteration of the space with the initial sample:
meshRes = 300
figSize = (10,8)

plotSpace(mySpace, 
        classifier=clf, 
        figsize = figSize, 
        meshRes=meshRes,
        legend = True, 
        showPlot=False,
        saveInfo = sInfo, 
        benchmark = myBench,
        constraints = consVector)
plt.close()
# Finishing time
initialReport.stopTime = datetime.now()
initialReport.setStop()
# Defining the exploiter: 
exploiter = GA_Exploiter(space = mySpace, 
                    epsilon = 0.05,
                    batchSize = simConfig.batchSize,
                    convergence_curve=False,
                    progress_bar=False,
                    clf = clf, 
                    constraints = consVector)

# Defining the explorer object:
explorer = GA_Voronoi_Explorer(space = mySpace, 
                    batchSize = simConfig.batchSize,
                    convergence_curve=False, 
                    progress_bar=False, 
                    constraints = consVector)

# Defining the convergence sample that implementes the change measure as well as
#   the performance metrics for the process. 
convergenceSample = ConvergenceSample(mySpace, constraints=consVector)
changeMeasure = [convergenceSample.getChangeMeasure(percent = True, 
                                    classifier = clf, 
                                    updateLabels=True)]

# Getting the initial accuracy:
acc = [convergenceSample.getPerformanceMetrics(benchmark = myBench,
                                            classifier=clf,
                                            percentage = True, 
                                            metricType=PerformanceMeasure.ACCURACY)]
print('Initial accuracy: ', acc[0])   
sampleNumbers = [mySpace.sampleNum]
# Calculating the remaining budget:
currentBudget = budget - initialSampleSize

# Setting up the iteration reports file:
iterationReports = []

# Getting the iteration report after the initial sample:
iterationNum = 0
initialReport.budgetRemaining = currentBudget
initialReport.setChangeMeasure(changeMeasure[0])
initialReport.batchSize = initialSampleSize
initialReport.iterationNumber = iterationNum
initialReport.setMetricResults(initialLabels)
initialReport.setSamples(initialSamples)

iterationReports.append(initialReport)
saveIterationReport(iterationReports, iterationReportFile)

prevClf = clf
convergenceSample = ConvergenceSample(space = mySpace, 
                                constraints = consVector,
                                size = 10000)
resourceAllocator = ResourceAllocator(convSample = convergenceSample,
                        epsilon = 0.2, 
                        simConfig = simConfig,
                        outputLocation = outputFolder,
                        l=1.2)


while currentBudget > 0:
    print('------------------------------------------------------------------------------')
    # Setting up the iteration report timing members:
    iterationNum += 1
    iterReport = IterationReport(dimNames, batchSize=batchSize)
    iterReport.setStart()
    print('Current budget: ', currentBudget, ' samples')
    # Finding new points using the exploiter object:
    # NOTE: The classifier has to be passed everytime to the exploiter for update.
    # exploiterPoints = exploiter.findNextPoints(min(currentBudget, batchSize))
    exploiterPoints = exploiter.findNextPoints(pointNum=exploitationBudget)
    explorerPoints = explorer.findNextPoints(pointNum=explorationBudget)
    # Updating the remaining budget:    
    currentBudget -= min(currentBudget, batchSize) 
    # Visualization and saving the results:
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_NotLabeled'
    plotSpace(mySpace, 
        figsize = figSize,
        legend = True, 
        showPlot=False, 
        classifier = clf,
        saveInfo=sInfo,
        meshRes = meshRes,
        newPoints=exploiterPoints,
        explorePoints=explorerPoints,
        benchmark = myBench,
        prev_classifier= prevClf,
        constraints = consVector)
    plt.close()
    # Evaluating the newly found samples: 
    newLabels = myBench.getLabelVec(exploiterPoints)
    exploreLabels = myBench.getLabelVec(explorerPoints)

    # Resource Allocation for the next iteration: 
    # This function saves the resource allocation report itself. 
    calcExploitBudget, calcExploreBudget = resourceAllocator.allocateResources(
        mainSamples = mySpace.samples,
        mainLabels = mySpace.eval_labels,
        exploitSamples = exploiterPoints,
        exploitLabels = newLabels,
        exploreSamples= explorerPoints,
        exploreLabels=exploreLabels,
        saveReport = True
    )






    # Adding the newly evaluated samples to the dataset:
    mySpace.addSamples(exploiterPoints, newLabels)
    mySpace.addSamples(explorerPoints, exploreLabels)
    # Updating the previous classifier before training the new one:
    prevClf = deepcopy(clf)
    # Training the next iteration of the classifier:
    clf = StandardClassifier(kernel = 'rbf', C=1000)
    clf.fit(mySpace.samples, mySpace.eval_labels)
    # Updating the classifier that the exploiter uses. The explorer samples independent of the classifier hence it does not need the classifier or update on it. 
    exploiter.clf = clf
    # Calculation of the new measure of change and accuracy after training:
    newChangeMeasure = convergenceSample.getChangeMeasure(percent = True, 
                                            classifier = clf, 
                                            updateLabels=True)
    changeMeasure.append(newChangeMeasure)
    newAccuracy = convergenceSample.getPerformanceMetrics(benchmark = myBench, 
                                            percentage=True, 
                                            classifier=clf,
                                            metricType=PerformanceMeasure.ACCURACY)
    acc.append(newAccuracy)
    print('Hypothesis change estimate: ', changeMeasure[-1:], ' %')
    print('Current accuracy estimate: ', acc[-1:], ' %')
    sampleNumbers.append(mySpace.sampleNum)
    # Visualization of the results after the new samples are evaluated:
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_Labeled'
    plotSpace(space = mySpace,
        figsize = figSize,
        legend = True,
        classifier = clf, 
        benchmark = myBench,
        meshRes = meshRes,
        newPoints=None,
        saveInfo=sInfo,
        showPlot=False,
        prev_classifier=prevClf,
        constraints = consVector)
    plt.close()
    # Adding the iteration information to the report for saving.
    iterReport.setStop()
    iterReport.budgetRemaining = currentBudget
    iterReport.iterationNumber = iterationNum
    iterReport.setMetricResults(newLabels)
    iterReport.setExploitatives(exploiterPoints)
    iterReport.setExplorers(explorerPoints)
    iterReport.setChangeMeasure(newChangeMeasure)
    iterationReports.append(iterReport)
    saveIterationReport(iterationReports, iterationReportFile)



# Final visualization of the results: 
plotSpace(space = mySpace, figsize=(10,8), legend = True,
                        saveInfo=sInfo, showPlot=True, classifier = clf, 
                        benchmark = myBench, constraints = consVector)




