from ActiveLearning.dataHandling import MetricsSaver, getFirstEmptyFolder
from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import *
import logging
from shutil import copyfile
import subprocess
from ActiveLearning.benchmarks import ArbitraryDimension, BumpyFunc, CorridorBenchmark, DistanceFromCenter, Branin, Benchmark, FourD, Hosaki, SineFunc, ThreeD1
from ActiveLearning.Sampling import *
import platform
import shutil
import matplotlib.pyplot as plt 
from enum import Enum
from sklearn import svm
from ActiveLearning.optimizationHelper import GA_Exploiter, GA_Voronoi_Explorer, ResourceAllocator
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
consVector = [] 


# Loading the config files of the process:
simConfigFile = './assets/yamlFiles/adaptiveTesting.yaml'
simConfig = simulationConfig(simConfigFile)
variableFiles = './assets/yamlFiles/varAdaptTest.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variableFiles, scalingScheme=Scale.LINEAR)

# Loading the parameters of the process:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize

# Individual budgets. Will be replaced by dynamic resource allocator:
# exploitationBudget = min(3,batchSize)
# explorationBudget = batchSize - exploitationBudget
exploitationBudget = 5 
explorationBudget = 0
print('Exploitation budget: ', exploitationBudget)
print('Exploration budget: ', explorationBudget)

# Defining the design space based on the variables config file: 
mySpace = SampleSpace(variableList=variables)
dimNames = mySpace.getAllDimensionNames()
initialReport = IterationReport(dimNames)
# Defining the benchmark:
# myBench = DistanceFromCenter(threshold=1.5, inputDim=mySpace.dNum, center = [4] * mySpace.dNum)
# myBench = Branin(threshold=8)
# myBench = Hosaki(threshold = -1)
# myBench = CorridorBenchmark(threshold = 0)
# myBench = BumpyFunc(threshold = 0)
# myBench = ThreeD1(threshold = 0)
# myBench = FourD(threshold = 0)
# myBench = ArbitraryDimension(inputDim = 5, threshold = 0)
myBench = SineFunc(threshold = 0)
# Generating the initial sample. This step is pure exploration MC sampling:

# Starting time:
initialReport.startTime = datetime.now()
initialReport.setStart()
initialSamples = generateInitialSample(space = mySpace, 
                                        sampleSize = initialSampleSize,
                                        method = InitialSampleMethod.CVT,
                                        checkForEmptiness=False,
                                        constraints=consVector,
                                        resample = True)
initialLabels = myBench.getLabelVec(initialSamples)

# Initial iteration of the classifier trained on the initial samples and their labels:
clf = StandardClassifier(kernel = 'rbf', C = 1000)
clf.fit(initialSamples, initialLabels)
# Adding the samples and their labels to the space: 
mySpace.addSamples(initialSamples, initialLabels)

# # Setting up the location of the output of the process:
outputFolder = f'{simConfig.outputFolder}/{getFirstEmptyFolder(simConfig.outputFolder)}'
print('Output folder for figures: ', outputFolder)
copyfile(simConfigFile, f'{outputFolder}/{os.path.basename(simConfigFile)}')
iterationReportFile = f'{outputFolder}/iterationReport.yaml'
figFolder = setFigureFolder(outputFolder)
sInfo = SaveInformation(fileName = f'{figFolder}/InitialPlot', savePDF=True, savePNG=True)

# Visualization of the first iteration of the space with the initial sample:
meshRes = 300
figSize = (12,10)

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
# The vector that stores the moving average of the change measure:
changeAvg = [ConvergenceSample._movingAverage(changeMeasure, n=5)]
# Getting the initial accuracy and other performance metrics:
acc = [convergenceSample.getPerformanceMetrics(benchmark = myBench,
                                            classifier=clf,
                                            percentage = True, 
                                            metricType=PerformanceMeasure.ACCURACY)]
precision = [convergenceSample.getPerformanceMetrics(benchmark = myBench,
                                            classifier=clf,
                                            percentage = True, 
                                            metricType=PerformanceMeasure.PRECISION)]
recall = [convergenceSample.getPerformanceMetrics(benchmark = myBench,
                                            classifier=clf,
                                            percentage = True, 
                                            metricType=PerformanceMeasure.RECALL)]
sampleCount = [len(initialLabels)]
print('Initial accuracy: ', acc[0])   
sampleNumbers = [mySpace.sampleNum]
# Calculating the remaining budget:
initialSampleSize = len(initialLabels)
currentBudget = budget - len(initialLabels)

# Setting up the iteration reports file:
iterationReports = []

# Getting the iteration report after the initial sample:
iterationNum = 0
initialReport.budgetRemaining = currentBudget
initialReport.setChangeMeasure(changeMeasure[0])
initialReport.batchSize = initialSampleSize
initialReport.iterationNumber = iterationNum
initialReport.setExploreResults(initialLabels)
initialReport.setExplorers(initialSamples)

iterationReports.append(initialReport)
saveIterationReport(iterationReports, iterationReportFile)

prevClf = clf

# Setting up the object that does the calculations for the resource allocation to exploration and exploitation:
resourceAllocator = ResourceAllocator(convSample = convergenceSample,
                        simConfig = simConfig,
                        outputLocation = outputFolder,
                        initSample = initialSampleSize)

# Setting up the object that saves the metrics outputs at each iteration: 
metricsSaver = MetricsSaver()
metricsSaver.saveMetrics(outputFolder, acc,changeMeasure, precision, recall)
exploreBudgets = []
exploitBudgets = []

while currentBudget > 0 and changeAvg[-1] > 0.4:
    print('------------------------------------------------------------------------------')
    # Setting up the iteration report timing members:
    iterationNum += 1
    iterReport = IterationReport(dimNames, batchSize=batchSize)
    iterReport.setStart()
    print('Current budget: ', currentBudget, ' samples')
    # Finding new points using the exploiter object:
    # NOTE: The classifier has to be passed everytime to the exploiter for update.
    
    # Dynamin resource allocation:
    # NOTE: The overall budget for each group is capped by the current remaining budget. The priority is with exploiration. Exploration is done only if budget is remained after exploitation.
    exploitationBudget = min(currentBudget, exploitationBudget)
    explorationBudget = min(explorationBudget, currentBudget - exploitationBudget)
    print('Dynamic Resource Allocation is active.')
    print('Exploitation budget:', exploitationBudget,' samples')
    print('Exploration budget:', explorationBudget,' samples')
    
    exploiterPoints = exploiter.findNextPoints(pointNum=exploitationBudget)
    explorerPoints = explorer.findNextPoints(pointNum=explorationBudget)
    # Updating the remaining budget:    
    currentBudget -= min(currentBudget, batchSize) 
    # Visualization and saving the results:
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_NotLabeled'
    if iterationNum % 1 ==0 and len(dimNames) <4:
        plotSpace(mySpace, 
            figsize = figSize,
            legend = True, 
            showPlot=False, 
            classifier = clf,
            saveInfo=sInfo,
            meshRes = meshRes,
            newPoints=exploiterPoints,
            explorePoints=explorerPoints if explorationBudget>0 else None,
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

    # Setting the budget for the next iteration:
    exploitBudgets.append(exploitationBudget)
    exploreBudgets.append(explorationBudget)
    # exploitationBudget, explorationBudget = calcExploitBudget, calcExploreBudget
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
    newAccuracy = convergenceSample.getPerformanceMetrics(benchmark = myBench, 
                                            percentage=True, 
                                            classifier=clf,
                                            metricType=PerformanceMeasure.ACCURACY)
    newPrecision = convergenceSample.getPerformanceMetrics(benchmark = myBench, 
                                            percentage=True, 
                                            classifier=clf,
                                            metricType=PerformanceMeasure.PRECISION)
    newRecall = convergenceSample.getPerformanceMetrics(benchmark = myBench, 
                                            percentage=True, 
                                            classifier=clf,
                                            metricType=PerformanceMeasure.RECALL)
    changeMeasure.append(newChangeMeasure)
    changeAvg.append(ConvergenceSample._movingAverage(changeMeasure,n=5))
    acc.append(newAccuracy)
    precision.append(newPrecision)
    recall.append(newRecall)
    sampleCount.append(len(mySpace.eval_labels))
    # Saving the new iteration of the metrics output:
    metricsSaver.saveMetrics(outputFolder, acc=acc,changeMeasure = changeMeasure, precision = precision, recall = recall, sampleCount = sampleCount, changeAvg = changeAvg)
    print('Hypothesis change estimate: ', changeMeasure[-1:], ' %')
    print('Current accuracy estimate: ', acc[-1:], ' %')
    sampleNumbers.append(mySpace.sampleNum)
    # Visualization of the results after the new samples are evaluated:
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_Labeled'
    if iterationNum % 1 == 0:
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
    iterReport.setExploitatives(exploiterPoints)
    iterReport.setExplorers(explorerPoints)
    iterReport.setExploitResults(newLabels)
    iterReport.setExploreResults(exploreLabels)
    iterReport.setChangeMeasure(newChangeMeasure)
    iterationReports.append(iterReport)
    saveIterationReport(iterationReports, iterationReportFile)

import pickle 
import repositories as repo 
pickleLoc = repo.picklesLoc
pickleName = f'{outputFolder}/testClf.pickle'
with open(pickleName, 'wb') as pickleOut:
    pickle.dump(clf, pickleOut) 

# Final visualization of the results: 
plotSpace(space = mySpace, figsize=figSize, legend = True,
                        saveInfo=None, showPlot=True, classifier = clf, 
                        benchmark = myBench, constraints = consVector)




