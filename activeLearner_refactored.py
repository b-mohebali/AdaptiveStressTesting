from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import *
import logging
import os,sys
import subprocess
from ActiveLearning.benchmarks import DistanceFromCenter, Branin, Benchmark
from ActiveLearning.Sampling import *
import platform
import shutil
import matplotlib.pyplot as plt 
from enum import Enum
from sklearn import svm
from geneticalgorithm import geneticalgorithm as ga
from ActiveLearning.optimizationHelper import GeneticAlgorithmSolver as gaSolver

from ActiveLearning.visualization import *
import time 
import numpy as np 

# Loading the config files of the process:
simConfig = simulationConfig('./assets/yamlFiles/adaptiveTesting.yaml')
variableFiles = './assets/yamlFiles/varAdaptTest.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variableFiles, scalingScheme=Scale.LINEAR)

# Loading the parameters of the process:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize

# Defining the design space based on the variables config file: 
mySpace = Space2(variableList=variables)

# Defining the benchmark:
myBench = DistanceFromCenter(threshold=1.5, inputDim=len(variables), center = [2] * len(variables))
# Generating the initial sample. This step is pure exploration MC sampling:
initialSamples = generateInitialSample(space = mySpace, 
                                        sampleSize = initialSampleSize,
                                        method = InitialSampleMethod.CVT,
                                        checkForEmptiness=False)
# Getting the labels for the initial sample:
initialLabels = myBench.getLabelVec(initialSamples)

# Initial iteration of the classifier trained on the initial samples and their labels:
clf = svm.SVC(kernel = 'rbf', C = 1000)
clf.fit(initialSamples, initialLabels)
# Adding the samples and their labels to the space: 
mySpace.addSamples(initialSamples, initialLabels)

# Visualization of the first iteration of the space with the initial sample:
figFolder = simConfig.figFolder
sInfo = SaveInformation(fileName = f'{figFolder}/InitialPlot', savePDF=True, savePNG=True)
plotSpace(mySpace, 
        classifier=clf, 
        figsize = (10,8), 
        legend = True, 
        showPlot=False,
        saveInfo = sInfo, 
        benchmark = myBench)
plt.close()

# Defining the optimizer: 
optimizer = gaSolver(space = mySpace, 
                    epsilon = 0.05,
                    batchSize = simConfig.batchSize,
                    convergence_curve=False,
                    progress_bar=False)

# Defining the convergence sample that implementes the change measure as well as
#   the performance metrics for the process. 
convergenceSample = ConvergenceSample(mySpace)
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

while currentBudget > 0:
    print('------------------------------------------------------------------------------')
    print('Current budget: ', currentBudget, ' samples')
    # Finding new points using the optimizer object:
    # NOTE: The classifier has to be passed everytime to the optimizer for update.
    newPointsfound = optimizer.findNextPoints(clf, min(currentBudget, batchSize))
    # Updating the remaining budget:
    currentBudget -= min(currentBudget, batchSize) 
    # Visualization and saving the results:
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_NotLabeled'
    plotSpace(mySpace, 
        figsize = (10,8),
        legend = True, 
        showPlot=False, 
        classifier = clf,
        saveInfo=sInfo,
        newPoint=newPointsfound,
        benchmark = myBench)
    plt.close()
    # Evaluating the newly found samples: 
    newLabels = myBench.getLabelVec(newPointsfound)
    # Adding the newly evaluated samples to the dataset:
    mySpace.addSamples(newPointsfound, newLabels)
    # Training the next iteration of the classifier:
    clf = svm.SVC(kernel = 'rbf', C=1000)
    clf.fit(mySpace.samples, mySpace.eval_labels)
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
        figsize = (10,8),
        legend = True,
        classifier = clf, 
        benchmark = myBench,
        newPoint=None,
        saveInfo=sInfo,
        showPlot=False)
    plt.close()

# Final visualization of the results: 
plotSpace(space = mySpace, figsize=(10,8), legend = True, newPoint = None, 
                        saveInfo=sInfo, showPlot=True, classifier = clf, 
                        benchmark = myBench)




