"""
    NOTE: This code is old and will probably not work due to code refactoring.
    Please refer to activeLearner_refactored.py as an example of how to use 
    the code base. 
"""

from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import *
import logging
import os,sys
import subprocess
from ActiveLearning.benchmarks import DistanceFromCenter, Branin
from ActiveLearning.Sampling import *
import platform
import shutil
import numpy as np 
import matplotlib.pyplot as plt 
from enum import Enum
from sklearn import svm
from geneticalgorithm import geneticalgorithm as ga
from ActiveLearning.optimizationHelper import GeneticAlgorithmSolver as gaSolver

from ActiveLearning.visualization import *

import time
import numpy as np 


simConfig = simulationConfig('./assets/yamlFiles/adaptiveTesting.yaml')
variablesFiles = './assets/yamlFiles/varAdaptTest.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variablesFiles, scalingScheme=Scale.LINEAR)


# Setting up the design space or the sampling space:
budget = simConfig.sampleBudget # Number of sampels:
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize

mySpace = Space(variableList = variables,initialSampleCount = initialSampleSize)
currentBudget = budget - initialSampleSize

myBench = DistanceFromCenter(threshold = 1.5, inputDim = len(variables), center = [2,2])

mySpace.generateInitialSample()
mySpace.getBenchmarkLabels(myBench)
print(mySpace.eval_labels)

mySpace.fit_classifier()
figFolder = simConfig.figFolder
sInfo = SaveInformation(fileName=f'{figFolder}/InitialPlot', savePDF=True, savePNG=True)

print(mySpace.getAllDimensionBounds())

plotSpace(mySpace,classifier = None,figsize = (10,8), legend = True, saveInfo=sInfo, showPlot=False, meshRes=50)
plt.close()
optimizer = gaSolver(space = mySpace, classifier = mySpace.clf, epsilon = 0.05, batchSize= 5, convergence_curve=False, progress_bar = False)
changeMeasure = [mySpace.getChangeMeasure(percent = True, updateConvLabels = True)]
# Iterations of the exploitation:
sampleNumbers = [len(mySpace.samples)]
currentBudget = budget - initialSampleSize
acc = [mySpace.getAccuracyMeasure(percent = True)]
print('Accuracy: ', acc)

while currentBudget > 0:
    print('Current budget = ', currentBudget)
    newPointsFound = optimizer.findNextPoints(min(currentBudget, batchSize))
    currentBudget -= min(currentBudget, batchSize)
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_NotLabeled'
    plotSpace(mySpace,figsize = (10,8), legend = True, newPoint = newPointsFound, saveInfo=sInfo, showPlot=False, meshRes=25)
    plt.close()
    mySpace.addPointsToSampleList(newPointsFound)
    mySpace.getBenchmarkLabels()
    mySpace.fit_classifier()
    changeMeasure.append(mySpace.getChangeMeasure(percent = True, updateConvLabels=True))
    acc.append(mySpace.getAccuracyMeasure(percent = True))
    print('Hypothesis change estimate: ', changeMeasure[-1:], '%')
    print('Current Accuracy estimate: ',acc[-1:],'%')
    sampleNumbers.append(len(mySpace.samples))
    sInfo.fileName = f'{figFolder}/bdgt_{currentBudget}_Labeled'
    plotSpace(mySpace,figsize = (10,8), legend = True, newPoint = None, saveInfo=sInfo, showPlot=False, meshRes=25)
    plt.close()
plotSpace(mySpace,figsize = (10,8), legend = True, newPoint = None, saveInfo=sInfo, showPlot=True)
