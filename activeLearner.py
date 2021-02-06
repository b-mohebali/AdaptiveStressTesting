from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import *
import logging
import os,sys
import subprocess
from ActiveLearning.benchmarks import Branin
from ActiveLearning.Sampling import *
import platform
import shutil
import numpy as np 
import matplotlib.pyplot as plt 
from enum import Enum
from sklearn import svm
from geneticalgorithm import geneticalgorithm as ga

from plotter import *

import time
import numpy as np 


simConfig = simulationConfig('./yamlFiles/adaptiveTesting.yaml')
variablesFiles = './yamlFiles/varAdaptTest.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variablesFiles, scalingScheme=Scale.LINEAR)


# Setting up the design space or the sampling space:
budget = 60 # Number of sampels:
initialSampleSize = 20

mySpace = Space(variableList = variables,initialSampleCount = initialSampleSize)
currentBudget = budget - initialSampleSize


myBench = DistanceFromOrigin(threshold = 7, inputDim = 2, center = [5,5])
# x1 = [1,2,4,4,3]
# x2 = [2,4,1,3,4]
# x = np.array([[1,2],[2,4],[4,1],[4,3],[3,4]])
# l = myBench.getLabelVec(x)
# print(l)
# s = myBench.getScoreVec(x)
# print(s)
mySpace.generateInitialSample()
mySpace.getBenchmarkLabels(myBench)

print(mySpace.eval_labels)

clf = svm.SVC(kernel = 'rbf', C =1000)
clf.fit(mySpace.samples, mySpace.eval_labels)

figFolder = simConfig.figFolder

sInfo = SaveInformation(fileName=f'{figFolder}/thisPLot', savePDF=True, savePNG=True)


print(mySpace.getAllDimensionBounds())

# plotSpace(mySpace,classifier = clf,figsize = (8,6), legend = True, newPoint=[[6,15],[1,1],[2,2],[3,3]], saveInfo=sInfo, showPlot=True)



# Iterations of the exploitation:
# while currentBudget > 0:




