from ActiveLearning.dataHandling import MetricsSaver, getFirstEmptyFolder
from yamlParseObjects.yamlObjects import * 
from yamlParseObjects.variablesUtil import *
import logging
from shutil import copyfile
import subprocess
from ActiveLearning.benchmarks import DistanceFromCenter, Branin, Benchmark, Hosaki
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
# outputFolder = f'{simConfig.outputFolder}/{72}'
print('Output folder for figures: ', outputFolder)
copyfile(simConfigFile, f'{outputFolder}/{os.path.basename(simConfigFile)}')
iterationReportFile = f'{outputFolder}/iterationReport.yaml'
figFolder = setFigureFolder(outputFolder)
sInfo = SaveInformation(fileName = f'{figFolder}/0_InitialPlot', savePDF=True, savePNG=True)

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

# Finding the next point: 
supportVectors = clf.getSupportVectors(standard = True)
print(supportVectors)

from collections import namedtuple 
SVDistance = namedtuple('SVDistance', ('distance', 'supportVector'))

labels = clf.predict(supportVectors)
print(labels)

r = mySpace.ranges 
supportVectorDistances = [] 

def minDistanceFromOppositeClass(sv, label, space:SampleSpace,r = None):
    if r is None:
        r = space.ones
    oppLabel = label ^1 
    # oppPoints = space.samples[sp]
    distances = np.linalg.norm(np.divide(space.samples - sv,r), axis = 1)
    distance = min(distances[space._eval_labels==oppLabel])
    foundPointIdx = np.where(distances == distance)[0]
    foundPoint = space.samples[foundPointIdx,:].squeeze()
    return foundPoint, distance

def findNextPoint(classifier: StandardClassifier, space:SampleSpace):
    # TODO: This implementation is only for one point. The batch sampling is simple and has to be done later.
    bestPoints = {}
    supportVectors = classifier.getSupportVectors(standard = True)
    labels = classifier.predict(supportVectors)
    for idx,sv in enumerate(supportVectors):
        svLabel = labels[idx]
        foundPoint, svDistance = minDistanceFromOppositeClass(sv, svLabel, space,r)
        print(f'Found point for SV # {idx+1}: {foundPoint}, SV itself: {sv}')
        bestPoints[svDistance] = (sv,foundPoint)
    bestDistance = max(bestPoints.keys())
    newPoint = (bestPoints[bestDistance][0] + bestPoints[bestDistance][1])/2
    return newPoint 

budget = 10
for _ in range(budget):
    newPoint = findNextPoint(clf, mySpace)
    sInfo.fileName = f'{figFolder}/{_+1}_before_label'
    plotSpace(mySpace, 
        classifier=clf, 
        figsize = figSize, 
        meshRes=meshRes,
        legend = True, 
        showPlot=False,
        newPoints = [newPoint],
        saveInfo = sInfo, 
        benchmark = myBench,
        constraints = consVector)
    plt.close()
    print(_,newPoint)
    newLabel = myBench.getLabel(newPoint)
    mySpace.addSample(newPoint, newLabel)
    clf = StandardClassifier(C = 1000)
    clf.fit(mySpace.samples, mySpace.eval_labels)
    sInfo.fileName = f'{figFolder}/{_+1}_labeled'
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
