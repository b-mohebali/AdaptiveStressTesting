# # ! /usr/bin/python

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
from sklearn import svm

import pickle

import repositories

simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
print(simConfig.name)
for p in simConfig.codeBase:
    sys.path.insert(0,p)
    print(p + ' has been added to the path.')

import slickml
from slickml.metrics import BinaryClassificationMetrics 
from sklearn.metrics import accuracy_score, roc_curve,auc
from sklearn.metrics import precision_recall_curve


currentDir = '.'

# Data location:
repoLoc = 'E:/Data/motherSample'
trainRepo = 'E:/Data/adaptiveRepo2'

# Config file location:
variableFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'
variables = getAllVariableConfigs(yamlFileAddress = variableFile, 
                                   scalingScheme= Scale.LINEAR)

dimNames = [var.name for var in variables]

dataset,labels = readDataset(trainRepo, dimNames)

# Defining the design space: 
designSpace = SampleSpace(variableList= variables)
designSpace._samples, designSpace._eval_labels = dataset, labels

# Defining the visualization parameters:
insigDims = [0,2] # 0-indexed dimensions that are on the grid axes.
figSize = (12,12)
gridRes = (3,3)


legend = True 
showPlot = True 
meshRes = 100

clf = svm.SVC(kernel = 'rbf', C = 1000)
clf.fit(dataset, labels)
classifier = clf
#########################################################
## Setting up the saving location:

outputFolder = currentDir + '/assets/Figures/test'
figFolder = setFigureFolder(outputFolder)
saveFigs = True 
sInfo = SaveInformation(fileName = f'{figFolder}/grid_plot', savePDF=True, savePNG=True)

# # Creating the mother sample classifier as athe benchmark for the visualiztion:

motherData, motherLabels = readDataset(repoLoc, dimNames)
motherClf = svm.SVC(kernel = 'rbf', C = 1000)
motherClf.fit(motherData, motherLabels)
from ActiveLearning.benchmarks import TrainedSvmClassifier
threshold = 0.5 if motherClf.probability else 0
classifierBench = TrainedSvmClassifier(motherClf, len(variables), threshold)

pickleFile = repositories.picklesLoc + 'classifier_bench.pickle'
with open(pickleFile, "wb") as pickleOut:
    pickle.dump(classifierBench, pickleOut)


# Pickling the mother sample and its trained classifier:


# Calling the visualization procedure:
plotSpace(designSpace,
            figsize = figSize,
            meshRes = 100,
            classifier = clf,
            gridRes = gridRes,
            showPlot=False,
            saveInfo=sInfo,
            insigDimensions=insigDims,
            legend = True,
            # prev_classifier=motherClf,
            benchmark = classifierBench)   
