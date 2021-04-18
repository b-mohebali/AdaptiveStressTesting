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


repoLoc = adaptRepo2
variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
dimNames = [var.name for var in variables]

dataset, labels, times = readDataset(repoLoc, dimNames)
print(dataset, labels)

clf = svm.SVC(kernel = 'rbf', C = 1000)
clf.fit(dataset,labels)

from sklearn.metrics import accuracy_score, roc_curve,auc

predLabels = clf.predict(dataset)
diff = 1 - np.sum(np.abs(predLabels - labels)) / len(labels)
print(diff * 100)
train_pred = clf.decision_function(dataset)
trainFpr, trainTpr, thresholds = roc_curve(labels, train_pred)

plt.plot(trainFpr, trainTpr, label = 'AUC train = '+str(auc(trainFpr, trainTpr)))
plt.plot([0,1],[0,1],'g--')

plt.show()

