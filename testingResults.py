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

simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
print(simConfig.name)
for p in simConfig.codeBase:
    sys.path.insert(0,p)
    print(p + ' has been added to the path.')

import slickml
from slickml.metrics import BinaryClassificationMetrics 
from sklearn.metrics import accuracy_score, roc_curve,auc


currentDir = '.'

repoLoc = 'E:/Data/motherSample'
trainRepo = 'E:/Data/adaptRepo2'
variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
dimNames = [var.name for var in variables]



dataset, labels, times = readDataset(repoLoc, dimNames)
print(len(dataset), len(labels))

clf = svm.SVC(kernel = 'rbf', C = 1000, probability=True)
clf.fit(dataset,labels)


predLabels = clf.predict(dataset)
diff = 1 - np.sum(np.abs(predLabels - labels)) / len(labels)
print(diff * 100)
print(np.sum(labels) / len(labels))

predictProba = clf.predict_proba(dataset)
train_pred = predictProba[:,1]
trainFpr, trainTpr, thresholds = roc_curve(labels, train_pred)
print(predictProba)


clf_metrics = BinaryClassificationMetrics(labels, train_pred) 
clf_metrics.plot()