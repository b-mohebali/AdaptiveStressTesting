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
from sklearn.metrics import precision_recall_curve


currentDir = '.'

repoLoc = 'E:/Data/motherSample'
trainRepo = 'E:/Data/adaptiveRepo2'
variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
dimNames = [var.name for var in variables]

dataset, labels, times = readDataset(repoLoc, dimNames)
print(len(dataset), len(labels))

trainData, trainLabels, trainTimes = readDataset(trainRepo, dimNames)


clf = svm.SVC(kernel = 'rbf', C = 1000, probability=True)
clf.fit(dataset,labels)
predLabels = clf.predict(dataset)
diff = 1 - np.sum(np.abs(predLabels - labels)) / len(labels)
print('Accuracy of the benchmark classifier: ',diff * 100)
print('Percentage of the positive points to all data points:',np.sum(labels) / len(labels))
predictProba = clf.predict_proba(dataset)
train_pred = predictProba[:,1]
trainFpr, trainTpr, thresholds = roc_curve(labels, train_pred)
precision,recall, thresholds = precision_recall_curve(labels, train_pred)
clf_metrics = BinaryClassificationMetrics(labels, train_pred) 

# clf = svm.SVC(kernel = 'rbf', C = 1000, probability=True)
# clf.fit(trainData, trainLabels)
# predLabels = clf.predict(trainData)
# diff = 1 - np.sum(np.abs(predLabels - trainLabels)) / len(trainLabels)
# print('Accuracy of the benchmark classifier: ',diff * 100)
# print('Percentage of the positive points to all data points:',np.sum(trainLabels) / len(trainLabels))
# predictProba = clf.predict_proba(trainData)
# train_pred = predictProba[:,1]
# trainFpr, trainTpr, thresholds = roc_curve(trainLabels, train_pred)

# clf_metrics = BinaryClassificationMetrics(trainLabels, train_pred) 
# precision,recall, thresholds = precision_recall_curve(trainLabels, train_pred)

clf_metrics.plot()

thresholds = np.insert(thresholds,0,0)

plt.subplots(3,1,figsize = (4,5))
ax = plt.subplot(311)
plt.plot(thresholds, precision)
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.grid(True)

ax = plt.subplot(312)
plt.plot(thresholds, recall)
plt.xlabel('Threshold')
plt.ylabel('recall')
plt.grid(True)

ax = plt.subplot(313)
plt.plot(recall, precision)
plt.ylabel('Precision')
plt.xlabel('recall')
plt.grid(True)
plt.show()


