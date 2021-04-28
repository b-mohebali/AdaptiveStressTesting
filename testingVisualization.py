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

# Data location:
repoLoc = 'E:/Data/motherSample'
trainRepo = 'E:/Data/adaptiveRepo2'

# Config file location:
variableFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'
variables = getAllVariableConfigs(yamlFileAddress = variableFile, 
                                   scalingScheme= Scale.LINEAR)

dimNames = [var.name for var in variables]

dataset,labels = readDataset(trainRepo, dimNames)


# def _plotSpace4D(space: Space,
#                 insigDimensions,
#                 showPlot = True, 
#                 classifier = None,
#                 figsize = (6,6),
#                 legend = True,
#                 benchmark = None,
#                 meshRes = 100,
#                 gridRes = (4,4),
#                 saveInfo: SaveInformation = None):

designSpace = SampleSpace(variableList= variables)
designSpace._samples, designSpace._eval_labels = dataset, labels

insigDims = [0,2] # 0-indexed dimensions that are on the grid axes.
figSize = (8,8)
gridRes = (4,4)


legend = True 
showPlot = True 
meshRes = 100

clf = svm.SVC(kernel = 'rbf', C = 1000)
clf.fit(dataset, labels)
classifier = clf
#########################################################
## Implementation of the 4D visualization: 
allDims = list(range(4))
sigDims = [_ for _ in allDims if _ not in insigDims]

ranges = designSpace.getAllDimensionBounds()

sigDim1Range = ranges[sigDims[0]]
sigDim2Range = ranges[sigDims[1]]
insigDim1Range = ranges[insigDims[0]]
insigDim2Range = ranges[insigDims[1]]

insigDim1Vals = np.linspace(start = insigDim1Range[0],
                            stop = insigDim1Range[1],
                            num = gridRes[0],
                            endpoint = True)
insigDim2Vals = np.linspace(start = insigDim2Range[0],
                            stop = insigDim2Range[1],
                            num = gridRes[1],
                            endpoint = True)


xx = np.linspace(start = sigDim1Range[0],
                stop = sigDim1Range[1],
                num = meshRes)
yy = np.linspace(start = sigDim2Range[0],
                stop = sigDim2Range[1],
                num = meshRes)

XX,YY = np.meshgrid(xx,yy, indexing = 'ij')
xy = np.vstack([XX.ravel(), YY.ravel()]).T

dataVec = np.zeros(shape = (YY.size, 4), dtype = float)


onesVec = np.ones((XX.size,),dtype = float)
insigVec1 = onesVec * insigDim1Vals[0]
insigVec2 = onesVec * insigDim2Vals[0]

dataVec[:,insigDims[0]] = insigVec1
dataVec[:,insigDims[1]] = insigVec2
dataVec[:,sigDims[0]] = xy[:,0]
dataVec[:,sigDims[1]] = xy[:,1]

# Creating the mother sample classifier as athe benchmark for the visualiztion:

motherData, motherLabels = readDataset(repoLoc, dimNames)
motherClf = svm.SVC(kernel = 'rbf', C = 1000)
motherClf.fit(motherData, motherLabels)
from ActiveLearning.benchmarks import TrainedSvmClassifier
threshold = 0.5 if motherClf.probability else 0
benchmark = TrainedSvmClassifier(motherClf, len(variables), threshold)


plotNum = 1
fig,ax = plt.subplots(nrows = gridRes[0], ncols = gridRes[1], figsize = (10,10))
fig.tight_layout()
for rowNum in range(gridRes[0]):
    for colNum in range(gridRes[1]):
        dataVec[:,insigDims[0]] = onesVec * insigDim1Vals[rowNum]
        dataVec[:,insigDims[1]] = onesVec * insigDim2Vals[colNum]
        ax = plt.subplot(gridRes[0],gridRes[1],plotNum)
        decisionFunction = classifier.decision_function(dataVec).reshape(XX.shape)
        cs2 = ax.contour(XX, YY, decisionFunction, colors='k', levels=[-1,0,1], alpha=1,linestyles=['dashed','solid','dotted'])
        csLabels2 = ['DF=-1','DF=0 (hypothesis)','DF=+1']
        if plotNum ==1:
            for i in range(len(csLabels2)):
                cs2.collections[i].set_label(csLabels2[i])
        
        ### Tagging and labeling the axes:
        if plotNum <= gridRes[1]:
            ax.set_title(f'{dimNames[insigDims[1]]} = {insigDim2Vals[colNum]:.4f}')
        if plotNum%gridRes[1]==0:
            ax2 = ax.twinx()
            ax2.set_ylabel(f'{dimNames[insigDims[0]]} = {insigDim1Vals[rowNum]:.4f}')
            ax2.set_yticklabels([])
        if plotNum%gridRes[1]==1:
            ax.set_ylabel(dimNames[sigDims[1]])
        if (plotNum+gridRes[1]) > (gridRes[0]*gridRes[1]):
            ax.set_xlabel(dimNames[sigDims[0]])
        # Plotting the benchmark classifier
        if benchmark is not None:
            scores = benchmark.getScoreVec(dataVec).reshape(XX.shape)
            cs = ax.contour(XX,YY,scores, colors='r', levels = [benchmark.threshold], 
                alpha = 1, linestyles = ['dashed']) 
            cslabels = ['Actual Boundary']
            ax.clabel(cs, inline=1, fontsize=10) 
            if plotNum==1:
                for i in range(len(cslabels)):
                    cs.collections[i].set_label(cslabels[i])
        
        # Updating the plot number, needed for locating the subplots.
        plotNum += 1
        
if legend:
    fig.legend(loc = 'upper left',bbox_to_anchor=(1.0, 1.0))
        



