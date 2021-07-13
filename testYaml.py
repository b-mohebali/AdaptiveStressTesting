#! /usr/bin/python3

from yamlParseObjects.yamlObjects import *
from yamlParseObjects.variablesUtil import *
import os
from profileExample.profileBuilder import * 
from eventManager.eventsLogger import * 
from scipy.linalg import hadamard
import matplotlib.pyplot as plt
from ActiveLearning.Sampling import *
from ActiveLearning.dataHandling import *
from ActiveLearning.visualization import * 
from ActiveLearning.optimizationHelper import GA_Exploiter, GA_Explorer
from ActiveLearning.benchmarks import TrainedSvmClassifier
from sklearn import svm
from ActiveLearning.simInterface import *
from repositories import *
from metricsRunTest import * 
from multiprocessing import freeze_support






simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
variablesFile = currentDir + '/assets/yamlFiles/ac_pgm_adaptive.yaml'
repoLoc = adaptRepo9
dataLoc = repoLoc + '/data'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)
designSpace = SampleSpace(variableList = variables)
dimNames = designSpace.getAllDimensionNames()
dataset, labels = readDataset(dataLoc, dimNames=dimNames)
print(dataset)
print(labels)

# Getting the mean value of all the dimensions: 
means = np.mean(dataset, axis = 0)
print(means)
stds = np.std(dataset, axis = 0)
print(stds)
normalDataset = dataset - means
print(normalDataset)
normalDataset = normalDataset / stds
print(normalDataset)