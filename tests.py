# #! /usr/bin/python3

from ActiveLearning.visualization import setFigureFolder, SaveInformation, plotSpace
from ActiveLearning.benchmarks import TrainedSvmClassifier
from ActiveLearning.dataHandling import *
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.Sampling import * 
import matplotlib.pyplot as plt 
import repositories as repos
import os 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import pickle 

motherClfPickle = repos.picklesLoc + 'mother_clf_constrained.pickle'
with open(motherClfPickle, 'rb') as pickleIn:
    motherClf = pickle.load(pickleIn)

print(motherClf)

sv = motherClf.getSupportVectors()
print(sv.shape)

simConfig = simulationConfig('./assets/yamlFiles/adaptiveTesting.yaml')
outputFolder = simConfig.outputFolder
print(outputFolder)

outputFolder= getFirstEmptyFolder(outputFolder) 
print(f'{simConfig.outputFolder}/{outputFolder}')