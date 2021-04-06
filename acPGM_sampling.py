# #! /usr/bin/python3

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

simConfig = simulationConfig('./yamlFiles/ac_pgm_conf.yaml')
print(simConfig.name)
for p in simConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

# from autoRTDS import Trial
# import case_Setup
# from rscad import rtds
# from repositories import *
# import simulation
# from ShowRunner import *

from ActiveLearning.Sampling import *
from ActiveLearning.dataHandling import *
# from metricsRunTest import getMetricsResults

print('This is the AC PGM sampling test file. ')

variablesFile = './assets/yamlFiles/ac_pgm_adaptive.yaml'

# Extracting the hyperparameters of the analysis:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize



variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

for var in variables:
    print(var.name, var.description)

# Setting the main files and locations:
descriptionFile = './assets/yamlFiles/varDescription.yaml'
sampleSaveFile = './assets/experiments/mother_sample.txt'
# repoLoc = motherSample

repoLoc = 'D:/Data/adaptiveRepo1'
# Figure folder for the metrics outputs:
figFolder = repoLoc + '/figures'

# Forming the space:
designSpace = Space(variableList= variables, initialSampleCount=initialSampleSize)
currentBudget = budget - initialSampleSize

# # Getting the initial sample and saving it to a location:
# designSpace.generateInitialSample()
# formattedSample = designSpace.getSamplePointsAsDict()
# saveSampleToTxtFile(samples = formattedSample, fileName = sampleSaveFile)
# saveVariableDescription(timeIndepVars, descriptionFile)
# copyDataToremoteServer(simRepo, descriptionFile)
# copyDataToremoteServer(simRepo, variablesFile)

# # Running the initial sample:
# runSample(sampleDictList=formattedSample, dFolder = dataFolder, remoteRepo=repoLoc)

# Running the metrics on the initial sample:
# for sampleIndex in range(1,initialSampleSize+1):
#     getMetricsResults(dataLocation=repoLoc, 
#                       sampleNumber=sampleIndex, 
#                       figFolderLoc=figFolder)
# initSample = generateInitialSample(space = designSpace, 
#                                     sampleSize = 2500)
# formattedSample = getSamplePointsAsDict(designSpace, initSample)
# saveSampleToTxtFile(formattedSample, fileName = sampleSaveFile)

# formattedSample = loadSampleFromTxtFile(sampleSaveFile)
# runSample(sampleDictList=formattedSample, dFolder = dataFolder, remoteRepo=repoLoc)
    
# Loading the labels from an evaluated repo:

startTime = time.time()
ds, l = readDataset(repoLoc,variables = variables)
endTime = time.time()
print(f'Time taken for {len(l)} samples:', endTime-startTime, ' seconds')
for var in variables:
    print(var.name)
for idx,dp  in enumerate(ds):
    print(l[idx], dp)
