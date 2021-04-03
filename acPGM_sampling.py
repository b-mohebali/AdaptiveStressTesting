## #! /usr/bin/python3

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

from ActiveLearning.Sampling import Space
from metricsRunTest import getMetricsResults

print('This is the AC PGM sampling test file. ')

variablesFile = './yamlFiles/ac_pgm_adaptive.yaml'
descriptionFile = 'varDescription.yaml'

# Extracting the hyperparameters of the analysis:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize



variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

for var in variables:
    print(var.name, var.description)

# Setting the main files and locations:
descriptionFile = 'varDescription.yaml'
sampleSaveFile = './experiments/adaptive_initial.txt'
# repoLoc = adaptRepo1

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
for sampleIndex in range(1,initialSampleSize+1):
    getMetricsResults(dataLocation=repoLoc, 
                      sampleNumber=sampleIndex, 
                      figFolderLoc=figFolder)

    
