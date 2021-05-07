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

simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
print(simConfig.name)
for p in simConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

from autoRTDS import Trial
import case_Setup
from rscad import rtds
from repositories import *
import simulation
from ShowRunner import *

from ActiveLearning.Sampling import *
from ActiveLearning.dataHandling import *
from metricsRunTest import getMetricsResults

print('This is the AC PGM sampling test file. ')

variablesFile = './assets/yamlFiles/ac_pgm_adaptive.yaml'

# Extracting the hyperparameters of the analysis:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize

# We use this script to generate the MC sample for benchmark:
initialSampleSize = 5000


variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

for var in variables:
    print(var.name, var.description)

# Setting the main files and locations:
descriptionFile = './assets/yamlFiles/varDescription.yaml'
sampleSaveFile = './assets/experiments/mother_sample_2.txt'
repoLoc = motherSample2

# repoLoc = 'D:/Data/adaptiveRepo1'
# Figure folder for the metrics outputs:
figFolder = repoLoc + '/figures'

# Forming the space:
designSpace = Space(variableList= variables, initialSampleCount=initialSampleSize)
dimNames = designSpace.getAllDimensionNames()
# # Getting the initial sample and saving it to a location:
designSpace.generateInitialSample(method = InitialSampleMethod.LHS)


# formattedSample = getSamplePointsAsDict(dimNames, sampleList)

# saveSampleToTxtFile(samples = formattedSample, fileName = sampleSaveFile)
# saveVariableDescription(variables, descriptionFile)
# copyDataToremoteServer(repoLoc, descriptionFile)
# copyDataToremoteServer(repoLoc, variablesFile)

# # Running the initial sample:
formattedSample = loadSampleFromTxtFile(sampleSaveFile)

sampleGroup = range(2400, initialSampleSize + 1)
runSample(sampleDictList=formattedSample, 
        dFolder = dataFolder, 
        remoteRepo=repoLoc,
        sampleGroup=sampleGroup)

