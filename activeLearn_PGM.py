#! /usr/bin/python3

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
from metricsRunTest import * 

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



"""
Steps of checks for correctness: 
    DONE 1- Run an FFD sample with the 4 variables in a new location with the control object
        from simulationHelper.py script. -> DONE
    2- Implement the initial sampling using the combination of the control objects and 
        the developed active learning code. 
    3- Implement the exploitation part. Save the change measures in each step in case the 
        process is interrupted for any reason.
        
"""
print('This is the AC PGM sampling test file. ')
variablesFile = './assets/yamlFiles/ac_pgm_adaptive.yaml'

# Extracting the hyperparameters of the analysis:
budget = simConfig.sampleBudget
batchSize = simConfig.batchSize
initialSampleSize = simConfig.initialSampleSize

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

# Setting the main files and locations:
descriptionFile = './assets/yamlFiles/varDescription.yaml'
sampleSaveFile = './assets/experiments/test_sample.txt'
repoLoc = adaptRepo2

# Defining the design space and the handler for the name of the dimensions. 
designSpace = Space2(variableList=variables)
dimNames = designSpace.getAllDimensionNames()

# Taking the initial sample based on the parameters of the process. 
initialSamples = generateInitialSample(space = designSpace,
                                        sampleSize=initialSampleSize,
                                        method = InitialSampleMethod.CVT,
                                        checkForEmptiness=False)

# Preparing and running the initial sample: 
formattedSample = getSamplePointsAsDict(designSpace, initialSamples)
saveSampleToTxtFile(formattedSample, sampleSaveFile)
runSample(sampleDictList=formattedSample, 
        dFolder = dataFolder,
        remoteRepo=repoLoc,
        simConfig=simConfig)

# Running the metrics on the first sample: 
setUpMatlab(simConfig=simConfig)
samplesList = list(range(1, initialSampleSize+1))
getMetricsResults(dataLocation=repoLoc, sampleNumber = samplesList)













