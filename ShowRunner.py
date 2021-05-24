#! /usr/bin/python3

from metricsRunTest import setUpMatlab
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
from ActiveLearning.simulationHelper import *

simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
print(simConfig.name)
for p in simConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup
from rscad import rtds
from repositories import *
import simulation

def main():
    # -------------------------- File path definitions ---------------------------------------------------------------


    #------------------------------- Setting up the variables -----------------------------------------------
    print('Current directory: ', currentDir)
    variablesFile = './assets/yamlFiles/variables_ac_pgm.yaml'
    descriptionFile = './assets/yamlFiles/varDescription.yaml'

    variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR, span = 0.65 )
    # variables = getAllVariableConfigs('variables_limited.yaml', scalingScheme=Scale.LOGARITHMIC)
    for v in variables: 
        print(f'Variable: {v.name}, mapped name: {v.mappedName}, Initial value: {v.initialState}')
    print('---------------------------------------------------------------')
    logFile = getLoggerFileAddress(fileName='MyLoggerFile')

    logging.basicConfig(filename=logFile, filemode='w', 
                        level = logging.DEBUG,
                        format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

    logging.debug('This is the debug message from the CAPS machine...')
    
    #----------------------------------------------------------------
    # First order Sensitivity Analysis:

    # samplesNum = 960
    # subInters = 16
    # varDict = getTimeIndepVarsDict(variables)
    # randList = randomizeVariablesList(varDict, samplesNum, subInters,scalingScheme=Scale.LOGARITHMIC, saveHists=True)
    # saveSampleToTxtFile(randList, './assets/experiments/limitedVarianceBased.txt')
    # runSampleFrom(sampleDictList= randList, dFolder = dataFolder, remoteRepo=testRepo, fromSample=960)


    #------------------------------------------------------------------
    # One at a time experiment design sensitivity analysis (Standard):

    # sample config
    # experFile = './assets/experiments/OATSampleStandard_Complete.txt'
    # simRepo = remoteRepo55
    # timeIndepVars = getTimeIndepVarsDict(variables)

    ### Standard OAT sample code:
    # exper = standardOATSampleGenerator(timeIndepVars, repeat = False)

    ### Strict OAT sample code:
    # experFile = './assets/experiments/OATSampleStrict_Complete.txt'
    # exper = strictOATSampleGenerator(timeIndepVars)

    # saveSampleToTxtFile(exper,experFile)
    # saveVariableDescription(variables, descriptionFile)
    # copyDataToremoteServer(simRepo, experFile)
    # copyDataToremoteServer(simRepo, descriptionFile)

    ### Load sample from pregenerated sample:
    # exper = loadSampleFromTxtFile(experFile)

    # runSample(sampleDictList=exper,dFolder = dataFolder, remoteRepo = simRepo)
    # runSampleFrom(sampleDictList=exper,dFolder = dataFolder, remoteRepo = simRepo, fromSample = 7)

    # ----------------------------------------------------------------------
    # The Fractional Factorial Desing with Hadamard matrices:

    experFile = './assets/experiments/FFD_AC_PGM.txt'
    simRepo = remoteRepo90
    # Taking the variables with non-zero initialState value

    timeIndepVars = getTimeIndepVars(variables, shuffle = True, omitZero = True)
    print('========================================================================')
    for var in variables: 
        print(f'Name: {var.name}, Span: [{var.lowerLimit},{var.upperLimit}]')    
    print('========================================================================')

    exper = fractionalFactorialExperiment(timeIndepVars, res4 = True)
    saveSampleToTxtFile(exper, fileName = experFile)
    saveVariableDescription(timeIndepVars, descriptionFile)
    copyDataToremoteServer(simRepo, experFile)
    copyDataToremoteServer(simRepo, descriptionFile)
    # exper = loadSampleFromTxtFile(experFile)

    runSample(sampleDictList=exper,
                dFolder = dataFolder, 
                remoteRepo = simRepo,
                simConfig=simConfig)
    # runSampleFrom(sampleDictList = exper, dFolder = dataFolder, remoteRepo = simRepo, fromSample = 12)

    # ----------------------------------------------------------------------
    # Returning all the variables to their standard value:
    # myControl = PGM_control('', './')   
    # myControl.setVariablesToInitialState(variables)


    #---------------------------------------------------------------------
    # Verification sample:

    # simRepo = remoteRepo22
    # experFile = './assets/experiments/VerifSample_TC_Added.txt'
    # logVariables = getAllVariableConfigs(variablesFile, scalingScheme=Scale.LOGARITHMIC)
    # timeIndepVars = getTimeIndepVars(logVariables)
    # exper = generateVerifSample(timeIndepVars)
    # saveSampleToTxtFile(exper, experFile)
    # saveVariableDescription(logVariables, descriptionFile)
    # copyDataToremoteServer(simRepo, experFile)
    # copyDataToremoteServer(simRepo, descriptionFile)
    # saveSampleToTxtFile(exper,fileName = experFile)

    # sGroup = [31]

    # runSample(sampleDictList=exper,dFolder=dataFolder, remoteRepo = simRepo, sampleGroup=sGroup)
    # runSampleFrom(sampleDictList=exper,dFolder=dataFolder, remoteRepo = simRepo, fromSample = 80)

if __name__=='__main__':
    main()

