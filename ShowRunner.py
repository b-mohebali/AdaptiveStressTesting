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

simConfig = simulationConfig('./yamlFiles/simulation.yaml')
print(simConfig.name)
for p in simConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup
from rscad import rtds
from repositories import *
# -------------------------- File path definitions ---------------------------------------------------------------


#------------------------------- Setting up the variables -----------------------------------------------

variables = getAllVariableConfigs('./yamlFiles/variables.yaml', scalingScheme=Scale.LINEAR)
# variables = getAllVariableConfigs('variables_limited.yaml', scalingScheme=Scale.LOGARITHMIC)
for v in variables: 
    print(f'Variable: {v.name}, mapped name: {v.mappedName}, Initial value: {v.initialState}')
print('---------------------------------------------------------------')
logFile = getLoggerFileAddress(fileName='MyLoggerFile')

logging.basicConfig(filename=logFile, filemode='w', 
                    level = logging.DEBUG,
                    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

logging.debug('This is the debug message from the CAPS machine...')

# buildSampleProfile(fileName='sampleProfile.csv')

# # Excluding the time independent variables from the scenario.
# scenarioVariables = [v for v in variables if v.varType.lower() != 'timeindep']
# createMappingFile(variables = scenarioVariables,fileName='mapping', profileFileName='sampleProfile')

# myEvent = VariableChangeSameTime(variables = variables[:2], simConfig = simConfig, startPoint=30, length = 15)
# print(myEvent)
# myEvent.updateCsv('sampleProfile.csv')

# print('-----------------------------')
# with open('sampleProfile.csv', 'r', newline='') as csvFile:
#     csvReader = csv.reader(csvFile)
#     firstRow = next(csvReader)# repo 10 for FFD with larger limits (50%) and the new scenario (just high load with a fixed length)
# print(firstRow)


# This is added from my cubicle machine.
class SaveType(Enum):
    SAVE_ALL = 1
    SAVE_ONE = 2

class PGM_control(Control):
    NAME = 'PGM_control'

    def __init__(self, ctrl_str, controls_dir):
        super().__init__(ctrl_str, controls_dir)
        logging.info('Instantiating the PGM control')
        self.start_file_name = None
        self.ctrl_str        = ctrl_str
        self.start_file      = None
        self.controls_dir    = controls_dir
        print(case_Setup.CEF_BASE_DIR)
        self.folder = f'{case_Setup.CEF_BASE_DIR}/MVDC_SPS/RTDS_V5.007/fileman/PGM_SampleSystem/V4Backup'
        self.simulation = self.pull_case(self.folder+'/')
        self.dft_file = self.simulation.dft_file
        # self.simulation.set_run_function('start_case()')
        self.simulation.set_run_script('Start_Case.scr')
        
        self.rtds_sys = rtds.RtdsSystem.from_dft(self.dft_file.str())
        self.simulation.set_int_control(internal_ctrl = True)
    
    def getAllVars(self):
        print('---------------')
        for dv in self.rtds_sys.get_draftvars():
            if dv['Name'] == 'myVar':  
                print('Value before change: ', dv['Value'])
                dv['Value'] = 2.47
                # outfile = self.dft_file.str()
                outfile = f'{self.folder}/PGM_V3.dft'
                self.rtds_sys.save_dft(fpath = outfile)
                print('Value after change: ', dv.attrs['Value'])
                dv.modified=True
        for sv in self.rtds_sys.get_sliders():
            if sv['Name']=='mySlider':
                print('Value before change: ', sv['Init'])
                sv['Init'] = 124
                outfile = f'{self.folder}/PGM_V3.dft'
                self.rtds_sys.save_dft(fpath = outfile)
                print('Value after change: ', sv.attrs['Init'])
                sv.modified=True
        print('---------------')
        return

    def _setVariableValues(self, randValues):
        print('------------------------------------------------------------')
        print('Setting draft variables to specific values within their range')
        draftVars = self.rtds_sys.get_draftvars()
        sliders = self.rtds_sys.get_sliders()
        for dftVar in draftVars:
            if dftVar['Name'] in randValues:
                print(f'Variable name: {dftVar["Name"]}, value before change: {dftVar["Value"]}, Random value: {randValues[dftVar["Name"]]}')
                dftVar['Value'] = randValues[dftVar['Name']]
        for sldr in sliders:
            if sldr['Name'] in randValues:
                print(f'Slider name: {sldr["Name"]}, value before change: {sldr["Init"]}, Random value: {randValues[sldr["Name"]]}')
                sldr['Init'] = randValues[sldr['Name']]
                
        outfile = f'/home/caps/.wine/drive_c/testing/fileman/PGM_V3.dft'
        self.rtds_sys.save_dft(fpath = outfile)    
        # Waiting for the draft file to be saved:
        print('------------------------------------------------------------')

    # This function will set all the variables to their nominal values.
    def setVariablesToInitialState(self, variables):
        inits = getVariablesInitialValueDict(variables)
        self._setVariableValues(inits)
        return 
    
    def setVariablesToRandom(self, variables, variableFile = SaveType.SAVE_ALL):
        timeIndepVars = getTimeIndepVarsDict(variables)
        randVars = randomizeVariables(timeIndepVars)
        os.chdir(currentDir)
        os.remove('variableValues.yaml')
        saveVariableValues(randVars, 'variableValues.yaml')
        self._setVariableValues(randVars)
        return 
    
    # This function gets the already randomized list of the variables
    # and their values.
    def setVariables(self, randVars, variableFile = SaveType.SAVE_ALL):
        os.chdir(currentDir)
        os.remove('variableValues.yaml')
        saveVariableValues(randVars, 'variableValues.yaml')
        self._setVariableValues(randVars)
        return 
        
# ------------------------------------------------------------


#-------------------------------------------------------------
def runSample(sampleDictList, dFolder, dRepo, remoteRepo = None, sampleGroup = None):
    experimentCounter = 1
    emptyFolder(dRepo)
    indexGroup = range(1, len(sampleDictList)+1)
    if sampleGroup is not None: 
        indexGroup = sampleGroup
    for sampleIndex in indexGroup:
        sample = sampleDictList[sampleIndex - 1]
        myControl = PGM_control('', './')   
        myControl.setVariables(sample)
        testDropLoc = Trial.init_test_drop(myControl.NAME)
        ctrl = myControl
        ctrl.initialize()
        trial = Trial(ctrl, ctrl.simulation, testDropLoc)
        # # HACK. This checks if it has to do fm metrics. 
        case_Setup.fm = False 
        trial.runWithoutMetrics()
        ### This is where the output is copied to a new location. 
        # newF = createNewDatafolder(dRepo)
        newF = createSpecificDataFolder(dRepo, sampleIndex)
        shutil.copyfile(f"{currentDir}/variableValues.yaml", f'{newF.rstrip("/")}/variableValues.yaml')
        copyDataToNewLocation(newF, dFolder)
        if remoteRepo is not None:
            copyDataToremoteServer(remoteRepo, newF)
            removeExtraFolders(dRepo,3)
        print('removed the extra folders from the source repository.')
        print(f'Done with the experiment {experimentCounter} and copying files to the repository.')
        experimentCounter+=1  
    return
 

def runSampleFrom(sampleDictList, dFolder, dRepo, remoteRepo = None, fromSample = None):
    N = len(sampleDictList)
    if fromSample is not None:
        sampleGroup = range(fromSample, N+1)
    else:
        sampleGroup = range(N)
    print('Starting sample: ')
    print(sampleDictList[fromSample-1])
    runSample(sampleDictList, dFolder, dRepo, remoteRepo = remoteRepo, sampleGroup=sampleGroup)
    return

#----------------------------------------------------------------
# First order Sensitivity Analysis:

# samplesNum = 960
# subInters = 16
# varDict = getTimeIndepVarsDict(variables)
# randList = randomizeVariablesList(varDict, samplesNum, subInters,scalingScheme=Scale.LOGARITHMIC, saveHists=True)
# saveSampleToTxtFile(randList, './experiments/limitedVarianceBased.txt')
# runSampleFrom(sampleDictList= randList, dFolder = dataFolder, dRepo = dataRepo, remoteRepo=testRepo, fromSample=960)


#------------------------------------------------------------------
# One at a time experiment design sensitivity analysis (Standard):

# timeIndepVars = getTimeIndepVarsDict(variables)
# randList = standardOATSampleGenerator(timeIndepVars)


# ----------------------------------------------------------------------
# The Fractional Factorial Desing with Hadamard matrices:
timeIndepVars = getTimeIndepVars(variables)
exper = fractionalFactorialExperiment(timeIndepVars, res4 = True)
saveSampleToTxtFile(exper, './experiments/FracFactEx.txt')
runSample(sampleDictList=exper,dFolder = dataFolder, dRepo = dataRepo, remoteRepo = remoteRepo1)

# ----------------------------------------------------------------------
# Returning all the variables to their standard value:
# myControl = PGM_control('', './')   
# myControl.setVariablesToInitialState(variables)


#---------------------------------------------------------------------
# Verification sample:

# timeIndepVars = getTimeIndepVars(variables)
# exper = generateVerifSample(timeIndepVars)
# saveSampleToTxtFile(exper, './experiments/VerifSample.txt')
# runSample(sampleDictList=exper,dFolder=dataFolder, dRepo = dataRepo, remoteRepo = remoteRepo14)

