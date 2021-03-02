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


simConfig = simulationConfig('./yamlFiles/ac_pgm_conf.yaml')
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
import glob
from modelDef import PGM_control
from ShowRunner import *

folder = f'{case_Setup.CEF_BASE_DIR}/{simConfig.modelLocation}'
print(folder)

class AC_PGM(PGM_control):
    NAME = 'AC_PGM'

    def __init__(self, configFile, start_scr='Start_Case.scr'):
        super().__init__('','./')
        self.start_file_name = None
        self.ctrl_str = ''
        self.start_file = None
        self.controls_dir = './'
        self.simConfig = configFile
        self.folder = f'{case_Setup.CEF_BASE_DIR}/{self.simConfig.modelLocation}'
        print(f'File folder: {self.folder}')
        self.simulation = self.pull_case(self.folder+'/')
        self.dft_file = self.simulation.dft_file
        self.simulation.set_run_script(start_scr)
        self.rtds_sys = rtds.RtdsSystem.from_dft(self.dft_file.str())
        self.simulation.set_int_control(internal_ctrl = True)
    
    def setVariables(self, randVars, variableFile = SaveType.SAVE_ALL):
        os.chdir(currentDir)
        os.remove('variableValues.yaml')
        saveVariableValues(randVars, 'variableValues.yaml')
        outfile = self._setVariableValues(randVars)
        return outfile

    def _setVariableValues(self, randValues, saveOriginal = False):
        print('------------------------------------------------------------')
        print('Setting draft variables to specific values within their range')
        draftVars = self.rtds_sys.get_draftvars()
        sliders = self.rtds_sys.get_sliders()
        for dftVar in draftVars:
            if dftVar['Name'] in randValues:
                print(f'Variable name: {dftVar["Name"]}, value before change: {dftVar["Value"]}, Value after change: {randValues[dftVar["Name"]]}')
                dftVar['Value'] = randValues[dftVar['Name']]
        for sldr in sliders:
            if sldr['Name'] in randValues:
                print(f'Slider name: {sldr["Name"]}, value before change: {sldr["Init"]}, Value after change: {randValues[sldr["Name"]]}')
                sldr['Init'] = randValues[sldr['Name']]
        outfile = f'{case_Setup.DRIVE_C}/testing/fileman/{self.simConfig.modelName}.dft'
        self.rtds_sys.save_dft(fpath = outfile)
        if saveOriginal:
            self.rtds_sys.save_dft(fpath = f'{self.folder}/{self.simConfig.modelName}.dft')
        # Waiting for the draft file to be saved:
        print('------------------------------------------------------------')
        return outfile

    # This function will set all the variables to their nominal values.
    def setVariablesToInitialState(self, variables):
        inits = getVariablesInitialValueDict(variables)
        outfile = self._setVariableValues(inits, saveOriginal=True)
        return outfile



variablesFile = './yamlFiles/variables_ac_pgm.yaml'
variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR)

## Setting up the factor screening experiment: 
experFile = './experiments/FFD_AC_PGM.txt'
timeIndepVars = getTimeIndepVars(variables, shuffle = True, omitZero=True)
exper = fractionalFactorialExperiment(variables, res4=True)
saveSampleToTxtFile(exper, experFile)



# myAcPgm = AC_PGM(configFile=simConfig, start_scr='Test_logger.scr')
# myAcPgm.initialize()
# testDropLoc = Trial.init_test_drop(myAcPgm.NAME)
# myTrial = Trial(myAcPgm, myAcPgm.simulation, testDropLoc)
# case_Setup.fm = False
# myTrial.runWithoutMetrics()
