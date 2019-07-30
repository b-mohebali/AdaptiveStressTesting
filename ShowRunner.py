#! /usr/bin/python3

from yamlParseObjects.yamlObjects import *
import logging 
import os, sys
import subprocess
from profileExample.profileBuilder import * 
import platform
from eventManager.eventsLogger import * 
import csv


simConfig = simulationConfig('simulation.yaml')
print(simConfig.name)
for p in simConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup

#-----------------------------------------------------------------------------------------


variables = getAllVariableConfigs('variables.yaml')
for v in variables: 
    print(f'Variable: {v.name}, mapped name: {v.mappedName}')
logFile = getLoggerFileAddress(fileName='MyLoggerFile')

logging.basicConfig(filename=logFile, filemode='w', 
                    level = logging.DEBUG,
                    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

logging.debug('This is the debug message from the CAPS machine...')

buildSampleProfile(fileName='sampleProfile.csv')
createMappingFile(variables = variables,fileName='mapping', profileFileName='sampleProfile')

# buildInitialCsv(variables,simConfig, fileName ='sampleProfile')


class PGM_control(Control):
    NAME = 'PGM_control'

    def __init__(self, ctrl_str, controls_dir):
        super().__init__(ctrl_str, controls_dir)
        logging.info('Instantiating the PGM control')
        self.start_file_name = None
        self.ctrl_str        = ctrl_str
        self.start_file      = None
        self.controls_dir    = controls_dir

        self.simulation = self.pull_case(f'{case_Setup.CEF_BASE_DIR}/MVDC_SPS/RTDS_V5.007/fileman/PGM_SampleSystem/V2/')
        self.dft_file = self.simulation.dft_file
        self.simulation.set_run_function('start_case()')
        #self.simulation.set_run_script('Start_Case.scr')
        
        self.simulation.set_int_control(internal_ctrl = True)


myControl = PGM_control('', './')
controlsToRun = [myControl]

testDropLoc = Trial.init_test_drop(myControl.NAME)
ctrl = myControl
ctrl.initialize()
trial = Trial(ctrl, ctrl.simulation, testDropLoc)
# HACK. This checks if it has to do fm metrics. 
case_Setup.fm = False 
trial.run()
