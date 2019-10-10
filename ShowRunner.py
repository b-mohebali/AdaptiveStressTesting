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
import numpy as np

simConfig = simulationConfig('simulation.yaml')
print(simConfig.name)
for p in simConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup
from rscad import rtds
#-----------------------------------------------------------------------------------------


variables = getAllVariableConfigs('variables.yaml')
for v in variables: 
    print(f'Variable: {v.name}, mapped name: {v.mappedName}')
print('---------------------------------------------------------------')
logFile = getLoggerFileAddress(fileName='MyLoggerFile')

logging.basicConfig(filename=logFile, filemode='w', 
                    level = logging.DEBUG,
                    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

logging.debug('This is the debug message from the CAPS machine...')

buildSampleProfile(fileName='sampleProfile.csv')

# Excluding the time independent variables from the scenario.
scenarioVariables = [v for v in variables if v.varType.lower() != 'timeindep']
createMappingFile(variables = scenarioVariables,fileName='mapping', profileFileName='sampleProfile')

# buildGenericScenarioCsv(scenarioVariables,simConfig, fileName ='sampleProfile')


myEvent = VariableChangeSameTime(variables = variables[:2], simConfig = simConfig, startPoint=30, length = 15)
print(myEvent)
myEvent.updateCsv('sampleProfile.csv')


print('-----------------------------')
with open('sampleProfile.csv', 'r', newline='') as csvFile:
    csvReader = csv.reader(csvFile)
    firstRow = next(csvReader)
print(firstRow)


# This is added from my cubicle machine.

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
        self.simulation = self.pull_case(f'{case_Setup.CEF_BASE_DIR}/MVDC_SPS/RTDS_V5.007/fileman/PGM_SampleSystem/V4/')
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
                outfile = '/home/caps/.wine/drive_c/cef/MVDC_SPS/RTDS_V5.007/fileman/PGM_SampleSystem/V4/PGM_V3.dft'
                self.rtds_sys.save_dft(fpath = outfile)
                print('Value after change: ', dv.attrs['Value'])
                dv.modified=True
        for sv in self.rtds_sys.get_sliders():
            if sv['Name']=='mySlider':
                print('Value before change: ', sv['Init'])
                sv['Init'] = 124
                outfile = '/home/caps/.wine/drive_c/cef/MVDC_SPS/RTDS_V5.007/fileman/PGM_SampleSystem/V4/PGM_V3.dft'
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
                
        outfile = '/home/caps/.wine/drive_c/cef/MVDC_SPS/RTDS_V5.007/fileman/PGM_SampleSystem/V4/PGM_V3.dft'
        self.rtds_sys.save_dft(fpath = outfile)            
        print('------------------------------------------------------------')

    def setVariablesToInitialState(self, variables):
        inits = getVariablesInitialValueDict(variables)
        self._setVariableValues(inits)
        return 
    
    def setVariablesToRandom(self, variables):
        timeIndepVars = getTimeIndepVarsDict(variables)
        randVars = randomizeVariables(timeIndepVars)
        self._setVariableValues(randVars)
        return 
    

# myControl = PGM_control('', './')
# controlsToRun = [myControl]
# myControl.setVariablesToRandom(variables)
# testDropLoc = Trial.init_test_drop(myControl.NAME)
# ctrl = myControl
# ctrl.initialize()
# trial = Trial(ctrl, ctrl.simulation, testDropLoc)
# # # HACK. This checks if it has to do fm metrics. 
# case_Setup.fm = False 
# print('This is the end of the script')
# draftVars = myControl.
# trial.run()
