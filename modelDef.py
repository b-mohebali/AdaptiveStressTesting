import logging
import sys, os
from yamlParseObjects.

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


class PGM_control(Control):
    NAME = 'PGM_control'
    def __init__(self, ctrl_str, controls_dir, configFile):
        super().__init__(ctrl_str, controls_dir)
        logging.info('Instantiating the PGM control')
        self.start_file_name = None
        self.ctrl_str        = ctrl_str
        self.start_file      = None
        self.controls_dir    = controls_dir
        self.simConfig = configFile
        # self.folder = f'{case_Setup.CEF_BASE_DIR}/MVDC_SPS/RTDS_V5.007/fileman/PGM_SampleSystem/V4_filterRedesign'
        self.folder = f'{case_Setup.CEF_BASE_DIR}/{self.simConfig.modelLocation}'
        print(f'File folder: {self.folder}')
        self.simulation = self.pull_case(self.folder+'/')
        self.dft_file = self.simulation.dft_file
        self.simulation.set_run_script('Start_Case.scr')
        self.rtds_sys = rtds.RtdsSystem.from_dft(self.dft_file.str())
        self.simulation.set_int_control(internal_ctrl = True)
    
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
    
    def setVariablesToRandom(self, variables, variableFile = SaveType.SAVE_ALL):
        timeIndepVars = getTimeIndepVarsDict(variables)
        randVars = randomizeVariables(timeIndepVars)
        os.chdir(currentDir)
        os.remove('variableValues.yaml')
        saveVariableValues(randVars, 'variableValues.yaml')
        outfile = self._setVariableValues(randVars)
        return outfile
    
    # This function gets the already randomized list of the variables
    # and their values.
    def setVariables(self, randVars, variableFile = SaveType.SAVE_ALL):
        os.chdir(currentDir)
        os.remove('variableValues.yaml')
        saveVariableValues(randVars, 'variableValues.yaml')
        outfile = self._setVariableValues(randVars)
        return outfile
        