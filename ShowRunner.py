
from yamlParseObjects.yamlObjects import *
import logging 
import os, sys
import subprocess
from profileExample.profileBuilder import * 
import platform

simConfig = simulationConfig('simulation.yaml')
print(simConfig.name)
for p in simConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup

variables = getAllVariableConfigs('variables.yaml')
for v in variables: 
    print(v.name)
logFile = getLoggerFileAddress(fileName='MyLoggerFile')
print(logFile)

logging.basicConfig(filename=logFile, filemode='w', 
                    level = logging.DEBUG,
                    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

logging.debug('This is the debug message from the CAPS machine...')

buildCsvProfile(fileLoc='profileExample', fileName='NewProfileExample')

# class PGM_control(Control):
#     NAME = 'PGM_control'

#     def __init__(self, ctrl_str, controls_dir):
#         super().__init__(ctrl_str, controls_dir)
#         logging.info('Instantiating the PGM control')
#         self.start_file_name = None
#         self.ctrl_str        = ctrl_str
#         self.start_file      = None
#         self.controls_dir    = controls_dir

#         self.simulation = self.pull_case(f'{case_Setup.CEF_BASE_DIR}/MVDC_SPS/RTDS_V5.007/fileman/PGM_SampleSystem/V2/')
#         self.dft_file = self.simulation.dft_file
#         self.simulation.set_run_function('start_case()')
#         #self.simulation.set_run_script('Start_Case.scr')
        
#         self.simulation.set_int_control(internal_ctrl = True)


# myControl = PGM_control('', './')
# controlsToRun = [myControl]

# testDropLoc = Trial.init_test_drop(myControl.NAME)
# ctrl = myControl
# ctrl.initialize()
# trial = Trial(ctrl, ctrl.simulation, testDropLoc)
# trial.run()
