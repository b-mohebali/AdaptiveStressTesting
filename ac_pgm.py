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

folder = f'{case_Setup.CEF_BASE_DIR}/{simConfig.modelLocation}'
print(folder)

class AC_PGM(Control):
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
    

myAcPgm = AC_PGM(configFile=simConfig, start_scr='Test_logger.scr')
myAcPgm.initialize()
testDropLoc = Trial.init_test_drop(myAcPgm.NAME)
myTrial = Trial(myAcPgm, myAcPgm.simulation, testDropLoc)
case_Setup.fm = False
myTrial.runWithoutMetrics()
