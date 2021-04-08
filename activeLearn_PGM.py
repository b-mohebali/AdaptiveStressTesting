#! usr/bin/python3

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
    print(p + ' has been added to the path.')

from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup
from rscad import rtds
from repositories import *
import simulation



"""
Steps of checks for correctness: 
    1- Run an FFD sample with the 4 variables in a new location with the control object
        from simulationHelper.py script.
    2- Implement the initial sampling using the combination of the control objects and 
        the developed active learning code. 
    3- Implement the exploitation part. Save the change measures in each step in case the 
        process is interrupted for any reason.

"""