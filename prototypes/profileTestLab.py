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
import matplotlib.pyplot as plt
simConfig = simulationConfig('simulation.yaml')
print(simConfig.name)
for p in simConfig.codeBase: 
    sys.path.insert(0,p)
    print(p + ' is added to the path')

from autoRTDS import Trial
from controls import Control, InternalControl
import case_Setup
from rscad import rtds
#--------------------------------------------------------------
currentDir = os.getcwd()


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

myEvent = VariableChangeSameTime(variables = variables[:2], simConfig = simConfig, startPoint=30, length = 15)
print(myEvent)
myEvent.updateCsv('sampleProfile.csv')

print('-----------------------------')
with open('sampleProfile.csv', 'r', newline='') as csvFile:
    csvReader = csv.reader(csvFile)
    firstRow = next(csvReader)
print(firstRow)



