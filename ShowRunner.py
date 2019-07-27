from yamlParseObjects.yamlObjects import *
import logging 
from eventManager.eventsLogger import * 
import os, sys
import subprocess
from profileExample.profileBuilder import * 
simConfig = simulationConfig('simulation.yaml')
print(simConfig.name)
for p in simConfig.codeBase: sys.path.insert(0,p)

#import simulation


dirPath = os.getcwd()
logFile = getLoggerFileAddress(fileName='MyLoggerFile')
print(logFile)

logging.basicConfig(filename=logFile, filemode='w', 
                    level = logging.INFO,
                    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

logging.debug('This is the debug message')

scriptName = 'test'
cmd = ['py', 'C:/Users/bm12m/Google Drive/codes/testing/'+f'{scriptName}.py']
print(cmd)
logCommand(cmd)
subprocess.call(cmd)
buildCsvProfile(simConfig.profileLoc)

# just a change I want to incorporate.
