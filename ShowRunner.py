from yamlParseObjects.yamlObjects import *
import logging 
from eventManager.eventsLogger import * 
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

dirPath = os.getcwd()
logFile = getLoggerFileAddress(fileName='MyLoggerFile')
print(logFile)

logging.basicConfig(filename=logFile, filemode='w', 
                    level = logging.DEBUG,
                    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

logging.debug('This is the debug message from the CAPS machine...')

print(platform.system())