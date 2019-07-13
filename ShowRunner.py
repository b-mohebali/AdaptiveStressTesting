from yamlParseObjects.yamlObjects import *
import logging 
from eventManager.eventsLogger import * 
import os, sys
from Codes import *

simConfig = simulationConfig('simulation.yaml')
print(simConfig.name)

dirPath = os.getcwd()
print(dirPath)
print(os.path.isdir(dirPath + "\log"))
logFile = getLoggerFileAddress(fileName='MyLoggerFile')
print(logFile)

logging.basicConfig(filename=logFile, filemode='w', 
                    level = logging.DEBUG,
                    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s')

logging.debug('This is the debug message')
print('This is added from the CAPS computer')

##
for p in simConfig.codeBase:
    sys.path.insert(0,p)

for p in sys.path:
    print(p)
sys.path.append('../')