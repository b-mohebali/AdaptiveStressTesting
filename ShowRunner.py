from yamlParseObjects.yamlObjects import *
import logging 
from eventManager.eventsLogger import * 
import os

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
