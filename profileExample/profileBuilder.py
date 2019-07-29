import csv
from math import sin,cos
from yamlParseObjects.yamlObjects import *
from abc import ABC, abstractmethod
import logging
import os

# This function creates a sample scenario with two sine waves. 
def buildSampleProfile(fileLoc = '.', fileName = 'sample'):
    fileNameExt = fileLoc + '/' + fileNamePlusExt(fileName, '.csv')
    print(f'The sample profile location: {os.path.abspath(fileNameExt)}')
    with open(fileNameExt, 'w', newline='') as csvFile: 
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['Time','var1','var2'])
        t = 0.0
        tStep = 0.01
        while t < 210.0 + tStep:
            data = [float("{0:.6f}".format(t)), 
                    float("{0:.6f}".format(1+0.5*sin(t*3))),  
                    float("{0:.6f}".format(7 + 0.5* cos(t)))]
            csvWriter.writerow(data)
            t += tStep
    logging.debug(f'The sample profile is created in {os.path.abspath(fileNameExt)}')
    return 



def buildInitialCsv(variables, simConfig, fileLoc='.', fileName = 'profile'):
    varNames = ['Time'] + [var.name for var in variables]
    fileNameExt = fileLoc + '/' + fileNamePlusExt(fileName, '.csv')
    with open(fileNameExt, 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        logging.debug(f'Creating the initial CSV scenario file on {os.path.abspath(fileNameExt)}')
        logging.debug('These are the vatiables: ' + varNames.__str__())
        csvwriter.writerow(varNames)
        initialDataRow = [0.0] + [var.initialState for var in variables]
        # initialDataRow.insert(0,0.0) # Added for the initial time.
        csvwriter.writerow(initialDataRow)

        # Inserting the data point for the start of the scenario:
        scenarioStartPoint = [simConfig.eventWindowStart] + [var.initialState for var in variables]
        csvwriter.writerow(scenarioStartPoint)
    return 

def fileNamePlusExt(fileName, ext):
    return fileName + (ext if not fileName.endswith(ext) else '')

    
def createMappingFile(variables, fileLoc = '.', fileName = 'mapping', profileFileName = 'sample'):
    fileNameExt = fileLoc + '/' + fileName + '.csv'
    profFile = profileFileName if profileFileName.endswith('.csv') else profileFileName + '.csv'
    print(f'Mapping file name: {os.path.abspath(fileNameExt)}')
    logging.debug(f'Mapping file name: {fileNameExt}')
    with open(fileNameExt, 'w', newline='') as csvFile: 
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow([profFile,'GTNETSKT1_from.txt','GTNETSKT1_to.txt'])
        csvWriter.writerow(['Time','',''])
        for var in variables:
            row = [var.name, var.mappedName,'']
            csvWriter.writerow(row)
    return 

class ScenarioBuilder:
    
    def __init__(self, simConfig,variables):
        self.timeStep = simConfig.timeStep
        self.vars = variables
    
    ## TODO
    def randomizeVariables(self):
        pass


'''
    This is the part that implements the concent of event in this
    context. The specific events are defined as children of the 
    'abstract' event. 
'''
class Event():

    def __init__(self):
        self.duration = 0
    
    @abstractmethod
    def randomize(self):
        pass
    
    


        
