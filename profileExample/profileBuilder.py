import csv
from math import sin,cos
from yamlParseObjects.yamlObjects import *
from yamlParseObjects.variablesUtil import *

from abc import ABC, abstractmethod, ABCMeta
import logging
import os
import numpy as np
from random import * 


class WrongConfigObject(Exception):
    pass

class WronfFileType(Exception):
    pass

def getCsvHeaders(csvFileAddress):
    with open(csvFileAddress, 'r') as csvFile:
        csvReader = csv.reader(csvFile)
        firstRow = next(csvReader)
    return firstRow


# This function creates a sample scenario with two sine waves. 
def buildSampleProfile(fileLoc = '.', fileName = 'sample'):
    fileNameExt = fileLoc + '/' + fileNamePlusExt(fileName, '.csv')
    print(f'The sample profile location: {os.path.abspath(fileNameExt)}')
    with open(fileNameExt, 'w', newline='') as csvFile: 
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['Time','var1','var2', 'loadArate', 'loadBrate'])
        t = 0.0
        tStep = 0.01
        while t < 210.0 + tStep:
            data = [float("{0:.10f}".format(t)), 
                    float("{0:.10f}".format(7+1.0*sin(t*4))),  
                    float("{0:.10f}".format(3 + 2* cos(t))),
                    4.5,
                    4.5]
            csvWriter.writerow(data)
            t += tStep
    logging.info(f'The sample profile is created in {os.path.abspath(fileNameExt)}')
    return 



def buildInitialCsv(variables, simConfig, fileLoc='.', fileName = 'profile'):
    varNames = ['Time'] + [var.name for var in variables]
    fileNameExt = fileLoc + '/' + fileNamePlusExt(fileName, '.csv')
    with open(fileNameExt, 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        logging.info(f'Creating the initial CSV scenario file on {os.path.abspath(fileNameExt)}')
        logging.info('These are the vatiables: ' + varNames.__str__())
        csvwriter.writerow(varNames)
        initialDataRow = [0.0] + [var.initialState for var in variables]
        # initialDataRow.insert(0,0.0) # Added for the initial time.
        csvwriter.writerow(initialDataRow)

        # Inserting the data point for the start of the scenario:
        scenarioStartPoint = [simConfig.eventWindowStart] + [var.initialState for var in variables]
        csvwriter.writerow(scenarioStartPoint)
    return 


# This is a generic scenario for sensitivity analysis of PGM. Just a ramp up and ramp down of the loads
def buildGenericScenarioCsv(variables, simConfig, fileLoc='.', fileName = 'profile'):
    timeDepVars = getTimeDepVariables(variables)
    varNames = ['Time'] + [var.mappedName for var in timeDepVars]
    fileNameExt = fileLoc + '/' + fileNamePlusExt(fileName, '.csv')
    with open(fileNameExt, 'w', newline='') as csvFile:
        csvwriter = csv.writer(csvFile)
        logging.info(f'Creating the initial CSV scenario file on {os.path.abspath(fileNameExt)}')
        logging.info('These are the vatiables: ' + varNames.__str__())
        csvwriter.writerow(varNames)
        initialRow = [0.0] + [var.initialState for var in timeDepVars]
        csvwriter.writerow(initialRow)
        data = [35.0, 15.0, 15.0,4.5, 4.5]
        csvwriter.writerow(data)
        data = [65.0, 0.1, 0.1,4.5, 4.5]
        csvwriter.writerow(data)

        
        


def fileNamePlusExt(fileName, ext):
    return fileName + (ext if not fileName.endswith(ext) else '')

    
# This is written according to the format that the CEF will accept.
def createMappingFile(variables, fileLoc = '.', fileName = 'mapping', profileFileName = 'sample'):
    fileNameExt = fileLoc + '/' + fileName + '.csv'
    profFile = profileFileName if profileFileName.endswith('.csv') else profileFileName + '.csv'
    print(f'Mapping file name: {os.path.abspath(fileNameExt)}')
    logging.info(f'Mapping file name: {fileNameExt}')
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
class Event(ABC):

    def __init__(self, variables, simConfig):
        # Checking the type of the variables config list and the simulation config object.
        if not isinstance(variables, list) or not isinstance(variables[0], variableConfig):
            errMessage = 'The variable configuration list does not match expected format.'
            logging.error(errMessage)
            raise WrongConfigObject(errMessage)
        if not isinstance(simConfig, simulationConfig):
            errMessage = 'The simulation configuration list does not match expected format.'
            logging.error(errMessage)
            raise WrongConfigObject(errMessage)

        # Constructing a dictionary that has the variables config keyed with their names.     
        self.varDict = getVariablesDict(variables)
        self.varNames = [var.name for var in variables]

        self.duration = 0
        self.name = 'AbsractEvent'
        self.variables = variables

    
    def setCsvFile(self, csvFileAddress):
        if not csvFileAddress.endswith('.csv'):
            errMessage= f'CSV file passed to {self.name} is missing or has the wrong type.'
            logging.error(errMessage)
            raise WronfFileType(errMessage)
        self.csvFile = csvFileAddress
        logging.info(f'{self.name} got the CSV file : {self.csvFile}')
        return 

               
    @abstractmethod
    def randomizeTime(self):
        pass

    @abstractmethod
    def __str__(self):
        return ""


class VariableChangeSameTime(Event):
    
    # The default length of the event window is set to be 30 seconds
    def __init__(self, variables, simConfig, startPoint, length):
        super().__init__(variables, simConfig)
        self.eventWindow = length
        self.name = 'Variables change at the same time'
        self.startPoint = startPoint
        self.endPoint = self.startPoint + length
        self.simConfig = simConfig
        self.createTimeVector()
        self.changeTimeStep = None
        self.changeTime = None
        self.randomizeTime()
        

    def __str__(self):
        descriptor = f'''Event:  {self.name}
        variables: {self.varNames}
        Event time: {'Not specified yet' if self.changeTime is None else self.changeTime}
        Event window: {self.startPoint,self.endPoint}
        '''
        return descriptor.__str__()


    def createTimeVector(self):
        steps = int(self.eventWindow / self.simConfig.timeStep)
        self.timeVec = np.linspace(start = self.startPoint, stop = self.endPoint, num=steps, endpoint=True)
        return 

    def randomizeTime(self):
        self.changeTimeStep = np.random.choice([_ for _ in range(len(self.timeVec))])
        self.changeTime = self.timeVec[self.changeTimeStep]
        return

    # TODO: complete the procedure for updateing the profile CSV.
    def updateCsv(self, csvFileAddress):
        allVars = getCsvHeaders(csvFileAddress)
        print(f'Variables: {allVars}')
        
        pass
     
    
    def applyChange(self):
        pass
    





    


        
