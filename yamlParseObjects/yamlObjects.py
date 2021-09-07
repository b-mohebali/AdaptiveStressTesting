import yaml
import math
import platform
from enum import Enum
from datetime import datetime 
import time 
from typing import List

class Scale(Enum):
    LINEAR = 1
    LOGARITHMIC = 2

variableSpan = 0.5
variableScale = 6

class simulationConfig():
    def __init__(self, yamlFileAddress):
        self.fileLocation = yamlFileAddress
        with open(yamlFileAddress,'rt') as fp:
            yamlString = fp.read()
        fp.close()
        self.yamlObj= yaml.load(yamlString, Loader = yaml.SafeLoader)
        self.name = self.yamlObj['name']
        self.eventWindowStart = self.yamlObj['eventWindowStart'] if 'eventWindowStart' in self.yamlObj else 0
        self.eventWindowEnd = self.yamlObj['eventWindowEnd'] if 'eventWindowEnd' in self.yamlObj else 0
        self.description = self.yamlObj['description'] if 'description' in self.yamlObj else None
        self.timeStep = self.yamlObj['timeStep'] if 'timeStep' in self.yamlObj else 0.05
        # This way we can have different settings for different OS platforms.
        codeBaseName = 'codeBase_' + platform.system()
        matlabPathName = 'matlab_path_' + platform.system()
        self.platform = platform.system()
        self.codeBase = self.yamlObj[codeBaseName] if codeBaseName in self.yamlObj else []
        self.matlabPaths = self.yamlObj[matlabPathName] if matlabPathName in self.yamlObj else []
        self.profileLoc = self.yamlObj['profileLoc'] if 'profileLoc' in self.yamlObj else '.'
        self.simLength = self.yamlObj['length'] if 'length' in self.yamlObj else self.eventWindowEnd
        self.modelName = self.yamlObj['modelName']
        self.sampleRepo = self.yamlObj['sampleRepo'] if 'sampleRepo' in self.yamlObj else None
        self.modelLocation = self.yamlObj['modelLocation']
        self.batchSize = self.yamlObj['batchSize'] if 'batchSize' in self.yamlObj else 1
        self.sampleBudget = self._getNecessaryProperty('sampleBudget')
        self.initialSampleSize = self._getNecessaryProperty('initialSampleSize')
        self.batchSize = self._getNecessaryProperty('batchSize')
        self.outputFolder = self._getNecessaryProperty('outputFolder')
        self.metricNames = self.yamlObj['metricNames'] if 'metricNames' in self.yamlObj else []
        self.exploreBatchSize = self._getProperty('exploreBatch', 0)
        self.resourceLambda = self._getProperty('resourceLambda', 1.0)
        self.resourceEpsilon = self._getProperty('resourceEpsilon', 0.2)

    def _getNecessaryProperty(self,propName):
        if propName not in self.yamlObj:
            raise ValueError(f'The {propName} is not specified for the process.')
        return self.yamlObj[propName]
    
    def _getProperty(self, propName, defaultValue):
        return self.yamlObj[propName] if propName in self.yamlObj else defaultValue
        
    def __str__(self):
        nl = '\n \t\t\t\t\t'
        descriptor = f'''Simulation name: {self.name}
            Model location: {self.modelLocation}
            Model name: {self.modelName} 
            code base location: {nl}{nl.join(self.codeBase)}
            Platfor: {self.platform}'''
        return descriptor.__str__()

class VariableConfig():

    def __init__(self, name, initial, varType, lowerLimit, upperLimit, mappedName, description, risingRateLimit = float('inf'), fallingRateLimit = float('inf')):
        self.name = name
        self.initialState = initial
        self.varType = varType
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.risingRateLimit = risingRateLimit
        self.fallingRateLimit = fallingRateLimit
        self.mappedName = mappedName if mappedName is not None else name
        self.description = description

    def __str__(self):
        descriptor = f'''Variable name: {self.name}
        type: {self.varType}
        Value: {self.initialState}
        Value range: [{self.lowerLimit:0.6f}, {self.upperLimit:0.6f}]
        '''
        return descriptor.__str__()

def getAllVariableConfigs(yamlFileAddress, span:float = variableSpan, scalingScheme = Scale.LINEAR, scale:float = variableScale):
    with open(yamlFileAddress,'rt') as fp:
        yamlString = fp.read()
    fp.close()
    varList = []
    for yamlVar in yaml.load_all(yamlString, Loader = yaml.SafeLoader):
        name = yamlVar['name']
        t = yamlVar['type']
        initialState = yamlVar['initialState']
        if scalingScheme==Scale.LINEAR:
            lowerLimit = yamlVar['lowerLimit'] if 'lowerLimit' in yamlVar else initialState*(1 - span) 
            upperLimit = yamlVar['upperLimit'] if 'upperLimit' in yamlVar else initialState*(1 + span)
        elif scalingScheme == Scale.LOGARITHMIC:
            scales = [initialState / scale, initialState * scale]
            lowerLimit = yamlVar['lowerLimit'] if 'lowerLimit' in yamlVar else min(scales)
            upperLimit = yamlVar['upperLimit'] if 'upperLimit' in yamlVar else max(scales) 
        mappedName = yamlVar['mappedName'] if 'mappedName' in yamlVar else name
        desc  = yamlVar['description'] if 'description' in yamlVar else name
        variableCon = VariableConfig(name = name, 
                                    initial=initialState, 
                                    varType=t,
                                    mappedName = mappedName,
                                    lowerLimit=lowerLimit, 
                                    upperLimit=upperLimit,
                                    description= desc)
        varList.append(variableCon)
    return varList
    

class FinalReport():
    def __init__(self, yamlFile):
        with open(yamlFile, 'rt') as yf:
            report = yaml.load(yf, Loader = yaml.FullLoader)
        self.elapsed_time = report['elapsed_time_sec']
        self.label = report['result_label']
        self.variables = report['variables']


class ResourceAllocationReport:

    def __init__(self, iterNum = None,
                exploreTendency = None,
                exploitTendency = None,
                R_explore = None,
                R_exploit = None,
                exploreBudget = None,
                exploitBudget = None):
        self.iteration_number = iterNum
        self.exploration_tendency = exploreTendency
        self.exploitation_tendency = exploitTendency
        self.R_explore = R_explore,
        self.R_exploit = R_exploit
        self.calculated_exploration_budget = exploreBudget
        self.calculated_exploitation_budget = exploitBudget
        self.exploitBounds = None 
        self.exploreBounds = None
        

class IterationReport():
    """
        TODO: The iteration report. Will be saved in a single file and contains information about how the iteration went. 
            The final report is a list of yaml objects one for each iteration. 

        NOTE: The starting time is counted from the moment the iteration report object is instantiated.
    """
    def __init__(self, varNames, 
                batchSize = 1, 
                iterationNumber = 1,
                startTime = datetime.now()):
        self.iterationNumber = iterationNumber
        self.batchSize = batchSize
        self.startTime = startTime
        self.stopTime = None
        self.budgetRemaining = 0
        self.changeMeasure = None
        self.variableNames = varNames

    def setStart(self):
        self.__start = time.time()
        self.startTime=datetime.now()
    def setStop(self):
        self.elapsedTimeSeconds = time.time() - self.__start
        self.stopTime = datetime.now()
        del self.__start
    
    # The list comprehension makes a dictionary of dictionaries based on a list of lists.
    #   The keys for the first dict is the index of the samples starting from 1
    #   The keys for the second dict is the names of the variables so that the 
    #       variable values are distinguishable in the report.
    def setSamples(self,samples):
        self.samples = dict((i1+1, dict((self.variableNames[_], float(sample[_])) for _ in range(len(sample)))) for i1, sample in enumerate(samples))
    
    def setExploitatives(self,exploiters):
        self.Exploitation = dict((i1+1, dict((self.variableNames[_], float(sample[_])) for _ in range(len(sample)))) for i1, sample in enumerate(exploiters))
    def setExplorers(self,explorers):
        self.Exploration = dict((i1+1, dict((self.variableNames[_], float(sample[_])) for _ in range(len(sample)))) for i1, sample in enumerate(explorers))
    

    
    def setChangeMeasure(self, changeMeasure):
        self.changeMeasure = float(changeMeasure)
    # Making a dictionary with sample index as key and its corresponding label as values.
    # NOTE: The keys' indexing starts from 1.
    def setMetricResults(self, metricResults):
        self.metricResults = dict((_+1,int(metricResults[_])) for _ in range(len(metricResults)))
    def setAccuracy(self, acc):
        self.accuracy = acc


def saveIterationReport(reports, yamlFileLoc):
    print(yamlFileLoc)
    with open(yamlFileLoc, 'w') as yamlFile:
        yaml.dump_all(reports, yamlFile)
    