import yaml
from math import inf

class simulationConfig():
    def __init__(self, yamlFileAddress):
        with open(yamlFileAddress,'rt') as fp:
            yamlString = fp.read()
        fp.close()
        yamlObj= yaml.load(yamlString)
        self.name = yamlObj['name']
        self.length = yamlObj['length']
        self.eventWindowStart = yamlObj['eventWindowStart']
        self.eventWindowEnd = yamlObj['eventWindowEnd']
        self.description = yamlObj['description'] if 'description' in yamlObj else None
        self.timeStep = yamlObj['timeStep'] if 'timeStep' in yamlObj else 0.05

class variableConfig():

    def __init__(self, name, initial, varType, lowerLimit, upperLimit, risingRateLimit = inf, fallingRateLimit = inf):
        self.name = name
        self.initialState = initial
        self.varType = varType
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.risingRateLimit = risingRateLimit
        self.fallingRateLimit = fallingRateLimit

def getAllVariableConfigs(yamlFileAddress):
    with open(yamlFileAddress,'rt') as fp:
        yamlString = fp.read()
    fp.close()
    varList = []
    for yamlVar in yaml.load_all(yamlString):
        name = yamlVar['name']
        t = yamlVar['type']
        initialState = yamlVar['initialState']
        lowerLimit = yamlVar['lowerLimit']
        upperLimit = yamlVar['upperLimit']
        variableCon = variableConfig(name = name, 
                                    initial=initialState, 
                                    varType=t,
                                    lowerLimit=lowerLimit, 
                                    upperLimit=upperLimit)
        varList.append(variableCon)
    return varList

            
        
