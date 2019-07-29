import yaml
import math
import platform
class simulationConfig():
    def __init__(self, yamlFileAddress):
        with open(yamlFileAddress,'rt') as fp:
            yamlString = fp.read()
        fp.close()
        yamlObj= yaml.load(yamlString, Loader = yaml.SafeLoader)
        self.name = yamlObj['name']
        self.length = yamlObj['length']
        self.eventWindowStart = yamlObj['eventWindowStart']
        self.eventWindowEnd = yamlObj['eventWindowEnd']
        self.description = yamlObj['description'] if 'description' in yamlObj else None
        self.timeStep = yamlObj['timeStep'] if 'timeStep' in yamlObj else 0.05
        codeBaseName = 'codeBase_' + platform.system() 
        self.codeBase = yamlObj[codeBaseName] if codeBaseName in yamlObj else []
        self.profileLoc = yamlObj['profileLoc'] if 'profileLoc' in yamlObj else '.'

class variableConfig():

    def __init__(self, name, initial, varType, lowerLimit, upperLimit, mappedName, risingRateLimit = float('inf'), fallingRateLimit = float('inf')):
        self.name = name
        self.initialState = initial
        self.varType = varType
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.risingRateLimit = risingRateLimit
        self.fallingRateLimit = fallingRateLimit
        self.mappedName = mappedName if mappedName is not None else name

def getAllVariableConfigs(yamlFileAddress):
    with open(yamlFileAddress,'rt') as fp:
        yamlString = fp.read()
    fp.close()
    varList = []
    for yamlVar in yaml.load_all(yamlString, Loader = yaml.SafeLoader):
        name = yamlVar['name']
        t = yamlVar['type']
        initialState = yamlVar['initialState']
        lowerLimit = yamlVar['lowerLimit']
        upperLimit = yamlVar['upperLimit']
        mappedName = yamlVar['mappedName']
        variableCon = variableConfig(name = name, 
                                    initial=initialState, 
                                    varType=t,
                                    mappedName = mappedName,
                                    lowerLimit=lowerLimit, 
                                    upperLimit=upperLimit)
        varList.append(variableCon)
    return varList

            
        
