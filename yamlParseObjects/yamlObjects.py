import yaml
import math
import platform
from enum import Enum


class Scale(Enum):
    LINEAR = 1
    LOGARITHMIC = 2


variableSpan = 0.3
variableScale = 6

class simulationConfig():
    def __init__(self, yamlFileAddress):
        with open(yamlFileAddress,'rt') as fp:
            yamlString = fp.read()
        fp.close()
        yamlObj= yaml.load(yamlString, Loader = yaml.SafeLoader)
        self.name = yamlObj['name']
        self.eventWindowStart = yamlObj['eventWindowStart']
        self.eventWindowEnd = yamlObj['eventWindowEnd']
        self.description = yamlObj['description'] if 'description' in yamlObj else None
        self.timeStep = yamlObj['timeStep'] if 'timeStep' in yamlObj else 0.05
        # This way we can have different settings for different OS platforms.
        codeBaseName = 'codeBase_' + platform.system()
        self.codeBase = yamlObj[codeBaseName] if codeBaseName in yamlObj else []
        self.profileLoc = yamlObj['profileLoc'] if 'profileLoc' in yamlObj else '.'
        self.simLength = yamlObj['length'] if 'length' in yamlObj else self.eventWindowEnd
                
        
class variableConfig():

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
        descroptor = f'''Variable name: {self.name}
        type: {self.varType}
        Value: {self.initialState}
        Value range: [{self.lowerLimit:0.6f}, {self.upperLimit:0.6f}]
        '''
        return descroptor.__str__()

def getAllVariableConfigs(yamlFileAddress, scalingScheme = Scale.LINEAR):
    with open(yamlFileAddress,'rt') as fp:
        yamlString = fp.read()
    fp.close()
    varList = []
    for yamlVar in yaml.load_all(yamlString, Loader = yaml.SafeLoader):
        name = yamlVar['name']
        t = yamlVar['type']
        initialState = yamlVar['initialState']
        if scalingScheme==Scale.LINEAR:
            lowerLimit = yamlVar['lowerLimit'] if 'lowerLimit' in yamlVar else initialState*(1 - variableSpan) # Lower bound set to 90% of the initial state.
            upperLimit = yamlVar['upperLimit'] if 'upperLimit' in yamlVar else initialState*(1 + variableSpan)
        elif scalingScheme == Scale.LOGARITHMIC:
            scales = [initialState / variableScale, initialState * variableScale]
            lowerLimit = yamlVar['lowerLimit'] if 'lowerLimit' in yamlVar else min(scales) # Lower bound set to 90% of the initial state.
            upperLimit = yamlVar['upperLimit'] if 'upperLimit' in yamlVar else max(scales) 
        

        mappedName = yamlVar['mappedName'] if 'mappedName' in yamlVar else name
        desc  = yamlVar['description'] if 'description' in yamlVar else name
        variableCon = variableConfig(name = name, 
                                    initial=initialState, 
                                    varType=t,
                                    mappedName = mappedName,
                                    lowerLimit=lowerLimit, 
                                    upperLimit=upperLimit,
                                    description= desc)
        varList.append(variableCon)
    return varList

    


