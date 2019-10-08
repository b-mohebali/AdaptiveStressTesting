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

    def __init__(self, name, initial, varType, lowerLimit, upperLimit, mappedName, risingRateLimit = float('inf'), fallingRateLimit = float('inf')):
        self.name = name
        self.initialState = initial
        self.varType = varType
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.risingRateLimit = risingRateLimit
        self.fallingRateLimit = fallingRateLimit
        self.mappedName = mappedName if mappedName is not None else name
    
    def __str__(self):
        descroptor = f'''Variable name: {self.name}
        type: {self.varType}
        Value: {self.initialState}
        Value range: [{self.lowerLimit:0.6f}, {self.upperLimit:0.6f}]
        '''
        return descroptor.__str__()

def getAllVariableConfigs(yamlFileAddress):
    with open(yamlFileAddress,'rt') as fp:
        yamlString = fp.read()
    fp.close()
    varList = []
    for yamlVar in yaml.load_all(yamlString, Loader = yaml.SafeLoader):
        name = yamlVar['name']
        t = yamlVar['type']
        initialState = yamlVar['initialState']
        lowerLimit = yamlVar['lowerLimit'] if 'lowerLimit' in yamlVar else initialState*0.9 # Lower bound set to 90% of the initial state.
        upperLimit = yamlVar['upperLimit'] if 'lowerLimit' in yamlVar else initialState*1.1
        mappedName = yamlVar['mappedName'] if 'mappedName' in yamlVar else name
        variableCon = variableConfig(name = name, 
                                    initial=initialState, 
                                    varType=t,
                                    mappedName = mappedName,
                                    lowerLimit=lowerLimit, 
                                    upperLimit=upperLimit)
        varList.append(variableCon)
    return varList

    

def getVariablesDict(variables):
    varMap = {}
    for var in [v for v in variables if v.varType.lower() != 'timeindep']: 
        varMap[var.name] = var
    return varMap    

def getTimeIndepVarsDict(variables):
    varMap = {}
    for var in [v for v in variables if v.varType.lower() == 'timeindep']:
        varMap[var.name] = var
    return varMap

