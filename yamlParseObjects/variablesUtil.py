from . import yamlObjects
import numpy as np

def getTimeDepVariables(variables):
    vars = []
    for var in [v for v in variables if v.varType.lower() != 'timeindep']: 
        vars.append(var)
    return vars



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



# This function gets a list of variables config and returns a random value for each one within
# its specified range of available values. 
def randomizeVariables(variables):
    randomValues = {}
    values = np.random.rand(len(variables))
    for idx,key in enumerate(variables):
        var = variables[key]
        range = var.upperLimit - var.lowerLimit
        randomValues[var.name] = values[idx] * range + var.lowerLimit
    return randomValues


def getVariablesInitialValueDict(variables):
    initials = {}
    for var in variables:
        initials[var.name] = var.initialState
    return initials