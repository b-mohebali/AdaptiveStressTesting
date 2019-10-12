from . import yamlObjects
import numpy as np
import yaml
import os
import shutil

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

# This function gets a dictionary of variables and their values and 
# saves them in a yaml file.
def saveVariableValues(variables, fileName):
    print("This is the variables map: " , variables)
    v={}
    for key in variables:
        v[key] = float(variables[key]) 
    with open(fileName, 'w') as outFile:
        yaml.dump(v, outFile, default_flow_style=False)
    return

def createNewDatafolder(parent):
    allFolders = os.listdir(parent)
    s = [int(f) for f in allFolders if f.isdigit()]
    nextOne = 1 if len(s)==0 else max(s) + 1 
    newFolderPath = f'{parent.rstrip("/")}/{nextOne}/'
    os.mkdir(newFolderPath)
    return newFolderPath

def copyDataToNewLocation(newLocation, dataFolder):
    allFiles = [f for f in os.listdir(dataFolder) if not os.path.isdir(f'{dataFolder.rstrip("/")}/{f}')]
    for f in allFiles:
        filePath = f'{dataFolder.rstrip("/")}/{f}'
        newPath = f'{newLocation.rstrip("/")}/{f}'
        shutil.copyfile(filePath, newPath)
    return


