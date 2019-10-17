from . import yamlObjects
import numpy as np
import yaml
import os
import shutil
import matplotlib.pyplot as plt

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

# This function creates a list of random samples for the given variables. 
# The creation of all the random samples for the factors can give the user 
# the control to change the sampling method more conveniently.
# Also this function uses stratified sampling technique to give a more uniform 
# distribution.
def randomizeVariablesList(variables, sampleNum, subIntervals, saveHists=False):
    randomLists = {}
    # Making the random lists with stratified 
    for key in variables:
        var = variables[key]
        randomValues = np.random.rand(sampleNum)
        interval = var.upperLimit - var.lowerLimit
        d = interval / subIntervals
        randomVar = [rv*d + var.lowerLimit+(idx%subIntervals)*d for idx,rv in enumerate(randomValues)] 
        # Shuffling is needed for disruption of correlation between the factors.
        np.random.shuffle(randomVar)
        randomLists[var.name] = randomVar
        if saveHists:
            plt.figure()
            plt.hist(x=randomVar, bins=subIntervals, range=(min(var.lowerLimit, var.upperLimit),max(var.lowerLimit, var.upperLimit)))
            plt.title(var.name)
            plt.savefig(f'./figures/histo{var.name}.png')
            plt.close()
    randValuesList = []
    for counter in range(sampleNum):
        randomSample = {}
        for key in variables:
            var = variables[key]
            randomSample[var.name] = randomLists[var.name][counter]
        randValuesList.append(randomSample)
    return randValuesList




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
    allFiles = [f for f in os.listdir(dataFolder) if not os.path.isdir(f'{dataFolder.rstrip("/")}/{f}') and f.endswith('.mat')]
    for f in allFiles:
        filePath = f'{dataFolder.rstrip("/")}/{f}'
        newPath = f'{newLocation.rstrip("/")}/{f}'
        shutil.copyfile(filePath, newPath)
    return

def copyMetricScript(scriptName, location, newFolder):
    filePath = f'{location.rstrip("/")}/{scriptName}'
    newFilePath = f'{newFolder.rstrip("/")}/{scriptName}'
    
    shutil.copyfile(filePath, newFilePath)
    return



