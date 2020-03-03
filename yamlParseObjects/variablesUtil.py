from .yamlObjects import *

import numpy as np
import yaml
import os
import shutil
import matplotlib.pyplot as plt
import subprocess
from scipy.linalg import hadamard


def trunkatedExponential(u, a,b):
    beta = np.e/(np.e-1.)
    xHat = -1 * np.log(1 - u /beta)
    x = a + xHat * (b-a)
    return x

trunkatedExponentialVec = np.vectorize(trunkatedExponential)

def getTimeDepVariables(variables):
    vars = []
    for var in [v for v in variables if v.varType.lower() != 'timeindep']: 
        vars.append(var)
    return vars

def getTimeindepVariablesDict(variables):
    varMap = {}
    for var in [v for v in variables if v.varType.lower() == 'timeindep']: 
        varMap[var.name] = var
    return varMap    

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

def getTimeIndepVars(variables):
    return [v for v in variables if v.varType.lower() == 'timeindep']

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
# An addition to this function will make it able to sample from a trunkated 
# exponential distribution instead of a uniform one. This is needed for the 
# time when the limits of the variation range are defined as logarithmic instead 
# of linear. 
def randomizeVariablesList(variables, sampleNum, subIntervals,scalingScheme = Scale.LINEAR, saveHists=False):
    randomLists = {}
    # Making the random lists with stratified 
    for key in variables:
        var = variables[key]
        randomValues = np.random.rand(sampleNum)
        if scalingScheme == Scale.LINEAR:
            interval = var.upperLimit - var.lowerLimit
            d = interval / subIntervals
            randomVar = [rv*d + var.lowerLimit+(idx%subIntervals)*d for idx,rv in enumerate(randomValues)] 
        elif scalingScheme == Scale.LOGARITHMIC:
            d = 1. / subIntervals
            stratified = [rv*d + d*(idx%subIntervals) for idx, rv in enumerate(randomValues)] 
            randomVar = trunkatedExponentialVec(stratified, var.lowerLimit, var.upperLimit)
        # Shuffling is needed for disruption of correlation between the factors.
        np.random.shuffle(randomVar)
        randomLists[var.name] = randomVar
        # Saving the plots of the distributions. 
        if saveHists:
            plt.figure()
            plt.hist(x=randomVar, bins=subIntervals, range=(min(var.lowerLimit, var.upperLimit),max(var.lowerLimit, var.upperLimit)), rwidth=0.95)
            plt.title(var.name)
            plt.xlabel('Bins')
            plt.ylabel('Number of samples in the bin')
            plt.savefig(f'./figures/histo_{var.name}.png')
            plt.close()
            plt.figure()
            plt.scatter(range(1,1 + sampleNum),randomVar,marker='.',s=2)
            plt.title(f'Scatter plot of {var.name} distribution')
            plt.savefig(f'./figures/scatter_{var.name}.png')
            plt.xlabel('Number of sample')
            plt.ylabel('Sample value')
            plt.close()
    randValuesList = []
    for counter in range(sampleNum):
        randomSample = {}
        for key in variables:
            var = variables[key]
            randomSample[var.name] = randomLists[var.name][counter]
        randValuesList.append(randomSample)
    return randValuesList

# This function creates the samples for One-At-A-Time sensitivity analysis
# The simulation will run the model for the factors at the extremes of their 
# range. In the sample set, each factor has k simulation when it is at its 
# minimum and k simulation when it is at its maximum.
def OATSampleGenerator(varDict, addMiddle = False):
    randValuesList = []
    currentSample = {}
    middleSample = {}
    for key in varDict:
        var = varDict[key]
        currentSample[var.name] = var.lowerLimit
        middleSample[var.name] = var.lowerLimit + (var.upperLimit - var.lowerLimit)/2.
    for key in varDict:
        var = varDict[key]
        currentSample[var.name] = var.upperLimit
        randValuesList.append(currentSample.copy())
    for key in varDict:
        var = varDict[key]
        currentSample[var.name] = var.lowerLimit
        randValuesList.append(currentSample.copy())
    # Adding the middle sample:
    if addMiddle:
        randValuesList.append(middleSample.copy())
    return randValuesList

def standardOATSampleGenerator(variables):
    varDict = getTimeindepVariablesDict(variables)
    valuesList = []
    standardSample = {}
    for key in varDict:
        var = varDict[key]
        standardSample[var.name]= var.initialState
    for key in varDict:
        var = varDict[key]
        nextSample = standardSample.copy()
        nextSample[var.name] = var.upperLimit
        valuesList.append(nextSample.copy())
        nextSample = standardSample.copy()
        nextSample[var.name] = var.lowerLimit
        valuesList.append(nextSample.copy())
    return valuesList


def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


# This function takes a LIST of the time independent variable. 
# The output of the function is an experiment in the form of a series of 
# simulation parameters that are generated using the Hadamard matrices. 
# To know more about the details please refer to Saltelli's Global Sensitivity
# Analysis, a primer book, chapter 2.  
def fractionalFactorialExperiment(timeIndepVariables,res4=False):
    h = getFFHMatrix(timeIndepVariables, res4 = res4)
    valueList = []
    simNumber = h.shape[0]
    for simIndex in range(simNumber):
        currentSample = {}
        for varIndex,var in enumerate(timeIndepVariables):
            currentSample[var.name] = h[simIndex, varIndex]
        valueList.append(currentSample.copy())
    return valueList

def getFFHMatrix(vars, res4 = False, dtype = float):
    k = len(vars)
    h = hadamard(next_power_of_2(k),dtype = float)
    if res4:
        h = np.concatenate((h,-h),axis = 0)
    for idx,var in enumerate(vars):
        h[h[:,idx+1]==1,idx+1] = var.upperLimit
        h[h[:,idx+1]==-1,idx+1] = var.lowerLimit
    return h[:,1:k+1] # Only returning the part of the H matrix that is used as simulation parameters.


def getVariablesInitialValueDict(variables):
    initials = {}
    for var in variables:
        initials[var.name] = var.initialState
    return initials

# This function gets a dictionary of variables and their values and 
# saves them in a yaml file.
def saveVariableValues(varDict, fileName):
    print("This is the variables map: " , varDict)
    v={}
    for key in varDict:
        v[key] = float(varDict[key]) 
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

def copyDataToremoteServer(dataRepo, dataFolder):
    cmd = f'scp -r {dataFolder} {dataRepo}'
    print(cmd)
    os.system(cmd)
    return   

# This function will remove the old data folders (that are already copied to the remote 
# server) so that the number of data folders in the source repo stays 
# at maxSize specified in the function signature. 
def removeExtraFolders(dataFolder, maxSize):
    s = [int(f) for f in os.listdir(dataFolder) if f.isdigit()]
    while len(s) > maxSize:
        toRemove = f'{dataFolder.rstrip("/")}/{min(s)}'
        print(f'Removing {toRemove}')
        shutil.rmtree(toRemove)
        s = [int(f) for f in os.listdir(dataFolder) if f.isdigit()]
    return 

def emptyFolder(folderName):
    s = [int(f) for f in os.listdir(folderName) if f.isdigit()]
    for f in s:
        toRemove = f'{folderName.rstrip("/")}/{f}'
        print(f'Removing {toRemove}')
        shutil.rmtree(toRemove)
    return 

    
def copyMetricScript(scriptName, location, newFolder):
    filePath = f'{location.rstrip("/")}/{scriptName}'
    newFilePath = f'{newFolder.rstrip("/")}/{scriptName}'
    
    shutil.copyfile(filePath, newFilePath)
    return


def saveSampleToTxtFile(samples, fileName):
    with open(fileName,'w') as f:
        for sample in samples:
            f.write(sample.__str__() + '\n')
 

# This function will generate the samples needed for verification of the 
# samples (FFD and OAT). The idea is to check the variation of the output
# around the initial point with logarithmic scales. 
def generateVerifSample(variables):
    varInitial = getVariablesInitialValueDict(variables)
    varDict = getTimeindepVariablesDict(variables)
    sampleList = []
    sampleList.append(varInitial.copy())
    scales = [0.1,0.5,2,10]
    for key in varInitial:
        var = varDict[key]
        for scale in scales:
            currentSample = varInitial.copy()
            currentSample[key] = currentSample[key] * scale
            if currentSample[key] > var.upperLimit: 
                currentSample[key] = var.upperLimit
            if currentSample[key] < var.lowerLimit:
                currentSample[key] = var.lowerLimit
            sampleList.append(currentSample.copy())
    return sampleList

def findDifferentValue(baseLine, sample):
    for key in baseLine:
        if baseLine[key] != sample[key]:
            return key
    return None