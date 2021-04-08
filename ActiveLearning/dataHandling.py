import numpy as np 
import os, sys, platform, shutil
import yaml
import string
from typing import List
from yamlParseObjects.yamlObjects import *


resultFileName = 'finalReport.yaml'

def readDataset(repoLoc, variables:List[variableConfig]):
    """
        Reading a dataset from a group of evaluated samples.
        Input: 
            - Location of the samples
            - List of the variables configurations for the order of their names

        Output:
            - Dataset in the form of data points as a list of lists
            - Label of each sample in the same order as the samples in the dataset
            - The time each sample took to be evaluated.


        Note: Assumes that the samples are in folders with numerical names. Ignores
        all the other types of names (alphabetical or symbolic characters)

    """
    sampleFolders = [name for name in os.listdir(repoLoc) if name.isdigit()]
    labels = []
    dataset = []
    elapsed_times = []
    for sampleFolder in sampleFolders: 
        d,l,t = readSingleSample(repoLoc, variables, sampleFolder)
        labels.append(l)
        dataset.append(d)
        elapsed_times.append(t)
    return dataset, labels, elapsed_times


def readSingleSample(repoLoc,variables: List[variableConfig], sampleNumber):
    """
        This function read the results of a single sample from the repo.

        Inputs:
            - Location of the samples (repoLoc)
            - List of varaibles configuration (variables)
            - Sample Number (sampleNumber)

        Output:
            - list of the values for the dimensions of the sample (varList)
            - Label of the sample as it was evaluated by the metrics (label)
            - The time the sample took to be evaluated.

    """
    yamlFile = repoLoc + f'/{sampleNumber}/{resultFileName}'
    reportObj = FinalReport(yamlFile)
    varList = []
    for var in variables:
        varList.append(reportObj.variables[var.name])
    return varList, reportObj.label, reportObj.elapsed_time


def getNextSampleLocation(repoLoc, createFolder:bool = False):
    """
        Gets the location of the repository and determines what is the path to save the next sample.
    """
    sampleFolders = [int(name) for name in os.listdir(repoLoc) if name.isdigit()]
    lastSampleNumber = max(sampleFolders)
    nextSampleNumber = lastSampleNumber + 1
    nextSamplePath = repoLoc + f'{nextSampleNumber}'
    if createFolder and not os.path.isdir(nextSamplePath):
        os.mkdir(nextSamplePath)
    return nextSamplePath