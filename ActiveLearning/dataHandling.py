import numpy as np 
import os, system, platform, shutil
import yaml

resultfileName = 'finalReport.yaml'

"""
    Reading a dataset from a group of evaluated samples.
    Input: 
        - Location of the samples

    Output:
        - Dataset in the form of data points and their corresponding labels

    Note: Assumes that the samples are in folders with numerical names. Ignores
    all the other types of names (alphabetical or symbolic characters)

"""
def readDataset(repoLoc):


def readSingleSample(repoLoc, sampleNumber):
