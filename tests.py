#! /usr/bin/python3

import repositories as repo
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.dataHandling import * 
import glob

repoLoc = repo.adaptRepo10
dataLoc= repoLoc + '/data' 
sampleFolders = getSampleFolders(dataLoc = dataLoc, sort = True, descending=True)
# print(sampleFolders)
lastSample = 1
for sampleFolder in sampleFolders:
    dataFolder = dataLoc + f'/{sampleFolder}'
    dataFile = glob.glob(dataFolder + '/*.mat')
    if len(dataFile)>0:
        lastSample = sampleFolder
        break 
print(lastSample)
