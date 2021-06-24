#! /usr/bin/python3

import sys 
from yamlParseObjects.variablesUtil import *
import repositories
from yamlParseObjects.yamlObjects import *
from ActiveLearning.simInterface import * 


repositories.addCodeBaseToPath()

simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
modelLoc = repositories.cefLoc + simConfig.modelLocation
print('Model Location: ' + modelLoc)
print('Current directory: ', repositories.currentDir)
variablesFile = './assets/yamlFiles/variables_ac_pgm.yaml'
descFile = './assets/yamlFiles/varDescription.yaml'

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR,
            span = 0.65) 
for v in variables: 
    print(f'Variable: {v.name}, mapped name: {v.mappedName}, Initial value: {v.initialState}')
experFile = './assets/experiments/FFD_new_Framework.txt'

timeIndepVars = getTimeIndepVars(variables, omitZero=True) 
exper = fractionalFactorialExperiment(timeIndepVars, res4 = True)

simRepo = repositories.remoteRepo101
saveSampleToTxtFile(samples = exper, fileName = './assets/experiments/FFD_sample.txt')
for idx, ex in enumerate(exper): 
    print(f'{idx+1}: ', ex)
runExperiment(modelLoc, timeIndepVars, simRepo, exper, experFile, descFile)



