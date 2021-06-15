#! /usr/bin/python3

import sys 
from yamlParseObjects.variablesUtil import *
import repositories
from yamlParseObjects.yamlObjects import *
from ActiveLearning.simInterface import * 

sys.path.append('/home/caps/.wine/drive_c/HIL-TB/py')
simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.yaml')
modelLoc = repositories.cefLoc + simConfig.modelLocation
print('Model Location: ' + modelLoc)
print('Current directory: ', repositories.currentDir)
variablesFile = './assets/yamlFiles/variables_ac_pgm.yaml'
descriptionFile = './assets/yamlFiles/varDescription.yaml'

variables = getAllVariableConfigs(yamlFileAddress=variablesFile, scalingScheme=Scale.LINEAR,
            span = 0.65) 
for v in variables: 
    print(f'Variable: {v.name}, mapped name: {v.mappedName}, Initial value: {v.initialState}')
simRepo = repositories.remoteRepo97
print(simRepo)
experFile = './assets/experiments/FFD_new_Framework.txt'
descFile = './varDescription.yaml'

timeIndepVars = getTimeIndepVars(variables, omitZero=True) 
# exper = fractionalFactorialExperiment(timeIndepVars, res4 = True)
# runExperiment(modelLoc, timeIndepVars, simRepo, exper, experFile, descFile)


exper = cvtMonteCarlo(timeIndepVars, sampleNum=200)
saveSampleToTxtFile(samples = exper, fileName = './assets/experiments/cvt_monteCarlo.txt')
for idx, ex in enumerate(exper): 
    print(f'{idx+1}: ', ex)
runExperiment(modelLoc, timeIndepVars, simRepo, exper, experFile, descFile)
