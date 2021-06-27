#! /usr/bin/python3

import sys
import repositories 
from yamlParseObjects.yamlObjects import * 
from ActiveLearning.simInterface import * 

def main():
    repoLoc = repositories.monteCarlo400Droop
    sampleLoc = repoLoc + '/data'
    figLoc = repoLoc + '/figures'
    simConfig = simulationConfig('./assets/yamlFiles/ac_pgm_conf.ymal')
    modelLoc = repositories.cefLoc + simConfig.modelLocation
    variablesFile = './assets/yamlFiles/ac_pgm_adaptive.yaml'
    

    return 




if __name__=='__main__':
    main()
