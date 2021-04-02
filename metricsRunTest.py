import os,sys
import string
from yamlParseObjects.yamlObjects import * 
import time
import matlab.engine


# Testing the function that runs the metrics and saves the label.
def getMetricsResults(dataLocation: string,sampleNumber, figFolderLoc: string):
    if isinstance(sampleNumber, list):
        labels = []
        for sampleNum in sampleNumber:
            l = getMetricsResults(dataLocation, sampleNum, figFolderLoc)
            labels.append(l)
        return labels 
    
    # TODO: check to see if the dataLocation is a valid path
    assert os.path.exists(dataLocation)
    # TODO: Make sure that MATLAB engine is already started.
    # Running the metrics
    startTime = time.time()
    output = eng.runMetrics(dataLocation, sampleNumber, figFolderLoc, nargout = 4)
    endTime = time.time()
    print(output)
    elapsed = endTime - startTime
    print('Time taken to calculate the metrics: ', elapsed)
    # capturing the label as an integer. 
    label = int(output[0])
    # TODO: Capturing the rest of the metric values that may be useful for the factor screening in case we do it on the python side.

    # saving the results to a report file.
    sampleLoc = f'{dataLocation}/{sampleNumber}'
    reportDict = {}
    reportDict['elapsed_time_sec'] = float('{:.5f}'.format(elapsed))
    with open(f'{sampleLoc}/variableValues.yaml') as varValues:
        values = yaml.load(varValues, Loader= yaml.FullLoader)
    reportDict['variables'] = values
    reportDict['result_label'] = label 
    with open(f'{sampleLoc}/finalReport.yaml','w') as reportYaml:
        yaml.dump(reportDict, reportYaml)
    
    return label

# This function is made to test the functionality of the features in this file.
def main():
    simConfig = simulationConfig('./yamlFiles/adaptiveTesting.yaml')
    matlabPath = './Matlab'

    figFolder = 'C:/Users/Behshad/Google Drive/codes/ScenarioGenerator/Figures/MATLAB_figures'

    print(simConfig.sampleRepo)
    eng = matlab.engine.start_matlab()
    eng.addpath(matlabPath)
    sampleNum = 12
    dataLocation = simConfig.sampleRepo

    getMetricsResults(dataLocation,sampleNum, figFolder)


if __name__=='__main__':
    main()

