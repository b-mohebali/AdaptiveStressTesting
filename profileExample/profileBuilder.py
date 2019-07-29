import csv
from math import sin,cos
from yamlParseObjects.yamlObjects import variableConfig

# This function creates a sample scenario with two sine waves. 
def buildCsvProfile(fileLoc = '.', fileName = 'sample'):
    fileNameExt = fileLoc + '/' +fileName + '.csv'
    print(fileNameExt)
    with open(fileNameExt, 'w') as csvFile: 
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['Time','var1','var2'])
        t = 0.0
        tStep = 0.01
        while t < 210.0 + tStep:
            data = [float("{0:.6f}".format(t)), 
                    float("{0:.6f}".format(1+0.5*sin(t*3))),  
                    float("{0:.6f}".format(7 + 0.5* cos(t)))]
            csvWriter.writerow(data)
            t += tStep
    return 


def createMappingFile(variables, fileLoc = '.', fileName = 'mapping', profileFileName = 'sample'):
    fileNameExt = fileLoc + '/' + fileName + '.csv'
    print(f'Mapping file name: {fileNameExt}')
    with open(fileNameExt, 'w') as csvFile: 
        csvWriter = csv.writer(csvFile)

        for var in variables:
            var.


class ScenarioBuilder:
    
    def __init__(self, simConfig,variables):
        self.timeStep = simConfig.timeStep
        self.vars = variables
    
    ## TODO
    def randomizeVariables(self):
        pass

    
class Event():

    def __init__(self):
        self.duration = 0
        

        
