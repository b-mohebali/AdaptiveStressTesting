import yaml
from yaml.dumper import SafeDumper
from yamlParseObjects.yamlObjects import IterationReport
from datetime import datetime

# fileName = 'yamlTest.yaml'

# varNames = 'this is just a string'.split()
# iterReport = IterationReport(varNames)
# iterReport.batchSize = 2
# iterReport.iterationNumber = 43
# iterReport.startTime = datetime.now()
# iterReport.stopTime = datetime.now()
# iterReport.budgetRemaining = 23

# iterReport.changeMeasure = 1

# with open(fileName, 'wt') as yFile:
#     yaml.dump_all([iterReport, iterReport, iterReport], yFile, Dumper = yaml.Dumper)
    

realFileName = '/home/caps/AdaptiveSamplingRepo/iterationReport.yaml'

with open (realFileName, 'rt') as fp:
    yamlString = fp.read()
yamlObj = yaml.load_all(yamlString, Loader = yaml.Loader)

for obj in yamlObj:
    print(obj.changeMeasure)
