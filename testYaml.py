import yaml
from yamlParseObjects.yamlObjects import IterationReport

# with open('./yamlTest.yaml') as f:
#     fList = yaml.load(f, Loader=yaml.FullLoader)
#     print(fList)
# repoLoc = './'
# sampleNum = 1
# reportFileName = 'finalReport.yaml'
# reportFileAddress = f'{repoLoc}/{sampleNum}/{reportFileName}'
# print(reportFileAddress)
# with open(reportFileAddress, 'rt') as fp:
#     report = yaml.load(fp, Loader = yaml.FullLoader)
#     print(report)
from datetime import datetime 
iterReports = []
myIter = IterationReport()
myIter.iterationNumber = 1
myIter.startTime = datetime.now()
myIter.stopTime = 12
myIter.samples = [1,2,3,4]
myIter.metricsResults = [0,1,0,0]
myIter.budgetRemaining = 100
myIter.changeMeasure = [0.5,0.6,0.4,0.3]
iterReports.append(myIter)

myIter2 = IterationReport()
myIter2.iterationNumber = 2
myIter2.startTime = 12
myIter2.stopTime = 14
myIter2.samples = [5,6,7,8]
myIter2.metricsResults = [1,0,1,0]
myIter2.budgetRemaining = 99
myIter2.changeMeasure = [0.5,0.2,0.1,0.05]

iterReports.append(myIter2)

myIter3 = IterationReport()
myIter3.iterationNumber = 3
myIter3.startTime = datetime.now()
myIter3.stopTime = 18
myIter3.samples = [9,10,11,12]
myIter3.metricsResults = [0,1,0,1]
myIter3.budgetRemaining = 98
myIter3.changeMeasure = [0.5,0.2,0.1,0.05]

iterReports.append(myIter3)

with open('./yamlTest.yaml','w') as yamlFile:
    docs = yaml.dump_all(iterReports, yamlFile)
import time 
start = datetime.now()
print(datetime.now())
for _ in range(10000):
    pass
stop = datetime.now()
print(stop)
print(stop-start)
