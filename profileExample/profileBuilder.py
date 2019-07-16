import csv
from math import sin,cos


def buildCsvProfile(fileLoc = '.', fileName = 'sample'):
    fileNameExt = fileLoc + '/' +fileName + '.csv'
    print(fileNameExt)
    with open(fileNameExt, 'w') as csvFile: 
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(['Time','var1','var2'])
        
        t = 0.0
        while t < 10.0 + 0.05:
            data = [float("{0:.6f}".format(t)), 1+0.5*sin(t), 7 + 0.5* cos(t)]
            csvWriter.writerow(data)
            t += 0.05
    return 



