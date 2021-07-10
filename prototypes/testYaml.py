#! /usr/bin/python3

import yaml
from yaml.loader import SafeLoader
from yamlParseObjects.yamlObjects import IterationReport



reportFile = r'/home/caps/Data/Tests/test7/data/1/finalReport.yaml'
with open(reportFile, 'r') as rf:
    yamlString = rf.read()
yamlObj = yaml.load(yamlString, Loader = SafeLoader)
print(yamlObj)
metName = 'Max dev V freq pu'
print(yamlObj[metName])