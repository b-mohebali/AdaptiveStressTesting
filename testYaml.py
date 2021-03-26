import yaml

dict_file = [{'sports' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis']},
{'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']}]
print(dict_file)

with open('./yamlTest.yaml','w') as yamlFile:
    docs = yaml.dump(dict_file, yamlFile)

with open('./yamlTest.yaml') as f:
    fList = yaml.load(f, Loader=yaml.FullLoader)
    print(fList)

with open('./varDescription.yaml') as varDesc:
    vDesc = yaml.load(varDesc, Loader = yaml.FullLoader)
print(vDesc)
totalYaml = {}
totalYaml['descs'] = vDesc
totalYaml['results'] = 0
totalYaml['time'] = 12
print(totalYaml)
with open('./yamlTest.yaml','w') as yamlFile:
    descs = yaml.dump(totalYaml, yamlFile)
