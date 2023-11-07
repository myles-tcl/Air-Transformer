import json

path = './'
with open(path + 'config_global.json', 'r') as f:
    config1 = json.load(f)
with open(path + 'config_model.json', 'r') as f:
    config2 = json.load(f)
