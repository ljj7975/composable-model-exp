import json
import pprint

f = open('../trained/summary.txt', 'r')
results = json.loads(f.read())

pprint.pprint(results)