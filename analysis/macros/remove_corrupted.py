#!/usr/bin/env python
import json
from optparse import OptionParser
import os
import sys

parser = OptionParser()
parser.add_option('-m', '--metadata', help='metadata', dest='metadata', default='2018')
parser.add_option('-l', '--lists', help='lists', dest='lists', default=None)
(options, args) = parser.parse_args()

try:
    os.system('ls '+options.lists)
except:
    sys.exit('File',options.lists,'does not exist')
  
remove = []
corrupted = open(options.lists, 'r')
for rootfile in corrupted.readlines():
    remove.append(rootfile.strip().split('store')[1])

dictionary={}
with open(options.metadata) as fin:
    dictionary.update(json.load(fin))

list_datasets = []
for key in dictionary:
    for rootfile in dictionary[key]["files"]:
        if rootfile.split('store')[-1] in remove:
            dictionary[key]["files"].remove(rootfile)
            if key not in list_datasets: list_datasets.append(key)
            
print("Found corrupted file in", ','.join(list_datasets))

with open(options.metadata, "w") as fout:
    json.dump(dictionary, fout, indent=4)
