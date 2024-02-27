#!/usr/bin/env python
import os
from data.process import *
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
parser.add_option('-p', '--pack', help='pack', dest='pack')
parser.add_option('-s', '--special', help='special', dest='special')
(options, args) = parser.parse_args()

globalredirect = "root://xrootd-cms.infn.it/"

campaigns ={}
campaigns['2016pre'] = '*UL*16*JMENano'
campaigns['2016post'] = '*UL*16*JMENano'
campaigns['2017'] = '*UL*17*JMENano'
campaigns['2018'] = '*UL*18*JMENano'

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

xsections={}
for k,v in processes.items():
     if v[1]=='MC':
          if not isinstance(k, str):
               if options.year!=str(k[1]): continue
               xsections[k[0]] = v[2]
          else: 
               xsections[k] = v[2]
     else:
          xsections[k] = -1

datadef = {}
datasets = []
for dataset in xsections.keys():
    if options.dataset:
        if not any(_dataset in dataset for _dataset in options.dataset.split(',')): continue
    query="dasgoclient --query=\"file dataset=/"+dataset+"*/"+campaigns[options.year]+"*/NANOAODSIM\""
    print(query)
    urllist = datasets.append(os.popen(query).read().split("\n"))
    for url in urllist[:]:
        if options.year not in str(url):
            urllist.remove(url)
            continue
        if '.root' not in url: 
            urllist.remove(url)
            continue
        urllist[urllist.index(url)]=globalredirect+url
    print('list lenght:',len(urllist))
    if options.special:
         for special in options.special.split(','):
              sdataset, spack = special.split(':')
              if sdataset in dataset:
                   print('Packing',spack,'files for dataset',dataset)
                   urllists = split(urllist, int(spack))
              else:
                   print('Packing',int(options.pack),'files for dataset',dataset)
                   urllists = split(urllist, int(options.pack))
    else:
         print('Packing',int(options.pack),'files for dataset',dataset)
         urllists = split(urllist, int(options.pack))
    print(len(urllists))
    if urllist:
        for i in range(0,len(urllists)) :
             datadef[dataset+"____"+str(i)+"_"] = {
                  'files': urllists[i],
                  'xs': xs,
                  }
        
folder = "metadata/"+options.metadata+".json"
with open(folder, "w") as fout:
    json.dump(datadef, fout, indent=4)
