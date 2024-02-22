#!/usr/bin/env python
import pickle
import json
import time
import gzip
import os
from optparse import OptionParser

import uproot
uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource

import numpy as np
from coffea import processor
from coffea.util import load, save
from coffea.nanoevents import NanoAODSchema

import warnings
warnings.filterwarnings("ignore")

class CustomNanoAODSchema(NanoAODSchema):
    mixins = {
        **NanoAODSchema.mixins,
        "AK15PFPuppiJet": "FatJet",
        "AK15PFPuppiSubjet": "PtEtaPhiMCollection",
    }
    all_cross_references = {
        **NanoAODSchema.all_cross_references,
        "AK15PFPuppiJet_subJetIdx1": "AK15PFPuppiSubjet",  
        "AK15PFPuppiJet_subJetIdx2": "AK15PFPuppiSubjet",  
    }
    nested_items = {
        **NanoAODSchema.nested_items,
        "AK15PFPuppiJet_subJetIdxG": ["AK15PFPuppiJet_subJetIdx1G", "AK15PFPuppiJet_subJetIdx2G"]
    }
    def __init__(self, base_form):
        for key in base_form["contents"].copy():
            if '_Jet' in key:
                popped = base_form["contents"].pop(key)
                base_form["contents"][key.replace('_Jet','Jet')] = popped
            if '_Subjet' in key:
                popped = base_form["contents"].pop(key)
                base_form["contents"][key.replace('_Subjet','Subjet')] = popped
        #print("Base form keys are:",base_form["contents"].keys())
        super().__init__(base_form)

def run(processor_instance, samplefiles):
    fileslice = slice(None)
    for dataset, info in samplefiles.items():
        filelist = {}
        if options.dataset:
            if not any(_dataset in dataset for _dataset in options.dataset.split(',')): continue
        print('Processing:',dataset)
        files = []
        for file in info['files'][fileslice]:
            files.append(file)
        filelist[dataset] = files
    
        tstart = time.time()
        output = processor.run_uproot_job(filelist,
                                          'Events',
                                          processor_instance=processor_instance,
                                          executor=processor.futures_executor,
                                          executor_args={'schema': CustomNanoAODSchema, 'workers': options.workers},
                                          ) 
        
        os.system("mkdir -p hists/"+options.processor)
        save(output,'hists/'+options.processor+'/'+dataset+'.futures')        
        dt = time.time() - tstart
        nworkers = options.workers
        print("%.2f us*cpu overall" % (1e6*dt*nworkers, ))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--processor', help='processor', dest='processor')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
    parser.add_option('-w', '--workers', help='Number of workers to use for multi-worker executors (e.g. futures or condor)', dest='workers', type=int, default=8)
    (options, args) = parser.parse_args()
    
    processor_instance=load('data/'+options.processor+'.processor')
    with open("metadata/"+options.metadata+".json") as fin:
        samplefiles = json.load(fin)
    run(processor_instance, samplefiles)
