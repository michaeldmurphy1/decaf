import concurrent.futures
import cloudpickle
import pickle
import gzip
import os
import numpy as np
from collections import defaultdict, OrderedDict
from coffea import hist, processor 
from coffea.util import load, save
from helpers.futures_patch import patch_mp_connection_bpo_17560

def merge(folder):

     hists = {}
     for file in os.listdir(folder):
          if '.reduced' not in file: continue
          filename = folder+'/'+file
          print('Opening:',filename)
          hin = load(filename)
          if filename.split('--')[0] not in hists: hists[filename.split('--')[0]]
          hists[file.split('--')[0]].update(hin[file.split('--')[0]])
     print(hists)
     save(hists,folder+'.merged')
     

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-f', '--folder', help='folder', dest='folder')
    (options, args) = parser.parse_args()

    patch_mp_connection_bpo_17560()    
    merge(options.folder)
