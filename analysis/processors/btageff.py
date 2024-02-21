import os
import numpy
import json
from coffea import processor, util
from hist import hist
from coffea.util import save, load
from optparse import OptionParser
import numpy as np
from coffea.lookup_tools.dense_lookup import dense_lookup

class BTagEfficiency(processor.ProcessorABC):

    def __init__(self, year, wp, ids):
        self._year = year
        self._btagWPs = wp
        self._ids = ids
        self.make_output = lambda: {
            'deepflav' :
            hist.Hist(
                axis.StrCategory([], growth=True, name="wp", label="Working point"),
                axis.StrCategory([], growth=True, name="btag", label="BTag WP pass/fail"),
                axis.IntCategory([0, 4, 5, 6], name="flavor", label="Jet hadronFlavour"),
                axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt", label=r'Jet $p_{T}$ [GeV]'),
                axis.Variable([0, 1.4, 2.0, 2.5], name="abseta", label=r'Jet |$\eta$|'),
            ),
            'deepcsv' :
            hist.Hist(
                axis.StrCategory([], growth=True, name="wp", label="Working point"),
                axis.StrCategory([], growth=True, name="btag", label="BTag WP pass/fail"),
                axis.IntCategory([0, 4, 5, 6], name="flavor", label="Jet hadronFlavour"),
                axis.Variable([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000], name="pt", label=r'Jet $p_{T}$ [GeV]'),
                axis.Variable([0, 1.4, 2.0, 2.5], name="abseta", label=r'Jet |$\eta$|'),
            )
        }

    def process(self, events):
        
        dataset = events.metadata['dataset']
        isGoodJet = self._ids['isGoodJet']

        j = events.Jet
        j['isgood'] = isGoodJet(j.pt, j.eta, j.jetId, j.puId, j.neHEF, j.chHEF, self._year)
        j_good = j[j.isgood]

        name = {}
        name['deepflav']= 'btagDeepFlavB'
        name['deepcsv']= 'btagDeepB'

        output = self.make_output()
        for wp in ['loose','medium','tight']:
            for tagger in ['deepflav','deepcsv']:
                passbtag = j_good[name[tagger]] > self._btagWPs[tagger][self._year][wp]
                out[tagger].fill(
                    wp=wp,
                    btag='pass',
                    flavor=j_good[passbtag].hadronFlavour.flatten(),
                    pt=j_good[passbtag].pt.flatten(),
                    abseta=abs(j_good[passbtag].eta.flatten()),
                )
                out[tagger].fill(
                    wp=wp,
                    btag='fail',
                    flavor=j_good[~passbtag].hadronFlavour.flatten(),
                    pt=j_good[~passbtag].pt.flatten(),
                    abseta=abs(j_good[~passbtag].eta.flatten()),
                )
        return out

    def postprocess(self, a):
        return a

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-y', '--year', help='year', dest='year')
    parser.add_option('-m', '--metadata', help='metadata', dest='metadata')
    parser.add_option('-n', '--name', help='name', dest='name')
    (options, args) = parser.parse_args()


    with open('metadata/'+options.metadata+'.json') as fin:
        samplefiles = json.load(fin)

    common = load('data/common.coffea')
    ids    = load('data/ids.coffea')
    processor_instance=BTagEfficiency(year=options.year,wp=common['btagWPs'],ids=ids)

    save(processor_instance, 'data/btageff'+options.name+'.processor')
