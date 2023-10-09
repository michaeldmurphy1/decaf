#! /usr/bin/env python
import correctionlib
import os
import awkward as ak

import uproot, uproot_methods
import numpy as np
from coffea import hist, lookup_tools
from coffea.lookup_tools import extractor, dense_lookup

###
# MET trigger efficiency SFs, 2017/18 from monojet. Depends on recoil.
###

met_trig_hists = {
    '2016': uproot.open("data/trigger_eff/metTriggerEfficiency_recoil_monojet_TH1F.root")['hden_monojet_recoil_clone_passed'],
    '2017': uproot.open("data/trigger_eff/met_trigger_sf.root")['120pfht_hltmu_1m_2017'],
    '2018': uproot.open("data/trigger_eff/met_trigger_sf.root")['120pfht_hltmu_1m_2018']
}
get_met_trig_weight = {}
for year in ['2016','2017','2018']:
    met_trig_hist=met_trig_hists[year]
    get_met_trig_weight[year] = lookup_tools.dense_lookup.dense_lookup(met_trig_hist.values, met_trig_hist.edges)


####
# Electron
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaSFJSON
# jsonPOG: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM
####

def get_ele_id_sf (year, eta, pt):
    evaluator = correctionlib.CorrectionSet.from_file('data/EGammaSF/'+year+'_UL/electron.json.gz')

    flateta, counts = ak.flatten(eta), ak.num(eta)
    flatpt = ak.flatten(pt)
    weight = evaluator["UL-Electron-ID-SF"].evaluate(year, "sf", "Medium", flateta, flatpt)

    return ak.unflatten(weight, counts=counts)


####
# Photon
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaSFJSON
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM
####

## photon tight id sf ##
def get_pho_id_sf(year, eta, pt):
    evaluator = correctionlib.CorrectionSet.from_file('data/EGammaSF/'+year+'_UL/photon.json.gz')

    flateta, counts = ak.flatten(eta), ak.num(eta)
    flatpt = ak.flatten(pt)
    weight = evaluator["UL-Photon-ID-SF"].evaluate(year, "sf", "Tight", flateta, flatpt)

    return ak.unflatten(weight, counts=counts)

## photon CSEV sf ##
# 



ex = '''
btvjson = correctionlib.CorrectionSet.from_file('data/BtagSF/'+year+'_UL/btagging.json.gz')
# case 2: fixedWP correction uncertainty (here tight WP and comb SF)
# evaluate('systematic', 'working_point', 'flavor', 'abseta', 'pt')
bc_jet_sf = btvjson["deepJet_comb"].evaluate("up_correlated", "T", 
            jet_flav[bc_jets], jet_eta[bc_jets], jet_pt[bc_jets])
light_jet_sf = btvjson["deepJet_incl"].evaluate("up_correlated", "T", 
            jet_flav[light_jets], jet_eta[light_jets], jet_pt[light_jets])
print("\njet SF up_correlated for comb at tight WP:")
print(f"SF b/c: {bc_jet_sf}")
print(f"SF light: {light_jet_sf}")
'''

def btag_weight(year, flavor, abseta, pt):
    btvjson = correctionlib.CorrectionSet.from_file('data/BtagSF/'+year+'_UL/btagging.json.gz')
    bc_jet_sf = btvjson["deepJet_comb"].evaluate("up_correlated", "T",
            jet_flav[bc_jets], jet_eta[bc_jets], jet_pt[bc_jets])
    light_jet_sf = btvjson["deepJet_incl"].evaluate("up_correlated", "T",
            jet_flav[light_jets], jet_eta[light_jets], jet_pt[light_jets])


####
# PU weight
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
####
#trueint = events.Pileup.nTrueInt
def pu_weight(year, trueint):
    correction = {'2018': 'Collisions18_UltraLegacy_goldenJSON',
                  '2017': 'Collisions17_UltraLegacy_goldenJSON',
                  '2016preVFP': 'Collisions16_UltraLegacy_goldenJSON',
                  '2016postVFP':'Collisions16_UltraLegacy_goldenJSON'}
    evaluator = correctionlib.CorrectionSet.from_file('data/PUweight/'+year+'_UL/puWeights.json.gz')
    weight = evaluator[correction[year]].evaluate(trueint, 'nominal')

    return weight


####
# XY MET Correction
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETRun2Corrections#xy_Shift_Correction_MET_phi_modu
####

# https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/
# correction_labels = ["metphicorr_pfmet_mc", "metphicorr_puppimet_mc", "metphicorr_pfmet_data", "metphicorr_puppimet_data"]

def XY_MET_Correction(year, events, pt, phi):
    if 'genWeight' in events.fields:
        isMC = True
    else:
        isData = True

    npv = events.PV.npvsGood
    mask = np.asarray(npv>100)
    npv = np.asarray(npv)
    npv[mask==True] = 100
    run = events.run

    evaluator = correctionlib.CorrectionSet.from_file('data/JetMETCorr/'+year+'_UL/met.json.gz')

    if isData:
        corrected_pt = evaluator['pt_metphicorr_pfmet_data'].evaluate(pt,phi,npv,run)
        corrected_phi = evaluator['phi_metphicorr_pfmet_data'].evaluate(pt,phi,npv,run)

    if isMC:
        corrected_pt = evaluator['pt_metphicorr_pfmet_mc'].evaluate(pt,phi,npv,run)
        corrected_phi = evaluator['phi_metphicorr_pfmet_mc'].evaluate(pt,phi,npv,run)

    return corrected_pt, corrected_phi


####
# Jet
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME
####

# Load CorrectionSet
#fname = "data/jsonpog/POG/JME/2017_EOY/2017_jmar.json.gz"
#if fname.endswith(".json.gz"):
#    import gzip
#    with gzip.open(fname,'rt') as file:
#        data = file.read().strip()
#        evaluator = _core.CorrectionSet.from_string(data)
#else:
#    evaluator = _core.CorrectionSet.from_file(fname)
#

###
# V+jets NLO k-factors
###

nlo_qcd_hists = {
    '2016':{
        'dy': uproot.open("data/vjets_SFs/merged_kfactors_zjets.root")["kfactor_monojet_qcd"],
        'w': uproot.open("data/vjets_SFs/merged_kfactors_wjets.root")["kfactor_monojet_qcd"],
        'z': uproot.open("data/vjets_SFs/merged_kfactors_zjets.root")["kfactor_monojet_qcd"],
        'a': uproot.open("data/vjets_SFs/merged_kfactors_gjets.root")["kfactor_monojet_qcd"]
    },
    '2017':{
        'z': uproot.open("data/vjets_SFs/SF_QCD_NLO_ZJetsToNuNu.root")["kfac_znn_filter"],
        'w': uproot.open("data/vjets_SFs/SF_QCD_NLO_WJetsToLNu.root")["wjet_dress_monojet"],
        'dy': uproot.open("data/vjets_SFs/SF_QCD_NLO_DYJetsToLL.root")["kfac_dy_filter"],
        'a': uproot.open("data/vjets_SFs/SF_QCD_NLO_GJets.root")["gjets_stat1_monojet"]
    },
    '2018':{
        'z': uproot.open("data/vjets_SFs/SF_QCD_NLO_ZJetsToNuNu.root")["kfac_znn_filter"],
        'w': uproot.open("data/vjets_SFs/SF_QCD_NLO_WJetsToLNu.root")["wjet_dress_monojet"],
        'dy': uproot.open("data/vjets_SFs/SF_QCD_NLO_DYJetsToLL.root")["kfac_dy_filter"],
        'a': uproot.open("data/vjets_SFs/SF_QCD_NLO_GJets.root")["gjets_stat1_monojet"]
    }
}
nlo_ewk_hists = {
    'dy': uproot.open("data/vjets_SFs/merged_kfactors_zjets.root")["kfactor_monojet_ewk"],
    'w': uproot.open("data/vjets_SFs/merged_kfactors_wjets.root")["kfactor_monojet_ewk"],
    'z': uproot.open("data/vjets_SFs/merged_kfactors_zjets.root")["kfactor_monojet_ewk"],
    'a': uproot.open("data/vjets_SFs/merged_kfactors_gjets.root")["kfactor_monojet_ewk"]
}    
get_nlo_qcd_weight = {}
get_nlo_ewk_weight = {}
for year in ['2016','2017','2018']:
    get_nlo_qcd_weight[year] = {}
    get_nlo_ewk_weight[year] = {}
    for p in ['dy','w','z','a']:
        get_nlo_qcd_weight[year][p] = lookup_tools.dense_lookup.dense_lookup(nlo_qcd_hists[year][p].values, nlo_qcd_hists[year][p].edges)
        get_nlo_ewk_weight[year][p] = lookup_tools.dense_lookup.dense_lookup(nlo_ewk_hists[p].values, nlo_ewk_hists[p].edges)



from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools.dense_lookup import dense_lookup

class BTagCorrector:

    def __init__(self, tagger, year, workingpoint):
        self._year = year
        common = load('data/common.coffea')
        self._wp = common['btagWPs'][tagger][year][workingpoint]
        
        btvjson = correctionlib.CorrectionSet.from_file('data/BtagSF/'+year+'_UL/btagging.json.gz')
        self.sf = btvjson # ["deepJet_comb", "deepCSV_comb"]

        files = {
            '2016': 'btageff2016.merged',
            '2017': 'btageff2017.merged',
            '2018': 'btageff2018.merged',
        }
        filename = 'hists/'+files[year]
        btag = load(filename)
        bpass = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag', 'pass').values()[()]
        ball = btag[tagger].integrate('dataset').integrate('wp',workingpoint).integrate('btag').values()[()]
        nom = bpass / np.maximum(ball, 1.)
        self.eff = lookup_tools.dense_lookup.dense_lookup(nom, [ax.edges() for ax in btag[tagger].axes()[3:]])

    def btag_weight(self, pt, eta, flavor, tag):
        abseta = abs(eta)
        
        #https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods#1b_Event_reweighting_using_scale
        def zerotag(eff):
            return (1 - eff).prod()
        '''
        Correction deepJet_comb has 5 inputs
        Input systematic (string): 
        Input working_point (string): L/M/T
        Input flavor (int): hadron flavor definition: 5=b, 4=c, 0=udsg
        Input abseta (real):
        Input pt (real):
        '''
        bc = flavor > 0
        light = ~bc
        
        eff = self.eff(flavor, pt, abseta)
        
        sf_nom = self.sf.eval('central', flavor, abseta, pt)
        sf_nom = self.sf["deepJet_comb"].evaluate('central','M', flavor, abseta, pt)
        
        bc_sf_up_correlated = pt.ones_like()
        bc_sf_up_correlated[~bc] = sf_nom[~bc]
        bc_sf_up_correlated[bc] = self.sf.eval('up_correlated', flavor, eta, pt)[bc]
        
        bc_sf_down_correlated = pt.ones_like()
        bc_sf_down_correlated[~bc] = sf_nom[~bc]
        bc_sf_down_correlated[bc] = self.sf.eval('down_correlated', flavor, eta, pt)[bc]

        bc_sf_up_uncorrelated = pt.ones_like()
        bc_sf_up_uncorrelated[~bc] = sf_nom[~bc]
        bc_sf_up_uncorrelated[bc] = self.sf.eval('up_uncorrelated', flavor, eta, pt)[bc]

        bc_sf_down_uncorrelated = pt.ones_like()
        bc_sf_down_uncorrelated[~bc] = sf_nom[~bc]
        bc_sf_down_uncorrelated[bc] = self.sf.eval('down_uncorrelated', flavor, eta, pt)[bc]

        light_sf_up_correlated = pt.ones_like()
        light_sf_up_correlated[~light] = sf_nom[~light]
        light_sf_up_correlated[light] = self.sf.eval('up_correlated', flavor, abseta, pt)[light]

        light_sf_down_correlated = pt.ones_like()
        light_sf_down_correlated[~light] = sf_nom[~light]
        light_sf_down_correlated[light] = self.sf.eval('down_correlated', flavor, abseta, pt)[light]

        light_sf_up_uncorrelated = pt.ones_like()
        light_sf_up_uncorrelated[~light] = sf_nom[~light]
        light_sf_up_uncorrelated[light] = self.sf.eval('up_uncorrelated', flavor, abseta, pt)[light]

        light_sf_down_uncorrelated = pt.ones_like()
        light_sf_down_uncorrelated[~light] = sf_nom[~light]
        light_sf_down_uncorrelated[light] = self.sf.eval('down_uncorrelated', flavor, abseta, pt)[light]

        eff_data_nom  = np.minimum(1., sf_nom*eff)
        bc_eff_data_up_correlated   = np.minimum(1., bc_sf_up_correlated*eff)
        bc_eff_data_down_correlated = np.minimum(1., bc_sf_down_correlated*eff)
        bc_eff_data_up_uncorrelated   = np.minimum(1., bc_sf_up_uncorrelated*eff)
        bc_eff_data_down_uncorrelated = np.minimum(1., bc_sf_down_uncorrelated*eff)
        light_eff_data_up_correlated   = np.minimum(1., light_sf_up_correlated*eff)
        light_eff_data_down_correlated = np.minimum(1., light_sf_down_correlated*eff)
        light_eff_data_up_uncorrelated   = np.minimum(1., light_sf_up_uncorrelated*eff)
        light_eff_data_down_uncorrelated = np.minimum(1., light_sf_down_uncorrelated*eff)
       
        nom = zerotag(eff_data_nom)/zerotag(eff)
        bc_up_correlated = zerotag(bc_eff_data_up_correlated)/zerotag(eff)
        bc_down_correlated = zerotag(bc_eff_data_down_correlated)/zerotag(eff)
        bc_up_uncorrelated = zerotag(bc_eff_data_up_uncorrelated)/zerotag(eff)
        bc_down_uncorrelated = zerotag(bc_eff_data_down_uncorrelated)/zerotag(eff)
        light_up_correlated = zerotag(light_eff_data_up_correlated)/zerotag(eff)
        light_down_correlated = zerotag(light_eff_data_down_correlated)/zerotag(eff)
        light_up_uncorrelated = zerotag(light_eff_data_up_uncorrelated)/zerotag(eff)
        light_down_uncorrelated = zerotag(light_eff_data_down_uncorrelated)/zerotag(eff)
        
        if '-1' in tag: 
            nom = (1 - zerotag(eff_data_nom)) / (1 - zerotag(eff))
            bc_up_correlated = (1 - zerotag(bc_eff_data_up_correlated)) / (1 - zerotag(eff))
            bc_down_correlated = (1 - zerotag(bc_eff_data_down_correlated)) / (1 - zerotag(eff))
            bc_up_uncorrelated = (1 - zerotag(bc_eff_data_up_uncorrelated)) / (1 - zerotag(eff))
            bc_down_uncorrelated = (1 - zerotag(bc_eff_data_down_uncorrelated)) / (1 - zerotag(eff))
            light_up_correlated = (1 - zerotag(light_eff_data_up_correlated)) / (1 - zerotag(eff))
            light_down_correlated = (1 - zerotag(light_eff_data_down_correlated)) / (1 - zerotag(eff))
            light_up_uncorrelated = (1 - zerotag(light_eff_data_up_uncorrelated)) / (1 - zerotag(eff))
            light_down_uncorrelated = (1 - zerotag(light_eff_data_down_uncorrelated)) / (1 - zerotag(eff))

        return np.nan_to_num(nom, nan=1.), np.nan_to_num(bc_up_correlated, nan=1.), np.nan_to_num(bc_down_correlated, nan=1.), np.nan_to_num(bc_up_uncorrelated, nan=1.), np.nan_to_num(bc_down_uncorrelated, nan=1.), np.nan_to_num(light_up_correlated, nan=1.), np.nan_to_num(light_down_correlated, nan=1.), np.nan_to_num(light_up_uncorrelated, nan=1.), np.nan_to_num(light_down_uncorrelated, nan=1.)




corrections = {}
corrections = {
    'XY_MET_Correction':        XY_MET_Correction,
    'get_ele_id_sf':            get_ele_id_sf,
    'get_pho_id_sf':            get_pho_id_sf,
    'pu_weight':                pu_weight,
}

ptjagged  = ak.Array([[10.1, 20.2, 30.3], [40.4, 50.5], [60.6]])
ptjaggeds = ak.Array([[30.1, 30.2, 40.3], [40.4, 50.5], [60.6]])
etajagged = ak.Array([[1.1, 1.2, 1.3], [1.4, 1.5], [1.6]])
phijagged = ak.Array([[1.2, 1.3, 1.4], [1.5, 1.6], [1.7]])

yr = '2018'
#print(corrections['get_ele_trig_weight'](yr, etajagged,ptjagged))
#print(corrections['get_pho_id_sf'](yr, etajagged,ptjaggeds))
#print(corrections['XY_MET_Correction'](yr, 100,1.2,100, 320395))
yr = '2017'
#print(corrections['get_ele_trig_weight'](yr, etajagged,ptjagged))
#print(corrections['get_pho_id_sf'](yr, etajagged,ptjaggeds))
yr = '2016postVFP'
#print(corrections['get_ele_trig_weight'](yr, etajagged,ptjagged))
#print(corrections['get_pho_id_sf'](yr, etajagged,ptjaggeds))
