#! /usr/bin/env python
import correctionlib
import os
import awkward as ak

####
# Electron
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgammaSFJSON
# jsonPOG: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/EGM
####

def get_ele_trig_weight (year, eta, pt):
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
# ?


####
# PU weight
####
year = '2016preVFP'
evaluator = correctionlib.CorrectionSet.from_file('data/PUweight/'+year+'_UL/puWeights.json.gz')
for corr in evaluator.values():
    print(f"Correction {corr.name} has {len(corr.inputs)} inputs")
    for ix in corr.inputs:
        print(f"   Input {ix.name} ({ix.type}): {ix.description}")

#events.Pileup.nTrueInt

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

corrections = {}
corrections = {
    'XY_MET_Correction':        XY_MET_Correction,
    'get_ele_trig_weight':      get_ele_trig_weight,
    'get_pho_id_sf':            get_pho_id_sf,
}

ptjagged =  ak.Array([[10.1, 20.2, 30.3], [40.4, 50.5], [60.6]])
ptjaggeds =  ak.Array([[30.1, 30.2, 40.3], [40.4, 50.5], [60.6]])
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
