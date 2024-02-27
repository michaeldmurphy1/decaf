<img src="https://user-images.githubusercontent.com/10731328/193421563-cf992d8b-8e5e-4530-9179-7dbd507d2e02.png" width="350"/>

# **D**ark matter **E**xperience with the **C**offea **A**nalysis **F**ramework

---

## Initial Setup

First, log into an LPC node:

```
ssh -L 9094:localhost:9094 <USERNAME>@cmslpc-sl7.fnal.gov
```

The command will also start forwarding the port 9094 (or whatever number you choose)to be able to use applications like jupyter once on the cluster. Then move into your `nobackup` area on `uscms_data`:

```
cd /uscms_data/d?/<USERNAME>
```

where '?' can be [1,2,3]. Install `CMSSW_11_3_4` (Note: `CMSSW 11_3_X` runs on slc7, which can be setup using apptainer on non-slc7 nodes ([see detailed instructions](https://cms-sw.github.io/singularity.html)):

```
#cmssw-el7 # uncomment this line if not on an slc7 node
cmsrel CMSSW_11_3_4
cd CMSSW_11_3_4/src
cmsenv
```

Install `combine` ([see detailed instructions](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/#installation-instructions)):

```
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v9.1.0 # current recommeneded tag (Jan 2024)
scramv1 b clean; scramv1 b # always make a clean build
```

Fork this repo on github and clone it into your `CMSSW_11_3_4/src` directory:

```
cd $CMSSW_BASE/src
git clone ttps://github.com/<USERNAME>/decaf.git
cd decaf
git switch UL
```

Then, setup the proper dependences:

```
source setup.sh
```

This script installs the necessary packages as user packages (Note: Pip gave errors when running `setup.sh` for the first time, but it seemed to install everything just fine. No errors showed up when running `setup.sh` a second time.). This is a one-time setup. When you log in next just do:

```
#cmssw-el7 # uncomment this line if not on an slc7 node
cd CMSSW_11_3_4/src
cmsenv
cd decaf
source env.sh
```

By running this script you will also initialize your grid certificate (Note: `setup.sh` also runs `env.sh`). This requires you to save your grid certificate password in `$HOME/private/$USER.txt`. Alternatively, you can comment this out and initialize it manually every time.

---

## Listing Input Files

The list of input files for the analyzer can be generated as a JSON file using the `macros/list.py` script. This script will run over the datasets listed in `data/process.py`, find the list of files for each dataset, “pack” them into small groups for condor jobs, and output the list of groups as a JSON file in `metadata/`.

The options for this script are:

- `-d` (`--dataset`)

Select a specific dataset to pack. By default, it will run over all datasets in `process.py`.

- `-y` (`--year`)

Data year. Options are `2016pre`, `2016post`, `2017`, and `2018`.

- `-m` (`--metadata`)

Name of metadata output file. Output will be saved in `metadata/<NAME>.json`

- `-p` (`--pack`)

Size of file groups. The smaller the number, the more condor jobs will run. The larger the number, the longer each condor job will take. We tend to pick `32`, but the decision is mostly arbitrary.

- `-s` (`--special`)

Size of file groups for special datasets. For a specific dataset, use a different size with respect to the one established with `--pack`. The syntax is `-s <DATASET>:<NUMBER>`.

- `-c` (`--custom`)

Boolean to decide to use public central NanoAODs (if `False`) or private custom NanoAODs (if `True`). Default is `False`.

As an example, to generate the JSON file for all 2017 data:

```
python macros/list.py -y 2017 -m 2017 -p 32
```

As a reminder, this script assumes that you are in the `decaf/analysis` directory when running. The output above will be saved in `metadata/2017.json`.

If using the `--custom` option, the script can take several hours to run, so it’s best to use a process manager such as `nohup` or `tmux` to avoid the program crashing in case of a lost connection. For example

```
nohup python macros/list.py -y 2017 -m 2017 -p 32 -c &
```

The `&` option at the end of the command lets it run in the background, and the std output and error is saved in `nohup.out`. 

The `nohup` command is useful and recommended for running most scripts, but you may also use tools like `tmux` or `screen`.

---

This README is a work in progress
