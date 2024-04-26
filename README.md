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
python3 macros/list.py -y 2017 -m 2017 -p 32
```

As a reminder, this script assumes that you are in the `decaf/analysis` directory when running. The output above will be saved in `metadata/2017.json`.

If using the `--custom` option, the script can take several hours to run, so it’s best to use a process manager such as `nohup` or `tmux` to avoid the program crashing in case of a lost connection. For example

```
nohup python3 macros/list.py -y 2017 -m 2017 -p 32 -c &
```

The `&` option at the end of the command lets it run in the background, and the std output and error is saved in `nohup.out`. 

The `nohup` command is useful and recommended for running most scripts, but you may also use tools like `tmux` or `screen`.

---

## Computing MC b-Tagging Efficiencies

MC b-tagging efficiencies are needed by most of the analyses to compute the b-tag event weight, once such efficiencies are corrected with the POG-provided b-tag SFs. To compute them, we first need to run the `common`, `ids`, and `corrections` modules in `util`:

```
python3 utils/common.py
python3 utils/ids.py
python3 utils/corrections.py
```

This will generate a series of auxiliary functions and information, like the AK4 b-tagging working points, and it will save such information in  `.coffea` files in the `data` folder (really all pickle files). AK4 b-tagging working points are essential to measure the MC efficiencies and they are used by the `btag` processor in the `processors` folder. All three output files are used as input when creating the processor. To generate the analysis processor file: 

```
python3 processors/btageff.py -y 2018 -m 2018 -n 2018
```

The options for this script are:

- `-y` (`--year`)

Data year. Options are `2016pre`, `2016post`, `2017`, and `2018`.

- `-m` (`--metadata`)

Metadata file to be used in input (filename sans extentions found in `metadata` folder).

- `-n` (`--name`)

Name of the output processor file (to be appended to processor name). In this case, it will generate a file called `btageff2018.processor` stored in the `data` folder.


To execute the processor, you `run` it over the input data:

```
python3 run.py -p btageff2018 -m 2018 -d QCD
```

The options for this script:

- `-p` (`--processor`)

Name of the processor file stored in the `data` folder (without `.processor` extension)

- `-m` (`--metadata`)

The metadata file to use as input (without `json.gz` extension)

- `-d` (`--dataset`)

Optional, the name of the dataset to use as input data. Specifying will run over all datasets with the passed string in its name. Without, it will run over all datasets (in series, taking a long time)


With this command you will run the `btag2018` processor over QCD MC datasets as defined by the `2018` metadata file. You will see a printout like:

```
Processing: QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8____4_
  Preprocessing 100% ━━━━━━━━━━━━━━━━━━ 32/32 [ 0:01:28 < 0:00:00 | ?   file/s ]
  Merging (local) 100% ━━━━━━━━━━━━━━━━ 31/31 [ 0:00:23 < 0:00:00 | ? merges/s ]
```
This means an output file with histograms as defined in the btag processor file has been generated. In this case a folder called `btageff2018` inside the `hists` folder has been created. Inside this folder you can see a file called `QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8____4_.futures`, that stores the histograms. 

In practice, you don't want to run the script serially. To take advantage of the parallelism offered by the HTCondor job scheduler, the `run_condor.py` script can be used:

```
python3 run_condor.py -p btag2018 -m 2018 -d QCD -c kisti -t -x
```

> Note: every user needs to insert their own certificate in `run_condor.py`, about 50 lines in on the line `Transfer_Input_files = run.sh, /tmp/x509up_uXXXXXXXXX`.

The options for this script are the same as for `run.py` (`-p`, `-m`, `-d`), with the addition of:

- `-c` (`--cluster`)

Specifies which cluster you are using.  At the moment supports `lpc` (default) or `kisti`.

- `-t` (`--tar`)
  
Tars the local python environment and the local CMSSW folder. 

- `-x` (`--copy`)

Copies these two tarballs to your EOS area. 

You don't need to create and copy the tarball every time. For example, to run the same setup but for a different year you won’t need to tar and copy again. You can simply do: `python3 run_condor.py -p btag2017 -m 2017 -d QCD -c kisti`, assuming the 2017 processor was already created when you created the tarball with `-t -x` to launch the 2018 jobs.

You can check the status of your HTCondor jobs by doing:

```
condor_q <YOUR_USERNAME>
```
This will indicate if the jobs are idle, running, or held. Any jobs on hold have encountered a problem and need debugging.

---
### Debugging Condor errors
The output of any condor-run script will be held in `logs/condor/SCRIPT_NAME`. For example, after calling the above `python3 run_condor.py ... `, there will be three folders within the `logs/condor/run/` directory:  `log/`, `err/`,and  `out/`. Any file that is read by a condor-run script will have its output directed to a `log/`-located file, while stderr is sent to `err/` and environment details are send to `out/`. __<- Comment is this true? I'm not sure what `out/` does__


#### Held Job

If a job is marked on hold, that means it encountered an error and is waiting around to be stopped now. 

First, to identify the error, you can copy the `JOB_ID` and use it to find the log file: `ls logs/condor/run/log/*JOB_ID*`. This gets you the process name and the batch number. From there, identify the problem from the contents of the log. (Or simply run `emacs -nw $( ls logs/condor/run/log/*JOB_ID* )` )

Once the problem is fixed, you must remove the held job:
```
condor_rm <JOB_ID>
```
Alternatively, if you want to just clear every job, call `condor_rm <YOUR_USERNAME>`.

Finally, you can resubmit the job using the `-d` option on either `run.py` (local) or `run_condor.py` (condor), specifying the specific job by passing the name and number as an argument (`python3 run.py -m 2018  -d specific_process_info-pythia8____2_`, with the `____N_` being the batch number and the `processorYear_` prefix removed) or running all jobs related to that process (`specific_process_info-pythia8`)

For any script that resulted in an error, it is a good idea to run it locally, either before or after fixing, to verify you can either recreate the error (e.g. was it a network issue) or to see that you actually fixed it.

To see all errors, use `grep`:  `grep '[Ee]rror' logs/condor/run/err/*`

You know if a job is successful by its output being stored in `hists/ProccessorYear/process.futures`

---

After obtaining all the histograms, a first step of data reduction is needed. This step is achieved by running the `reduce.py` script:

```
python3 reduce.py -f hists/btag2018
```

The options of this script are:

- `-f` (`--folder`)

The folder that holds your `.future` files (usually `hists/processorName`).

- `-d` (`--dataset`)

Optional, the name of the dataset to use as input data. Specifying will run over all datasets with the passed string in its name.

- `-e` (`--exclude`)

Optional, string of the names of datasets (separated by commas) that you wish not be run. __Comment: CHECK__

- `-v` (`--variable`)

Optional, a single metadata variable to include in the analysis (disregarding all others). __Comment: CHECK__


All the different datasets produced at the previous step will be reduced. A different file for each variable for each reduced dataset will be produced. For example, the command above will produce the following reduced files:

```
hists/btageff2018/deepcsv--QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8.reduced
hists/btageff2018/deepcsv--QCD_Pt_120to170_TuneCP5_13TeV_pythia8.reduced
...
hists/btageff2018/deepflav--QCD_Pt_800to1000_TuneCP5_13TeV_pythia8.reduced
hists/btageff2018/deepflav--QCD_Pt_80to120_TuneCP5_13TeV_pythia8.reduced
```

This step can be run in HTCondor by using the `reduce_condor.py` script. The `reduce_condor.py` script has the same options of `reduce.py`, with addition of the same `--cluster`, `--tar`, and `--copy` options descibed above when discussing `run_condor.py`.

A second step of data reduction is needed to merge all the `.reduced` files corresponding to a single variable. This is achieved by using the `merge.py` script:

```
python3 merge.py -f hists/btageff2018
```

The options of this script are:

- `-f` (`--folder`)

The folder that holds your `.future` files (usually `hists/processorName`).

- `-v` (`--variable`)

Optional, a single metadata variable to include in the analysis (disregarding all others). __Comment: CHECK__

- `-e` (`--exclude`)

Optional, string of the names of datasets (separated by commas) that you wish not be run. __Comment: CHECK__

Also this step can be run in HTCondor by using the `merge_condor.py` script. The `merge_condor.py` script has the same options of `merge.py`, with addition of the same `--cluster`, `--tar`, and `--copy` options descibed above when discussing `run_condor.py`.

This command will produce the following files:

```
hists/btageff2018/deepcsv.merged  hists/btageff2018/deepflav.merged
```

The same script is then used to merge the the files corresponding to each single variable into a single file, using the `-p` or `—postprocess` option:

```
python3 merge.py -f hists/btageff2018 -p
```

This will create a `hists/btageff2018.merged` file that is penultimate to the final result. The last step is to run `macros/scale.py` will extract the histograms from the `.merged` file and scale values by their cross section:

```
python3 macros/scale.py -f hists/hadmonotop2018.merged
```

The option of this script is `-f` (`--folder`), as seen before, where your `processorName.merged` file is located. The output is stored as `hists/processorName.scaled`

---

This README is a work in progress

