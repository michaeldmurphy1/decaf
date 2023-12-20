# https://lcginfo.cern.ch/release_packages/104c/x86_64-centos7-gcc11-opt/
# Configured for Python 3.9 on CC7
if uname -r | grep -q el7; then
  source /cvmfs/sft.cern.ch/lcg/views/LCG_104c/x86_64-centos7-gcc11-opt/setup.sh
else
  echo "Cannot source LCG setup.sh file. Only configured for Python 3.9 on CC7"
fi

export PYTHONPATH=~/.local/lib/python3.9/site-packages:$PYTHONPATH
export PYTHONPATH=$PWD/analysis:$PYTHONPATH
export PYTHONWARNINGS="ignore"
#export PATH=~/.local/bin:$PATH
cat $HOME/private/$USER.txt | voms-proxy-init -voms cms --valid 140:00 -pwstdin
