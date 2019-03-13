#!/bin/bash
source ./path.sh

# 2. Create soft links for 'steps' and 'utils' files
ln -s ../wsj/s5/steps steps
ln -s ../wsj/s5/utils utils
echo "Soft links created successfully"

# 3. Create a folder named 'local'. Inside it create
# a soft link to score_kaldi.sh.
mkdir local
cd local
ln -s ../steps/score_kaldi.sh score_kaldi.sh
cd ..
echo "Local created successfully"

# 4. Create a folder named 'conf'. Inside it copy
# mfcc.conf file.
mkdir conf
cp lab2_help_scripts/mfcc.conf conf/mfcc.conf
echo "Conf created successfully"

# 5. Create some useful folders
mkdir data/lang
mkdir data/local
mkdir data/local/dict
mkdir data/local/lm_tmp
mkdir data/local/nist_lm
echo "All done successfully"
