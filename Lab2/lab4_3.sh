#!/bin/bash
source ./path.sh

# Make spk2utt files that are necessary for the MFCC extraction
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

# Extract MFCC features.
for x in train test dev; do
        steps/make_mfcc.sh  --mfcc-config conf/mfcc.conf --cmd  "run.pl" \
        data/$x exp/make_mfcc/$x mfcc_${x}
        steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
done
