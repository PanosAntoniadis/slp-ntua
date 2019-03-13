#!/bin/bash
source ./path.sh

# Dimension of characteristics
feat-to-dim ark:mfcc_train/raw_mfcc_train.1.ark -

# Number of frames per sentence
feat-to-len scp:data/train/feats.scp ark,t:data/train/feats.lengths
# Print first 5 sentences.
head -5 data/train/feats.lengths
