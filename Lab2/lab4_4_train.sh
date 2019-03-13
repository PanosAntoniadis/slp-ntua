#!/bin/bash
source ./path.sh

# Train the monophone GMM-HMM acoutic model over
# the training data
echo "Training the monophone model"
steps/train_mono.sh data/train data/lang_test exp/mono0
# Align the monophone model
echo "Align the monophone model"
steps/align_si.sh data/train data/lang_test exp/mono0 exp/mono0_ali
