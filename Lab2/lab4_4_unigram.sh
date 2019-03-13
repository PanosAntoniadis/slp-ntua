#!/bin/bash
source ./path.sh

# Rename G_train_unigram.fst to G.fst in order to
# run the mkgraph.sh which searches for an G.fst.
cp data/lang_test/G_train_unigram.fst data/lang_test/G.fst
# Make HCLG graph.
utils/mkgraph.sh --mono data/lang_test exp/mono0 exp/mono0/graph_nosp_tgpr

# Decode for the dev set.
steps/decode.sh exp/mono0/graph_nosp_tgpr data/dev exp/mono0/decode_dev
# Print PER.
[ -d exp/mono0/decode_dev ] && grep WER exp/mono0/decode_dev/wer_* | utils/best_wer.sh
# Decode for the test set.
steps/decode.sh exp/mono0/graph_nosp_tgpr data/test exp/mono0/decode_test
# Print PER.
[ -d exp/mono0/decode_test ] && grep WER exp/mono0/decode_test/wer_* | utils/best_wer.sh
