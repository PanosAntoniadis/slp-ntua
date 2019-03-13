#!/bin/bash

# Create files that will contain pointer-files and will describe
# the train, dev and test data accordingly.
mkdir data
mkdir data/train && mkdir data/dev && mkdir data/test

# Copy utterance ids into a new file named uttids in the corresponding folder.
cat slp_lab2_data/filesets/train_utterances.txt > data/train/uttids
cat slp_lab2_data/filesets/validation_utterances.txt > data/dev/uttids
cat slp_lab2_data/filesets/test_utterances.txt > data/test/uttids
echo "uttids files created successfully"

# Create utt2spk file for each of the three folders.
python utt2spk.py train
python utt2spk.py dev
python utt2spk.py test
echo "utt2spk files created successfully"

# Create wav.scp file for each of the tree folders.
python wav-scp.py train
python wav-scp.py dev
python wav-scp.py test
echo "wav.scp files created successfully"

# Create text file for each of the tree folders.
python text.py train
python text.py dev
python text.py test
echo "text files created successfully"

# Create temp text files with phonemes.
python text2phonem.py train
python text2phonem.py dev
python text2phonem.py test
# Then substitute old text files by text_temp.
mv data/train/text_temp data/train/text
mv data/dev/text_temp data/dev/text
mv data/test/text_temp data/test/text
echo "text files changed to phonemes successfully"
