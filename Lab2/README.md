# Implementation of a phone recognition system using the KALDI toolkit

## The design of the system can be divided in 4 steps

- Extraction of Mel-Frequency Cepstral Coefficients.
- Language model training.
- Acoustic model training.
- Combination of above models and evaluation.

## Dataset 

The dataset used includes recordings from four speakers from USC-timit database . Each speaker corresponds to 460 sentences resulting in one .wav file per sentence along the transciption. 

## Table of contents

- __report.pdf:__ The report of the project where all the procedure of the design is described and the evaluation of the model takes place.

- __utt2spk.py, wav-scp.py, text.py, text2phonem.py:__ All these files create text files that Kaldi needs in order to train language and acoustic model.

- __cmd.sh, path.sh:__ Kaldi scripts that should be placed in the project folder.

- __lab2_help_scripts/mfcc.conf:__ Contains the sample frequency of the audio data.

- __create_lm.py:__ Creates lm_train.text, lm_dev.text and lm_test.text that will be used for the implementation of the language model.

- __create_nonsilence.py:__ Create nonsilence_phones.txt that contains all the non silence phones of the corpus. 

- __perplexity.sh:__ Computes the perplexity of the language model.

- __Remaining scripts are based on the steps in the probled-description:__
  - PROLAB: run ./prolab.sh
  - LAB 4.1: run ./lab4_1.sh
  - LAB 4.2: run ./lab4_2.sh
  - LAB 4.3: run ./lab4_3.sh and ./lab4_3_questions2.sh
  - LAB 4.4: run ./lab4_4_train.sh, ./lab4_4_unigram.sh and ./lab4_4_bigram.sh
