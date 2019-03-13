#!/bin/bash
source ./path.sh

# 1. Create useful files for the language model.
echo "Creating useful files for the language model"
# silence_phones.txt and optional_silence.txt contain only
# the silence phonem.
cd data/local/dict
echo 'sil' > silence_phones.txt
echo 'sil' > optional_silence.txt
cd ../../..
# nonsilence_phones.txt contains all the remaining phonemes.
python create_nonsilence.py
# copy two times the contents of nonsilence_phones.txt separated by a space.
paste -d ' ' data/local/dict/nonsilence_phones.txt data/local/dict/nonsilence_phones.txt > data/local/dict/lexicon.txt
# add the phonem of silence and then sort them
echo 'sil sil' >> data/local/dict/lexicon.txt
sort -o data/local/dict/lexicon.txt{,}
# for each text file add a <s> and a </s> in the start and the end of the text.
python create_lm.py dev
python create_lm.py train
python create_lm.py test
# create an empty file named extra_questions.txt
touch data/local/dict/extra_questions.txt

# 2.Create a temporary form of the language model inside the folder data/local.lm_temp
# using the build-lm.sh command.
echo "Creating .ilm.gz files"
build-lm.sh -i data/local/dict/lm_dev.text -n 1 -o data/local/lm_tmp/lm_dev_unigram.ilm.gz
build-lm.sh -i data/local/dict/lm_dev.text -n 2 -o data/local/lm_tmp/lm_dev_bigram.ilm.gz

build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/lm_train_unigram.ilm.gz
build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/lm_train_bigram.ilm.gz

build-lm.sh -i data/local/dict/lm_test.text -n 1 -o data/local/lm_tmp/lm_test_unigram.ilm.gz
build-lm.sh -i data/local/dict/lm_test.text -n 2 -o data/local/lm_tmp/lm_test_bigram.ilm.gz

# 3. Create the ARPA form of the compiled language model using the compile-lm command.
echo "Creating ARPA files"
compile-lm data/local/lm_tmp/lm_dev_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_dev_unigram.arpa.gz
compile-lm data/local/lm_tmp/lm_dev_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_dev_bigram.arpa.gz

compile-lm data/local/lm_tmp/lm_train_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_train_unigram.arpa.gz
compile-lm data/local/lm_tmp/lm_train_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_train_bigram.arpa.gz

compile-lm data/local/lm_tmp/lm_test_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_test_unigram.arpa.gz
compile-lm data/local/lm_tmp/lm_test_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_test_bigram.arpa.gz

# 4. Create inside the data/lang folder the FST of the dictionary using
# the prepare_lang command
echo "Creating L.fst"
prepare_lang.sh data/local/dict "<oov>" data/local/lang data/lang

# 5. Create the FST of the grammar following the procedure of file timit of KALDI
./timit_format_data
