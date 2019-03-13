# A simple script that creates a file text that contains
# in each line the text of the audio file that corresponds to each sentence in
# the following form:   utterance_id_1 <κενό> <utterance 1 text>

import sys

data_file = sys.argv[1]
rf = open("data/" + data_file + "/uttids", 'r')
tf = open("slp_lab2_data/transcription.txt", 'r')
texts = tf.readlines()
wf = open("data/" + data_file + "/text", 'w')
for i, line in enumerate(rf):
    number = int(line[16:19])
    wf.write(line[:19] + ' ' + texts[number-1])
rf.close()
tf.close()
wf.close()
