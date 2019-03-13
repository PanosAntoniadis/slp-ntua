# A simple script that creates a file wav.scp that contains
# in each line the path of the audio file that contains each sentence in
# the following form:   utterance_id_1 <κενό> path/to/wav1

import sys

data_file = sys.argv[1]
rf = open("data/" + data_file + "/utt2spk", 'r')
wf = open("data/" + data_file + "/wav.scp", 'w')
for i, line in enumerate(rf):
    line = line.split(' ')
    utt = line[0]
    spk = line[1].strip('\n')
    path = "slp_lab2_data/wav/" + spk + '/' + utt + ".wav"
    wf.write(line[0] + ' ' + path + '\n')
rf.close()
wf.close()
