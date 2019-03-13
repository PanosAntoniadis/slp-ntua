# A simple script that creates a file utt2spk that contains
# in each line the speaker that contains in each sentence in
# the following form:   utterance_id_1 <κενό> speaker_id
# where speaker_id is chosen between m1, m3, f1 and f5.

import sys

data_file = sys.argv[1]
rf = open("data/" + data_file + "/uttids", 'r')
wf = open("data/" + data_file + "/utt2spk", 'w')
for i, line in enumerate(rf):
    if line[13:15] in ['m1', 'm3', 'f1', 'f5']:
        wf.write(line[:19] + ' ' + line[13:15] + '\n')
    else:
        print("Error in line " + str(i))
rf.close()
wf.close()
