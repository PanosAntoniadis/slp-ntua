import sys

data_file = sys.argv[1]
rf = open("data/" + data_file + "/text", 'r')
sf = open("data/local/dict/lm_" + data_file + ".text", 'w')
for line in rf:
    text = line.split(' ', 1)[1].strip('\n')
    sf.write('<s> ' + text + '</s>' + '\n')
rf.close()
sf.close()
