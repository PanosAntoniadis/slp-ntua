# A simple script that convert the sentences in the text files
# into a sequence of phonemes using lexicon.txt file.
import sys
import re

# A function that creates a dictionary with the words of the
# lexicon.txt as keys and a list of phonemes as values.
def createDict():
    f = open("slp_lab2_data/lexicon.txt", 'r')
    dict = {}
    for line in f:
        words = line.split()
        dict[words[0].lower()] = words[1:]
    f.close()
    return dict

# A function that removes from a string all special characters
# except from single quotes.
def tokenize(s):
    return re.compile("[^\w']|_").sub(" ",s).strip()

dict = createDict()
data_file = sys.argv[1]
# File to read the text of each sentence,
rf = open("data/" + data_file + "/text", 'r')
# File to write the phonemes of the text of each sentence. Temporary
# file that will then be copied in text.
sf = open("data/" + data_file + "/text_temp", 'w')
for i, line in enumerate(rf):
    # Get uttid.
    id = line.split(" ", 1)[0]
    # Get the text.
    text = tokenize(line.split(" ", 1)[1])
    text = text.split()
    sf.write(id + ' ')
    sf.write('sil' + ' ')
    for word in text:
        word = word.lower()
        if word in dict:
            phonemes = dict[word]
            for phonem in phonemes:
                sf.write(phonem + ' ')
        else:
            print("Missing word " + word + ' in line ' + str(i) + '\n')
    sf.write('sil' + ' ' + '\n')
sf.close()
rf.close()
