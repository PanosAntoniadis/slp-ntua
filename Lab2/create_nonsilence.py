# Creates the file nonsilence_phones.txt which contains
# all the phonemes except sil.
source = open("slp_lab2_data/lexicon.txt", 'r')
phones = []
# Get all the separate phonemes from lexicon.txt.
for line in source:
    line_phones = line.split(' ')[1:]
    for phone in line_phones:
        phone = phone.strip(' ')
        phone = phone.strip('\n')
        if phone not in phones and phone!='sil':
            phones.append(phone)
source.close()
phones.sort()
# Write phonemes to the file.
wf = open("data/local/dict/nonsilence_phones.txt", 'w')
for x in phones:
    wf.write(x+'\n')
wf.close()
