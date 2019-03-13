
# coding: utf-8

# # 1η εργαστηριακή άσκηση: Εισαγωγή στις γλωσσικές αναπαραστάσεις

# <h2><center> Περιγραφή </center></h2>

# __Σκοπός__ αυτού του μέρους της 1ης εργαστηριακής άσκησης είναι να γίνει μια εισαγωγή σε διαφορετικές γλωσσικές αναπαραστάσεις και τη χρήση τους για γλωσσικά tasks. Στο πρώτο μέρος θα εμπλουτίσουμε τον ορθογράφο που φτιάξαμε στην προπαρασκευή με character level και word level unigram γλωσσικά μοντέλα. Στο δεύτερο μέρος θα κάνουμε μια εισαγωγή στις λεξικές αναπαραστάσεις bag-of-words και word2vec και θα τις χρησιμοποιήσουμε σε ένα απλό πρόβλημα ταξινόμησης.

# <h2><center> Μέρος 1: Ορθογράφος </h2></center>

# Αρχικά κατεβάζουμε το corpus που θα χρησιμοποιήσουμε. Θα ασχοληθούμε με το βιβλίο __War of the Worlds__ όπως και στην προπαρασκευή έτσι ώστε να μπορούμε να συγκρίνουμε τα αποτελέσματα πάνω στο ίδιο corpus. Με την παρακάτω εντολή, λοιπόν, το κατεβάζουμε από το project Gutenberg σε plain txt μορφή και το αποθηκεύουμε με το όνομα __War.txt__.

# In[1]:


get_ipython().system(' wget  -c http://www.gutenberg.org/files/36/36-0.txt -O War.txt')


# ### Βήμα 10: Εξαγωγή στατιστικών

# Στο βήμα αυτό θα κατασκευάσουμε 2 πηγές στατιστικών για τα γλωσσικά μας μοντέλα, μία __word/token level__ και μία __character level__.

# Για το βήμα αυτό αλλά και για την συνέχεια της άσκησης θα χρειαστούμε ορισμένες συναρτήσεις που υλοποιήθηκαν στην προπαρασκευή και μας βοηθάνε στην επεξεργασία του corpus. Συγκεκριμένα έχουμε τις εξής συναρτήσεις (η περιγραφή της λειτουργίας τους βρίσκεται στην προπαρασκευή):

#  __1. identity_preprocess:__ 

# In[2]:


# Gets a string as input and just returns the same string.
def identity_preprocess(string_var):
    return string_var


#  __2. read_path:__

# In[3]:


# Reads a file tokenizing each line.
def read_path(file_path, preprocess = identity_preprocess):
    # Initilize the list of processed lines
    processed_lines = []
    # Open file to read mode
    with open(file_path, "r") as f:
        for line in f:
            # Omit spaces
            if not line.isspace():
                processed_lines.extend(preprocess(line))
    return processed_lines


#  __3. tokenize:__

# In[4]:


import string 
# Tokenize a sttring
def tokenize(s):
    # Remove possible spaces from the start or the end of the string and
    # turn all letters lowercase.
    s = s.strip().lower()
    # Remove all punctuations, symbols and numbers from the string leaving
    # only lowercase alphabetical letters.
    s = "".join((char for char in s if char not in string.punctuation and not char.isdigit()))
    # Replace new line characters with spaces
    s = s.replace('\n',' ')
    # Split the string in every space resulting in a list of tokens
    res = s.split(" ")
    return res


#  __4. get_tokens:__

# In[5]:


# Get all separate tokens from a file.
def get_tokens(file_path):
    tokens = read_path(file_path, tokenize)
    distinct_tokens = list(dict.fromkeys(tokens))
    return distinct_tokens


#  __5. get_alphabet:__

# In[6]:


# Get the alphabet of a file given its tokens.
def get_alphabet(tokens):
    alphabet = []
    for token in tokens:
        alphabet.extend(list(token))
    alphabet = list(dict.fromkeys(alphabet))
    return alphabet


# Τώρα, λοιπόν, που έχουμε ορίσει τις συναρτήσεις που χρειαζόμαστε από την προπαρασκευή μπορούμε να συνεχίσουμε κανονικά στο βήμα 10.

# __α) token level:__ Πρέπει να εξάγουμε την πιθανότητα εμφάνισης κάθε token (λέξης) του βιβλίου και να την αποθηκεύσουμε σε ένα λεξικό με __key το token και value την πιθανότητα εμφάνισής του__. 

# __Διαδικασία: __
# - Θα φτιάξουμε μία συνάρτηση η οποία θα δέχεται ως όρισμα το path του corpus και θα επιστρέφει το ζητούμενο λεξικό. Αρχικά, θα αποθηκεύει σε μία λίστα όλα τα tokens χρησιμοποιώντας την συνάρτηση get_tokens και θα αρχικοποιεί το λεξικό μας με αυτά τα tokens ως keys και με value ίσο με 0. Στη συνέχεια, για κάθε λέξη του corpus θα αυξάνουμε το αντίστοιχο value στο λεξικό μας. Έτσι αφού διαιρέσουμε και κάθε value με τον αριθμό όλων των λέξεων του βιβλίου (για να μετατραπεί σε μία πιθανότητα) θα έχουμε δημιουργήσει το ζητούμενο λεξικό.

# In[7]:


def token_level(path):
    # Keys of the dictionary are all discrete tokens.
    keys = get_tokens(path)
    # Initialize the dictionary with the above keys and all values equal to 0.
    dict_token = dict.fromkeys(keys, 0)
    # Get a list with all the words containing in the corpus.
    words = read_path(path, tokenize)
    # For each word increase the value of the corresponding key.
    for word in words:
        dict_token[word] += 1
    # Divide each value with the total number of words to get the probability of each key.
    dict_token = {k: v / len(words) for k, v in dict_token.items()}
    return dict_token


# - Καλούμε την συνάρτηση που ορίσαμε παραπάνω και αποθηκεύουμε το λεξικό μας ως __dict_token__.

# In[8]:


# Get the dictionary of the frequency of each token.
dict_token = token_level("War.txt")


# __β) character level:__  Εδώ πρέπει να εξάγουμε την πιθανότητα εμφάνισης κάθε χαρακτήρα του corpus και, αντίστοιχα με πριν, να την αποθηκεύσουμε σε ένα λεξικό με key τον χαρακτήρα και value την πιθανότητα εμφάνισής του.

# __Διαδικασία:__ 
# - Αντίστοιχα λοιπόν παραπάνω θα φτιάξουμε μία παρόμοια συνάρτηση, η οποία αυτή τη φορά θα κάνει την ίδια διαδικασία για κάθε χαρακτήρα του corpus αντί για κάθε λέξη. Εδώ θα χρησιμοποιηθεί η συνάρτηση get_alphabet η οποία θα μας δώσει τα keys του λεξικού μας. Τα values θα υπολογιστούν διατρέχοντας μία φορά όλα το βιβλίο και αυξάνοντας κάθε φορά κατά 1 το value του που αντιστοιχεί στον χαρακτήρα που συναντάμε. Τέλος, πρέπει να διαιρέσουμε με όλους τους εμφανιζόμενους χαρακτήρες.

# In[9]:


def character_level(path):
    # Keys of the dictionary are the alphabet of the corpus.
    keys = get_alphabet(get_tokens(path))
    # Initialize the dictionary with the above keys and all values equal to 0.
    dict_character = dict.fromkeys(keys, 0)
    # Get a list with all the words containing in the corpus.
    words = read_path(path, tokenize)
    # Counter that will keep track of all the characters in the corpus.
    total = 0
    # For each letter of each word increase the corresponding value.
    for word in words:
        for char in list(word):
            total += 1
            dict_character[char] += 1
    # Divide each value with the total number of characters to get the probability of each key.
    dict_character = {k: v / total for k, v in dict_character.items()}
    return dict_character


# Καλούμε την συνάρτηση που ορίσαμε παραπάνω και αποθηκεύουμε το λεξικό μας ως __dict_character__.

# In[10]:


dict_character = character_level("War.txt")


# Ολοκληρώνοντας, λοιπόν, το βήμα 10 έχουμε δύο λεξικά που αποτελούν τις πηγές στατιστικών για τα γλωσσικά μας μοντέλα, ένα word/token level και ένα character level.

# ### Βήμα 11: Κατασκευή μετατροπέων FST

# Για τη δημιουργία του ορθογράφου θα χρησιμοποιήσουμε μετατροπείς βασισμένους στην απόσταση Levenshtein. Θα χρησιμοποιήσουμε 3 τύπους από edits κάθε ένα από τα οποία χαρακτηρίζεται από ένα κόστος. Έχουμε: 
#  - __εισαγωγές χαρακτήρων__ 
#  - __διαγραφές χαρακτήρων__
#  - __αντικαταστάσεις χαρακτήρων__

# __α)__ Στο βήμα αυτό θα υπολογίσουμε την μέση τιμή των βαρών του word level μοντέλου που κατασκευάσαμε στο βήμα 10α, το οποίο θα αποτελεί το κόστος w των edits. Συγκεκριμένα, αφού έχουμε την πιθανότητα εμφάνισης κάθε λέξης, το βάρος της ορίζεται ως ο αρνητικός λογάριθμος της πιθανότητας εμφάνισής της, δηλαδή __w = -log(P)__. Υπολογίζοντας, λοιπόν, το βάρος κάθε λέξης και παίρνοντας την μέση τιμή όλων των βαρών έχουμε το κόστος w, το οποίο επειδή προκύπτει από το token level μοντέλο το ονομάζουμε __w_token__.

# In[11]:


from math import log10

# Calculate weight of each word.
token_weights = {k:(-log10(v)) for k,v in dict_token.items()}
# Get the mean value of weigths.
w_token = sum(token_weights.values()) / len(token_weights.values())


# __β)__ Στο βήμα αυτό θα κατασκευάσουμε τον μετατροπέα μας με μία κατάσταση που υλοποιεί την απόσταση Levenshtein αντιστοιχίζοντας:
# - Kάθε χαρακτήρα στον εαυτό του με βάρος 0 __(no edit)__.
# - Kάθε χαρακτήρα στο <epsilon\> (ε) με βάρος w __(deletion)__.
# - Tο <epsilon\> σε κάθε χαρακτήρα με βάρος w __(insertion)__.
# - Kάθε χαρακτήρα σε κάθε άλλο χαρακτήρα με βάρος w __(substitution)__.

# Όπως και στην προπαρασκευή θα ορίσουμε την συνάρτηση format_arc η οποία διαμορφώνει μία γραμμή του αρχείου περιγραφής του κάθε FST. Συγκεκριμένα δέχεται ως όρισμα τα __src__, __dest__, __ilabel__, __olabel__ και το __weight__ (με default τιμή το 0) και τα επιστρέφει στην κατάλληλη μορφή όπως αναφέρεται και εδώ http://www.openfst.org/twiki/bin/view/FST/FstQuickTour#CreatingFsts/.

# In[12]:


def format_arc(src, dest, ilabel, olabel, weight=0):
    return (str(src) + " " + str(dest) + " " + str(ilabel) + " " + str(olabel) + " " + str(weight))


# Ακόμη, από την στιγμή που θα κατασκευάσουμε ορισμένα FSTs θα χρειαστούμε ένα αρχείο __chars.syms__ το οποίο θα αντιστοιχίζει κάθε χαρακτήρα του αλφαβήτου με έναν αύξοντα ακέραιο αριθμό. Η διαδικασία αυτή έγινε στο βήμα 4 της προπαρασκευής και περιλαμβάνει την συνάρτηση alphabet_to_int όπως βλέπουμε και παρακάτω:

# In[13]:


def alphabet_to_int(alphabet):
    # Open file
    f = open("chars.syms", "w")
    # Match epsilon to 0
    f.write("EPS" + 7*" " + str(0) + '\n')
    num = 21
    for character in alphabet:
        # Match every other character to an increasing index
        f.write(character + 7*" " + str(num) + '\n')
        num += 1
    f.close()


# In[14]:


alphabet_to_int(get_alphabet(get_tokens("War.txt")))


# Στη συνέχεια, διαμορφώνουμε το αρχείο περιγραφής του μετατροπεά μας σύμφωνα με τις παραπάνω αντιστοιχίσεις. Το αποτέλεσμα αποθηκεύεται στο αρχείο __transducer_token.fst__ (συμβολίζουμε το (ε) με "EPS").

# In[15]:


# Get alphabet of the corpus
alphabet = get_alphabet(get_tokens("War.txt"))
# Open file to write mode
f = open("transducer_token.fst", "w")
for letter in alphabet:
    # no edit
    f.write(format_arc(0, 0, letter, letter) + "\n")
    # deletion
    f.write(format_arc(0, 0, letter, "EPS", w_token) + "\n")
    # insertion
    f.write(format_arc(0, 0, "EPS", letter, w_token) + "\n")
for i in range(len(alphabet)):
    for j in range(len(alphabet)):
        if i != j:
            # substitution
            f.write(format_arc(0, 0, alphabet[i], alphabet[j], w_token) + "\n")

# Make initial state also final state
f.write("0")
# Close file
f.close()


# Αντίστοιχα με την προπαρασκευή τρέχουμε το παρακάτω shell command που κάνει compile τον μετατροπέα μας. Το binary αρχείο που προκύπτει με όνομα __transducer_token.fst__ είναι αυτό που θα χρησιμοποιήσουμε στις επόμενες λειτουργίες.

# In[16]:


get_ipython().system(' fstcompile --isymbols=chars.syms --osymbols=chars.syms transducer_token.fst transducer_token.fst')


# __γ)__ Τώρα θα επαναλάβουμε την ίδια διαδικασία χρησιμοποιώντας το unigram γλωσσικό μοντέλο του βήματος 10β. Θα υπολογίσουμε αρχικά το νέο κόστος των edit το οποίο ισούται με τη μέση τιμή των βαρών του character level μοντέλου και στη συνέχεια θα γράψουμε στο αρχείο __transducer_char.fst__ την περιγραφή του μετατροπέα που θα χρησιμοποιεί το μοντέλο αυτό.

# In[17]:


# Calculate weight of each character.
character_weigths = {k: (-log10(v)) for k,v in dict_character.items()}
# Get the mean value of weigths.
w_char = sum(character_weigths.values()) / len(character_weigths.values())


# In[18]:


# Open file to write mode
f = open("transducer_char.fst", "w")
for letter in alphabet:
    # no edit
    f.write(format_arc(0, 0, letter, letter) + "\n")
    # deletion
    f.write(format_arc(0, 0, letter, "EPS", w_char) + "\n")
    # insertion
    f.write(format_arc(0, 0, "EPS", letter, w_char) + "\n")
for i in range(len(alphabet)):
    for j in range(len(alphabet)):
        if i != j:
            # substitution
            f.write(format_arc(0, 0, alphabet[i], alphabet[j], w_char) + "\n")

# Make initial state also final state
f.write("0")
# Close file
f.close()


# In[19]:


get_ipython().system(' fstcompile --isymbols=chars.syms --osymbols=chars.syms transducer_char.fst transducer_char.fst')


# __δ)__ Αυτός είναι ένας αρκετά αφελής τρόπος για τον υπολογισμό των βαρών για κάθε edit. Αν τώρα είχαμε στη διάθεση μας ό,τι δεδομένα θέλουμε αυτό που θα κάναμε είναι ότι θα υπολογίζαμε τα βάρη με βάση το πόσο συχνά γίνεται αυτό το λάθος. Πιο συγκεκριμένα, θα υπολογίζαμε για κάθε σύμβολο του αλφαβήτου την πιθανότητα κάποιος να το διαγράψει, να το προσθέσει ή να το αντικαταστήσει με κάποιο άλλο. Στη συνέχεια, θα μετατρέπαμε αυτές τις πιθανότητες σε κόστη παίρνοντας τον αρνητικό λογάριθμο και θα είχαμε τα τελικά βάρη μας για κάθε σύμβολο στο deletion και το insertion και για κάθε δυάδα συμβόλων στο substitution. Ο υπολογισμός αυτός μπορεί να γίνει σε περίπτωση που είχουμε το ίδιο corpus αλλά με λάθη για να μπορούμε να βρούμε πολύ απλά τις μετρικές που θέλουμε.

# ### Βήμα 12: Κατασκευή γλωσσικών μοντέλων

# __α)__ Στο βήμα αυτό θα κατασκευάσουμε έναν αποδοχέα με μία αρχική κατάσταση που θα αποδέχεται κάθε λέξη του λεξικού όπως αυτό ορίστηκε στην προπαρασκευή του εργαστηρίου στο βήμα 3α. Τώρα, όμως, ως βάρη θα χρησιμοποιήσουμε τον αρνητικό λογάριθμο της πιθανότητας εμφάνισης κάθε λέξης __-logP(w)__. Πρέπει το κόστος αυτό να κατανεμηθεί κάπως στην λέξη έτσι ώστε όλη η λέξη συνολικά να έχει το παραπάνω κόστος. Για λόγους βελτιστοποίησης και απλότητας προφανώς συμφέρει να βάλουμε όλο το κόστος της λέξης μόνο στην πρώτη ακμή της και τις υπόλοιπες να τις θέσουμε 0. Το αρχείο περιγραφής του αποδοχέα αποθηκεύεται ως __acceptor_token.fst__.

# In[20]:


# Get tokens of the corpus (our acceptor should accept only these words)
tokens = get_tokens("War.txt")
# Open file to write mode
f = open("acceptor_token.fst", "w")
s = 1
for token in tokens:
    cost = token_weights[token]
    letters = list(token)
    for i in range(0, len(letters)):
        if i == 0:
            # For each token make state 1 its first state
            f.write(format_arc(1, s+1, letters[i], letters[i], cost) + "\n")
        else:
            f.write(format_arc(s, s+1, letters[i], letters[i]) + "\n")
        s += 1
        if i == len(letters) - 1:
            # When reaching the end of a token go to final state 0 though an ε-transition
            f.write(format_arc(s, 0, "EPS", "EPS") + "\n")
# Make state 0 final state
f.write("0")
# Close the file
f.close()


# In[21]:


get_ipython().system(' fstcompile --isymbols=chars.syms --osymbols=chars.syms acceptor_token.fst acceptor_token.fst')


# __β)__ Στη συνέχεια καλούμε τις συναρτήσεις fstrmepsilon, fstdeterminize και fstminimize για να βελτιστοποιήσουμε το μοντέλο μας (η λειτουργία τους έχει αναφερθεί στην προπαρασκευή).

# In[22]:


get_ipython().system(' fstrmepsilon acceptor_token.fst acceptor_token.fst')


# In[23]:


get_ipython().system(' fstdeterminize acceptor_token.fst acceptor_token.fst')


# In[24]:


get_ipython().system(' fstminimize acceptor_token.fst acceptor_token.fst')


# __γ)__ Τώρα θα επαναλάβουμε την ίδια διαδικασία για το character level γλωσσικό μοντέλο. Αυτό που θα αλλάξει δηλαδή είναι ότι αντί να τοποθετούμε στην πρώτη ακμή της λέξης το κόστος ολόκληρης της λέξης θα ορίζουμε για την μετάβαση σε κάθε γράμμα της λέξης το αντίστοιχο κόστος του. Σημειώνεται ότι αντίστοιχα με πριν το κόστος ενός χαρακτήρα ισούται με τον αρνητικό λογάριθμο της πιθανότητας εμφάνισής του. Το αρχείο περιγραφής του αποδοχέα αποθηκεύεται ως __acceptor_char.fst__.

# In[25]:


# Get tokens of the corpus (our acceptor should accept only these words)
tokens = get_tokens("War.txt")
# Open file to write mode
f = open("acceptor_char.fst", "w")
s = 1
for token in tokens:
    letters = list(token)
    for i in range(0, len(letters)):
        if i == 0:
            # For each token make state 1 its first state
            f.write(format_arc(1, s+1, letters[i], letters[i], character_weigths[letters[i]]) + "\n")
        else:
            f.write(format_arc(s, s+1, letters[i], letters[i], character_weigths[letters[i]]) + "\n")
        s += 1
        if i == len(letters) - 1:
            # When reaching the end of a token go to final state 0 though an ε-transition
            f.write(format_arc(s, 0, "EPS", "EPS") + "\n")
# Make state 0 final state
f.write("0")
# Close the file
f.close()


# In[26]:


get_ipython().system(' fstcompile --isymbols=chars.syms --osymbols=chars.syms acceptor_char.fst acceptor_char.fst')


# In[27]:


get_ipython().system(' fstrmepsilon acceptor_char.fst acceptor_char.fst')


# In[28]:


get_ipython().system(' fstdeterminize acceptor_char.fst acceptor_char.fst')


# In[29]:


get_ipython().system(' fstminimize acceptor_char.fst acceptor_char.fst')


# ### Βήμα 13: Κατασκευή ορθογράφων

# Στο βήμα αυτό θα κατασκευάσουμε δύο ορθογράφους χρησιμοποιώντας τα FST από τα παραπάνω βήματα. Η διαδικασία για κάθε έναν ορθογράφο θα είναι ίδια με αυτή που ακολουθήθηκε στο βήμα 7 της προπαρασκευής.

# __α)__ Ο πρώτος ορθογράφος που θα κατασκευάσουμε θα προκύψει συνθέτοντας τον word level transducer με το word level γλωσσικό μοντέλο.

# Αρχικά θα ταξινομήσουμε τις εξόδους του transducer_token και τις εισόδους του acceptor_token με την συνάρτηση __fstarcsort__.

# In[30]:


get_ipython().system(' fstarcsort --sort_type=olabel transducer_token.fst transducer_token.fst')
get_ipython().system(' fstarcsort --sort_type=ilabel acceptor_token.fst acceptor_token.fst')


# Στη συνέχεια συνθέτουμε τον transducer_token με τον acceptor_token με την συνάρτηση fstcompose αποθηκεύοντας τον spell checker μας στο αρχείο __spell_checker1.fst__.

# In[31]:


get_ipython().system(' fstcompose transducer_token.fst acceptor_token.fst spell_checker1.fst')


# __β)__ Ο δεύτερος ορθογράφος θα προκύψει συνθέτοντας τον word level tranducer με το unigram γλωσσικό μοντέλο.

# Αρχικά θα ταξινομήσουμε τις εισόδους του acceptor_char με την συνάρτηση __fstarcsort__.

# In[32]:


get_ipython().system(' fstarcsort --sort_type=ilabel acceptor_char.fst acceptor_char.fst')


# Στη συνέχεια συνθέτουμε τον transducer_token με τον acceptor_char με την συνάρτηση fstcompose αποθηκεύοντας τον spell checker μας στο αρχείο __spell_checker2.fst__.

# In[33]:


get_ipython().system(' fstcompose transducer_token.fst acceptor_char.fst spell_checker2.fst')


# __γ)__ Η διαφορά των δύο ορθογράφων βρίσκεται στο γλωσσικό μοντέλο που χρησιμοποιούν. Συγκεκριμένα:
#  1. __Word-Level μοντέλο:__ Ο 1ος ορθογράφος για να διορθώσει μία λέξη κοιτάει (πέρα από τον αριθμό των edits) την συχνότητα εμφάνισης της κάθε λέξης στο corpus. Έτσι, διορθώνει μία λέξη σε μία άλλη που είναι πιο πιθανό να είχε εμφανιστεί.
#  2. __Unigram μοντέλο:__ Ο 2ος ορθογράφος για να διορθώσει μία λέξη κοιτάει (πέρα από τον αριθμό των edits) την συχνότητα εμφάνισης κάθε γράμματος της διορθωμένης λέξης. Έτσι, διορθώνει μία λέξη αλλάζοντας κάθε γράμμα της στο πιο πιθανό που ήταν να εμφανιστεί.

# Για παράδειγμα έστω ότι έχουμε την λέξη __cit__ και οι δύο πιθανές λέξεις που βρίσκονται στο λεξικό μας και έχουν μόνο 1 αλλαγή είναι η __cat__ και η __cut__. Ο 1ος ορθογράφος πιθανώς να επιλέξει την cut επειδή είναι μία πιο συνιθισμένη λέξη. Από την άλλη, ο 2ο ορθογράφος μπορεί να επιλέξει την cat επειδή το γράμμα a εμφανίζεται πιο συχνά από το γράμμα u. Ένα αντίστοιχο παράδειγμα παρουσιάζεται στο τέλος του επόμενου βήματος όπου δίνουμε την λέξη qet στους δύο ορθογράφους.

# ### Βήμα 14: Αξιολόγηση των ορθογράφων

# __α)__ Για να κάνουμε το evaluation των δύο ορθογράφων κατεβάζουμε το παρακάτω σύνολο δεδομένων:

# In[34]:


get_ipython().system(' wget https://raw.githubusercontent.com/georgepar/python-lab/master/spell_checker_test_set')


# __β)__ Δημιουργούμε αρχικά μία συνάρτηση __predict__ η οποία δέχεται μία λέξη που πρέπει να διορθωθεί και γράφει σε ένα αρχείο __pred_word.fst__ την περιγραφή ενός FST το οποίο αποδέχεται την συγκεκριμένη λέξη. Το FST αυτό θα το κάνουμε στη συνέχεια compose με τον ορθογράφο για να πάρουμε το τελικό αποτέλεσμα.

# In[35]:


def predict(word):
    s= 1
    letters = list(word)
    # Open file to write mode
    f = open("pred_word.fst", "w")
    for i in range(0, len(letters)):
        # For each letter of the word make a transition with zero weight
        f.write(format_arc(s, s+1, letters[i], letters[i], 0) + '\n')
        s += 1
        if i == len(letters) - 1:
            # When reaching the end the word make a ε-transition to the final state 0 
            f.write(format_arc(s, 0, "EPS",  "EPS", 0) + '\n')
    # Final state
    f.write("0")
    # Close the file
    f.close()


# Είμαστε έτοιμοι, λοιπόν, τώρα να αξιολογήσουμε τους δύο ορθογράφους. Θα επιλέξουμε 10 τυχαίες λέξεις από το evaluation set που κατεβάσαμε και θα τις διορθώσουμε χρησιμοποιώντας τους 2 ορθογράφους μας.

# In[36]:


import random
random.seed(1)
test_words = []
for _ in range(10):
    random_lines = random.choice(open('spell_checker_test_set').readlines())
    test_words.append(random.choice(random_lines.strip('\n').split()[1:]))


# In[37]:


for word in test_words:
    print(word + ":" + " ",end='')
    predict(word)
    print("1: ",end='')
    get_ipython().system(' ./predict.sh spell_checker1.fst')
    print(" 2: ",end='')
    get_ipython().system(' ./predict.sh spell_checker2.fst')
    print('\n')


# __γ)__ Παρατηρούμε ότι έχουν μία αρκετά καλή επίδοση οι δύο ορθογράφοι μας η οποία αυξάνοντας το corpus (το οποίο είναι μόνο ένα βιβλίο) θα μπορούσαν να γίνουν ακόμα καλύτεροι. Συγκεκριμένα:
# - Ο 1ος ορθογράφος κατασκευάστηκε συνθέτοντας το word-level γλωσσικό μοντέλο με το word-level μετατροπέα. Αυτό σημαίνει, ότι ο ορθογράφος προσπαθεί να διορθώσει μία λέξη όχι μόνο λαμβάνοντας υπόψιν τις λιγότερες αλλαγές (όπως στην προπαρασκευή) αλλά και το πόσο πιθανή είναι η λέξη στην οποία θα μετατραπεί. Αυτό αυξάνει την επίδοσή του γιατί προφανώς όσο πιο πιθανή είναι μία λέξη τόσο και πιο πιθανό είναι να έχει γραφτεί λάθος. Ο μετατροπέας, τώρα, έγινε word-level έτσι ώστε να φέρουμε τα βάρη των edits στην ίδια τάξη μεγέθους με τα βάρη του γλωσσικού μοντέλου.
# - Ο 2ος ορθογράφος κατασκευάστηκε συνθέτοντας το unigram γλωσσικό μοντέλο με το word-level μετατροπέα. Αυτό σημαίνει ότι ο ορθογράφος προσπαθεί να διορθώσει μία λέξη λαμβάνοντας υπόψιν αυτή τη φορά πόσο πιθανό είναι το γράμμα το οποίο θέλει να διορθώσει. Αυτό το γλωσσικό μοντέλο επίσης αυξάνει την απόδοση γιατί όσο πιο πιθανό είναι ένα γράμμα τόσο και πιο πιθανό είναι να έχει γραφτεί λάθος το συγκεκριμένο γράμμα. Τα βάρη του μετατροπέα τώρα κάνουν την ίδια δουλειά που αναφέρθηκε και παραπάνω.

# Για να κατανοήσουμε καλύτερα την διαφορετική λειτουργία των 2 ορθογράφων δίνουμε ως είσοδο για διόρθωση την λέξη __qet__.

# In[38]:


word = "qet"
print(word + ":" + " ",end='')
predict(word)
print("1: ",end='')
get_ipython().system(' ./predict.sh spell_checker1.fst')
print(" 2: ",end='')
get_ipython().system(' ./predict.sh spell_checker2.fst')


# Παρατηρούμε ότι ο ορθογράφος με το word level γλωσσικό μοντέλο την διόρθωσε σε __get__, ενώ ο ορθογράφος με το unigram γλωσσικό μοντέλο την διόρθωσε σε __set__. Ο λόγος που συνέβη αυτό βρίσκεται στις πιθανότητες εμφάνισης κάθε λέξης αλλά και του συνολικού συνδυασμού των γραμμάτων κάθε λέξης.

# In[39]:


print("Propability of word get: " + str(dict_token["get"]))
print("Propability of word set: " + str(dict_token["set"]))
print("Propability of characters g: " + str(dict_character["g"]))
print("Propability of characters s: " + str(dict_character["s"]))


# Βλέπουμε ότι η πιθανότητα να δούμε get είναι μεγαλύτερη από το να δούμε set και γι´ αυτό ο word-level ορθογράφος μας που κοιτάει τα word-level βάρη επέλεξε να διορθώσει το qet σε get. Από την άλλη η πιθανότητα να δούμε s είναι μεγαλύτερη από το να δούμε g με αποτέλεσμα ο 2ος ορθογράφος που βασίζεται στις πιθανότητες εμφάνισης των γραμάτων διορθώνει την λέξη qet σε set.

# <h2><center> Μέρος 2: Χρήση σημασιολογικών αναπαραστάσεων για ανάλυση συναισθήματος</center></h2>

# Στο πρώτο μέρος της άσκησης ασχοληθήκαμε κυρίως με συντακτικά μοντέλα για την κατασκευή ενός ορθογράφου. Εδώ θα 
# ασχοληθούμε με τη __χρήση λεξικών αναπαραστάσεων για την κατασκευή ενός ταξινομητή συναισθήματος__ . Ως δεδομένα θα 
# χρησιμοποιήσουμε σχόλια για ταινίες από την ιστοσελίδα IMDB και θα τα ταξινομήσουμε σε θετικά και αρνητικά ως 
# προς το συναίσθημα.

# ### Βήμα 16: Δεδομένα και προεπεξεργασία 

# __α)__ Αρχικά κατεβάζουμε τα δεδομένα που θα χρησιμοποιήσουμε. Επειδή το αρχείο είναι μεγάλο η εντολή είναι σε σχόλιο σε περίπτωση που υπάρχει ήδη κατεβασμένο.

# In[40]:


# ! wget -N http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz


# Στη συνέχεια το αποσυμπιέζουμε το αρχείου που κατεβάσαμε στον ίδιο φάκελο με το όνομα __aclImdb__.

# In[41]:


# ! tar -zxf aclImdb_v1.tar.gz


# Οι φάκελοι που μας ενδιαφέρουν είναι οι εξής:
#  - __train__ που περιέχει όλες τις κριτικές που θα χρησιμοπιήσουμε για την εκπαίδευση του μοντέλου μας και χωρίζεται σε:
#      - __train/pos__ το οποίο περιέχει αυτές που έχουν χαρακτηριστεί ως θετικές και
#      - __train/neg__ το οποίο περιέχει αυτές που έχουν χαρακτηριστεί ως αρνητικές.
#  - __test__ που περιέχει όλες τις κριτικές που θα χρησιμοποιήσουμε για να ελέγξουμε την επίδοση του μοντέλου μας και αντίστοιχα χωρίζεται σε:
#      - __test/pos__ με τις θετικές και 
#      - __test/neg__ με τις αρνητικές.

# __β)__ Στη συνέχεια πρέπει να διαβάσουμε και να προεπεξεργαστούμε τα δεδομένα μας. Ο κώδικας ανάγνωσης και κάποιες απλές συναρτήσεις προεπεξεργασίας (τα οποία μας δώθηκαν έτοιμα για διευκόλυνση) παρουσιάζονται παρακάτω.

# - Αρχικά κάνουμε όλα τα απαραίτητα import.

# In[42]:


import random
import os
import numpy as np
import re
try:
    import glob2 as glob
except ImportError:
    import glob


# - Στη συνέχεια δηλώνουμε τα path των αρχείων που θα μας φανούνε χρήσιμα και κάποιες ακόμη μεταβλητές.

# In[43]:


# Useful paths
data_dir = './aclImdb/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
pos_train_dir = os.path.join(train_dir, 'pos')
neg_train_dir = os.path.join(train_dir, 'neg')
pos_test_dir = os.path.join(test_dir, 'pos')
neg_test_dir = os.path.join(test_dir, 'neg')

# For memory limitations. These parameters fit in 8GB of RAM.
# If you have 16G of RAM you can experiment with the full dataset / W2V
MAX_NUM_SAMPLES = 5000
# Load first 1M word embeddings. This works because GoogleNews are roughly
# sorted from most frequent to least frequent.
# It may yield much worse results for other embeddings corpora
NUM_W2V_TO_LOAD = 1000000
# Fix numpy random seed for reproducibility
SEED = 42
np.random.seed(42)


# - Η συνάρτηση __strip_punctuation__ δέχεται ως είσοδο ένα string και αντικαθιστά κάθε σύμβολό του που δεν είναι γράμμα με το κενό. Έτσι επιστρέφει ένα string που αποτελείται μόνο από κεφαλαία και μικρά γράμματα και κενά.

# In[44]:


def strip_punctuation(s):
    return re.sub(r'[^a-zA-Z\s]', ' ', s)


# - Η συνάρτηση __preprocess__ δέχεται ένα string και απαλείφει τα σημεία στίξης χρησιμοποιώντας την strip_punctuation, μετατρέπει όλα τα γράμματα σε μικρά και, τέλος, αντικαθιστά τα συνεχόμενα κενά από ένα μόνο κενό.

# In[45]:


def preprocess(s):
    return re.sub('\s+',' ', strip_punctuation(s).lower())


# - Η συνάρτηση __tokenize__ δέχεται ένα string και το διασπάσει στα κενά του, επιστρέφοντας μία λίστα με κάθε λέξη του string.

# In[46]:


def tokenize(s):
    return s.split(' ')


# - Η συνάρτηση __preproc_tok__ δέχεται ένα string και επιστρέφει μία λίστα με τα tokens, τις λέξεις δηλαδή μόνο με μικρά γράμματα και χωρίς σημεία στίξης.

# In[47]:


def preproc_tok(s):
    return tokenize(preprocess(s))


# - Η συνάρτηση __read_samples__ δέχεται ως ορίσματα το path ενός φακέλου που περιέχει τα samples και μία συνάρτηση preprocess (με default μία συνάρτηση που επιστρέφει ακριβώς όπως είναι το όρισμά της). Ανοίγει κάθε ένα από τα samples που είναι σε μορφή αρχείων .txt και καλεί την συνάρτηση preprocess. Το αποτέλεσμα __data__ είναι μία λίστα, όπου κάθε στοιχείο της αντιστοιχεί στο αποτέλεσμα της preprocess πάνω στην κάθε κριτική.

# In[48]:


def read_samples(folder, preprocess=lambda x: x):
    # Get all the .txt files that the folder contains
    samples = glob.iglob(os.path.join(folder, '*.txt'))
    data = []
    for i, sample in enumerate(samples):
        if MAX_NUM_SAMPLES > 0 and i == MAX_NUM_SAMPLES:
            break
        # Open the .txt file, preprocess each line and add the result to a list
        with open(sample, 'r') as fd:
            x = [preprocess(l) for l in fd][0]
            data.append(x)
    return data


# - Η συνάρτηση __create_corpus__ δέχεται δύο λίστες που περιέχουν κριτικές για ταινίες με την πρώτη να έχει τις θετικές κριτικές και την δεύτερη τις αρνητικές. Επιστρέφει μία λίστα που περιέχει τις δωσμένες κριτικές σε τυχαία σειρά και μία λίστα που περιέχει το label της κάθε κριτικής. Ουσιαστικά αυτή η συνάρτηση δημιουργεί το training και το test set μας σε raw μορφή αφού η κάθε γραμμή είναι μία κριτική σε μορφή ενός string.

# In[49]:


def create_corpus(pos, neg):
    corpus = np.array(pos + neg)
    y = np.array([1 for _ in pos] + [0 for _ in neg])
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    return list(corpus), list(y)


# Αφού ορίσαμε, λοιπόν, όλες μας τις συναρτήσεις τώρα πρέπει να διαβάσουμε τις κριτικές και την αντίστοιχη κατηγορία τους. Αυτό που θα κάνουμε είναι να δημιουργήσουμε τις εξής τέσσερις λίστες:
#  - __X_train_raw__ η οποία περιέχει όλες τις κριτικές που θα χρησιμοποιηθούν για το train του μοντέλου μας σε text μορφή.
#  - __Y_train__ η οποία περιέχει τα labels των παραπάνω κριτικών.
#  - __X_test_raw__ η οποία περιέχει όλες τις κριτικές που θα χρησιμοποιηθούν για το test του μοντέλου μας σε text μορφή.
#  - __Y_test__ η οποία περιέχει τα labels των παραπάνω κριτικών.

# In[50]:


X_train_raw, Y_train = create_corpus(read_samples(pos_train_dir), read_samples(neg_train_dir))
X_test_raw, Y_test = create_corpus(read_samples(pos_test_dir), read_samples(neg_test_dir))


# Μπορούμε να ελέγξουμε την 1η κριτική του training set και το αντιστοιχο label της για να δούμε ότι όλα πήγαν καλά.

# In[51]:


print(X_train_raw[0])
print("Postive" if Y_train[0] else "Negative")


# ### Βήμα 17: Κατασκευή BOW αναπαραστάσεων και ταξινόμηση

# Η πιο βασική αναπαράσταση για μια πρόταση είναι η χρήση __Bag of Words__. Σε αυτή την αναπαράσταση μια λέξη κωδικοποιείται σαν ένα one hot encoding πάνω στο λεξιλόγιο και μια πρόταση σαν το άθροισμα αυτών των encodings. Για παράδειγμα στο λεξιλόγιο [cat, dog, eat] η αναπαράσταση της λέξης cat είναι [1, 0,0], της λέξης dog [0, 1, 0] κοκ. Η αναπαράσταση της πρότασης dog eat dog είναι [0, 2, 1]. Επιπλέον μπορούμε να πάρουμε σταθμισμένο άθροισμα των one hot word encodings για την αναπαράσταση μιας πρότασης με βάρη TF-IDF (https://en.wikipedia.org/wiki/Tf–idf).

# __α)__  Στην __Bag of Words__ αναπαράσταση υπολογίζουμε απλά πόσες φορές υπάρχει η κάθε λέξη στην κάθε κριτική. Έτσι, προκύπτει για κάθε κριτική ένας μεγάλος και αραιός πίνακας (με μήκος ίσο με το μέγεθος του λεξικου) που σε κάθε θέση του έχει τις φορές που παρουσιάζεται η κάθε λέξη στην κριτική. Αυτή η αναπαράσταση έχει δύο σημαντικά μειονεκτήματα τα οποία αντιμετωπίζονται με την προσθήκη βαρών __TF_IDF__. Συγκεκριμένα έχουμε ότι:
# - Πρέπει να λαμβάνουμε υπόψιν και το μέγεθος της κάθε κριτικής γιατί άλλη βαρύτητα έχει η ύπαρξη μιας λέξης σε μία κριτική με μικρό μέγεθος και άλλη σε μία με μεγάλο. Γι' αυτό και στον πρώτο όρο της TF_IDF που είναι το __term frequency__ αφού υπολογίσουμε πόσες φορές υπάρχει μία λέξη στην κριτική, μετά διαιρούμε με τον συνολικό μέγεθος της κριτικής.
# - Λέξεις οι οποίες είναι συνηθισμένες λαμβάνουν μεγάλο score σε κάθε κριτική χωρίς να πρέπει. Το νόημα είναι ότι οι σπάνιες λέξεις μας δίνουν περισσότερη πληροφορία από τις συνηθισμένες. Έτσι, ο δεύτερος όρος που είναι το __inverse document frequency__ είναι ο συνολικός αριθμός των κριτικών διαιρεμένος από τον αριθμό των κριτικών στις οποίες βρίσκεται η λέξη μας, με αποτέλεσμα ο όρος αυτός να αυξάνεται όσο πιο σπάνια είναι η λέξη.

# __β)__  Τώρα θα χρησιμοποιήσουμε τον transformer CountVectorizer του sklearn για να εξάγουμε __μη σταθμισμένες BOW αναπαραστάσεις__.

# In[52]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the vectorizer using our preprocess and tokenize function.
vectorizer = CountVectorizer(analyzer = preproc_tok)
# Get training data X_train.
X_train = vectorizer.fit_transform(X_train_raw)
# Get test data X_test.
X_test = vectorizer.transform(X_test_raw)


# __γ)__ Σε αυτό το στάδιο έχουμε τους πίνακες με τα training και τα test data και τα αντίστοιχα labels. Οπότε μπορούμε να εφαρμόσουμε τον ταξινομητή Linear Regression του sklearn για να ταξινομήσουμε τα σχόλια σε θετικά και αρνητικά.

# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import zero_one_loss

# Define the clasifier
clf = LogisticRegression()
# Train the model
clf.fit(X_train, Y_train)


# In[54]:


# Compute error on training data.
print("Training error =", zero_one_loss(Y_train, clf.predict(X_train)))
# Compute error on test data
print("Test error =", zero_one_loss(Y_test, clf.predict(X_test)))


# __δ)__ Τώρα θα επαναλάβουμε την ίδια διαδικασία χρησιμοποιώντας τον TfidfVectorizer για την εξαγώγη TF-IDF αναπαραστάσεων.

# In[55]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer = preproc_tok)
X_train = tfidf_vectorizer.fit_transform(X_train_raw)
X_test = tfidf_vectorizer.transform(X_test_raw)


# In[56]:


# Define the clasifier
clf_tfidf = LogisticRegression()
# Train the model
clf_tfidf.fit(X_train, Y_train)
# Compute error on training data.
print("Training error =", zero_one_loss(Y_train, clf_tfidf.predict(X_train)))
# Compute error on test data
print("Test error =", zero_one_loss(Y_test, clf_tfidf.predict(X_test)))


# #### Σύγκριση αποτελεσμάτων:
# Παρατηρούμε ότι το test error μειώνεται κατά 1% περίπου όταν χρησιμοποιούμε βάρη TF-IDF για την αναπαράσταση μιας πρότασης. Το αποτέλεσμα αυτό ήταν αναμενόμενο γιατί όπως ειπώθηκε στο α) η αναπαράσταση αυτή καλύπτει κάποια κενά που είχε η μη σταθμισμένη BOW αναπαράσταση. 

# ### Βήμα 18: Χρήση Word2Vec αναπαραστάσεων για ταξινόμηση

# Ένας άλλος τρόπος για να αναπαραστήσουμε λέξεις και προτάσεις είναι να κάνουμε χρήση προεκπαιδευμένων embeddings. Σε αυτό το βήμα θα εστιάσουμε στα word2vec embeddings. Αυτά τα embeddings προκύπτουν από ένα νευρωνικό δίκτυο με ένα layer το οποίο καλείται να προβλέψει μια λέξη με βάση το context της (παράθυρο 3-5 λέξεων γύρω από αυτή). Αυτό ονομάζεται CBOW μοντέλο. Εναλλακτικά το δίκτυο καλείται να προβλέψει το context με βάση τη λέξη (skip-gram μοντέλο). Τα word2vec vectors είναι πυκνές (dense) αναπαραστάσεις σε λιγότερες διαστάσεις από τις BOW και κωδικοποιούν σημασιολογικά χαρακτηριστικά μιας λέξης με βάση την υπόθεση ότι λέξεις με παρόμοιο νόημα εμφανίζονται σε παρόμοια συγκείμενα (contexts). Μια πρόταση μπορεί να αναπαρασταθεί ως ο μέσος όρος των w2v διανυσμάτων κάθε λέξης που περιέχει (Neural Bag of Words).

# Αρχικά θα επαναλάβουμε τα βήματα 9α, 9β της προπαρασκευής γιατί θα μας χρειαστούν για τα δύο πρώτα ερωτήματα.

# - Διαβάζουμε το βιβλίο War of the Worlds που είχαμε κατεβάσει για το μέρος Α σε μία λίστα από tokenized προτάσεις.

# In[57]:


import nltk

# We split the corpus in a list of tokenized sentences.
file_path = "War.txt"
tokenized_sentences = []
with open(file_path, "r") as f:
    text = f.read()
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [preproc_tok(sentence) for sentence in sentences]


# - Xρησιμοποιούμε την κλάση Word2Vec του gensim για να εκπαιδεύσουμε 100-διάστατα word2vec embeddings με βάση τις παραπάνω προτάσεις. Θα χρησιμοποιήσουμε window = 5 και 1000 εποχές.

# In[58]:


from gensim.models import Word2Vec

# Initialize word2vec. Context is taken as the 2 previous and 2 next words
myModel = Word2Vec(tokenized_sentences, window=5, size=100, workers=4)
# Train the model for 1000 epochs
myModel.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=1000)


# Η μεταβλητή __voc__ κρατάει το λεξικό μας ενώ η __dim__ το μέγεθος του κάθε embedding.

# In[59]:


# get ordered vocabulary list
voc = myModel.wv.index2word
# get vector size
dim = myModel.vector_size


# Η συνάρτηση __to_embeddings_Matrix__ δέχεται ως όρισμα το μοντέλο μας και επιστρέφει έναν 2-διάστατο πίνακα όπου κάθε γραμμή αναπαριστάσει ένα embedding και ένα λεξικό.

# In[60]:


# Convert to numpy 2d array (n_vocab x vector_size)
def to_embeddings_Matrix(model):  
    embedding_matrix = np.zeros((len(model.wv.vocab), model.vector_size))
    for i in range(len(model.wv.vocab)):
        embedding_matrix[i] = model.wv[model.wv.index2word[i]]
    return embedding_matrix, model.wv.index2word


# __α)__ Σε αυτό το βήμα πρέπει να υπολογίσουμε το ποσοστό __out of vocabulary (OOV) words__ για τις παραπάνω αναπαραστάσεις. 

# In[61]:


tokens = get_tokens("War.txt")
oov = (1 - len(voc)/len(tokens)) * 100
print("Out of vocabulary words: " + str(oov) + "%")


# __β)__ Τώρα χρησιμποιώντας αυτές τις αναπαραστάσεις θα κατασκευάσουμε ένα __Neural Bag of Words αναπαραστάσεων__ για κάθε σχόλιο στο corpus και θα εκπαιδεύσουμε ένα Logistic Regression μοντέλο για ταξινόμηση.

# Αρχικά, αποθηκεύουμε τo training και το test set σε raw text μορφή.

# In[62]:


X_train_raw, Y_train = create_corpus(read_samples(pos_train_dir), read_samples(neg_train_dir))
X_test_raw, Y_test = create_corpus(read_samples(pos_test_dir), read_samples(neg_test_dir))


# Στη συνέχεια, για κάθε κριτική υπολογίζουμε το neural bag of words, που ορίζεται ως ο μέσος όρος των w2v διανυσμάτων κάθε λέξης που περιέχει.

# In[63]:


# Initialize training set
X_train = np.zeros((len(X_train_raw), 100))
for row, sample in enumerate(X_train_raw):
    words_included = 0
    # Tokenize current review
    sample_toks = preproc_tok(sample)
    for tok in sample_toks:
        # For each token check if it has a w2v representation
        # and if yes add it.
        if tok in myModel.wv:
            X_train[row] += myModel.wv[tok]
            words_included += 1
    # Get the mean value
    X_train[row] = X_train[row]/words_included


# In[64]:


# Initialize test set
X_test = np.zeros((len(X_test_raw), 100))
for row, sample in enumerate(X_test_raw):
    words_included = 0
    # Tokenize current review
    sample_toks = preproc_tok(sample)
    for tok in sample_toks:
        # For each token check if it has a w2v representation
        # and if yes add it.
        if tok in myModel.wv:
            X_test[row] += myModel.wv[tok]
            words_included += 1
    # Get the mean value
    X_test[row] = X_test[row]/words_included


# In[65]:


# Define the clasifier
clf = LogisticRegression()
# Train the model
clf.fit(X_train, Y_train)


# In[66]:


# Compute error on training data.
print("Training error =", zero_one_loss(Y_train, clf.predict(X_train)))
# Compute error on test data
print("Test error =", zero_one_loss(Y_test, clf.predict(X_test)))


# Και τα δύο error είναι πάρα πολύ υψηλά με αποτέλεσμα το μοντέλο μας να έχει πάρα πολύ χαμηλή απόδοση. Η εξήγηση για αυτό είναι ότι έχουμε κατασκευάσει τα word embeddings με βάση ένα πάρα πολύ μικρό corpus το οποίο και έχει μικρό λεξικό (με αποτέλεσμα πολλές λέξεις να μην έχουν αναπαράσταση) και δεν βοηθάει στο να δημιουργηθούν παρόμοιες αναπαραστάσεις για κοντινά σημασιολογικά λέξεις (αυτό το παρατηρήσαμε και στην προπαρασκευή όταν είδαμε τις κοντινές σημασιολογικά λέξεις 10 τυχαίων λέξεν).

#  __γ, δ)__ Κατεβάζουμε το προεκπαιδευμένα GoogleNews vectors, τα φορτώνουμε με το gensim και εξάγουμε αναπαραστάσεις με βάση αυτά.

# In[67]:


from gensim.models import KeyedVectors
googleModel = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True, limit=NUM_W2V_TO_LOAD)


# Επαναλαμβάνουμε το ερώτημα 9γ της προπαρασκευής για να το συγκρίνουμε με τα GoogleNews. 

# In[68]:


selected_words = random.sample(voc, 10)


# In[69]:


for word in selected_words:
    # get most similar words
    sim = myModel.wv.most_similar(word, topn=5)
    print('"' + word + '"' + " is similar with the following words:")
    for s in sim:
        print('"' + s[0] + '"' + " with similarity " + str(s[1]))
    print()


# In[70]:


for word in selected_words:
    # get most similar words
    sim = googleModel.most_similar(word, topn=5)
    print('"' + word + '"' + " is similar with the following words:")
    for s in sim:
        print('"' + s[0] + '"' + " with similarity " + str(s[1]))
    print()


# Αυτό που παρατηρούμε είναι ότι προφανώς με τα Google Vectors τα αποτελέσματα είναι εντυπωσικά αφού όλες οι κοντινές λέξεις είναι και στην πραγματικότητα πολύ κοντινές. Από την άλλη, το δικό μας μοντέλο έχει πολύ χαμηλές επιδόσεις που οφείλεται στο γεγονός ότι τα embeddings προέκυψαν από πολύ μικρό corpus. Τα Google Vectors από την άλλη έχουν ένα τεράστιο corpus από πίσω με αποτέλεσμα και να έχει τεράστιο λεξικό αλλά και οι σημασιολογικά κοντινές λέξεις να έχει και παρόμοια αναπαράσταση.

# __ε)__ Αντίστοιχα με το myModel τώρα θα εκπαιδεύσουμε ένα Logistic Regression ταξινομητή με το μοντέλο που προέκυψε από τα Google Vectors.

# In[71]:


# Initialize training set
X_train = np.zeros((len(X_train_raw), 300))
for row, sample in enumerate(X_train_raw):
    words_included = 0
    # Tokenize current review
    sample_toks = preproc_tok(sample)
    for tok in sample_toks:
        # For each token check if it has a w2v representation
        # and if yes add it.
        if tok in googleModel:
            X_train[row] += googleModel[tok]
            words_included += 1
    # Get the mean value
    X_train[row] = X_train[row]/words_included


# In[72]:


# Initialize test set
X_test = np.zeros((len(X_test_raw), 300))
for row, sample in enumerate(X_test_raw):
    words_included = 0
    # Tokenize current review
    sample_toks = preproc_tok(sample)
    for tok in sample_toks:
        # For each token check if it has a w2v representation
        # and if yes add it.
        if tok in googleModel:
            X_test[row] += googleModel[tok]
            words_included += 1
    # Get the mean value
    X_test[row] = X_test[row]/words_included


# In[73]:


# Define the clasifier
clf = LogisticRegression()
# Train the model
clf.fit(X_train, Y_train)


# In[74]:


# Compute error on training data.
print("Training error =", zero_one_loss(Y_train, clf.predict(X_train)))
# Compute error on test data
print("Test error =", zero_one_loss(Y_test, clf.predict(X_test)))


# Όπως ήταν αναμενόμενο το error μειώθηκε κατά πολύ καθώς τώρα τα embeddings ήταν καλύτερα. Σε σύγκριση με το TF_IDF το error εδώ είναι λίγο μεγαλύτερο αλλά κερδίζουμε πολύ σε χώρο και χρόνο καθώς οι πίνακες με τα training και test data είναι πολύ πιο μικροί και πυκνοί.

# __στ)__ Τώρα θα δημιουργήσουμε αναπαραστάσεις των κριτικών με χρήση σταθμισμένου μέσου των w2v
# αναπαραστάσεων των λέξεων. Ως βάρη θα χρησιμοποιήσουμε τα TF-IDF βάρη των λέξεων.

# In[77]:


# Get the vocabulary of the words in the training set 
# that contains their tf-idf value.
tfidf_vectorizer = TfidfVectorizer(analyzer = preproc_tok)
X_train_temp = tfidf_vectorizer.fit_transform(X_train_raw)
voc = tfidf_vectorizer.vocabulary_
# Do the same as before but now, we multiply each represantation by a the tf-idf of the word.
# Initialize training set
X_train = np.zeros((len(X_train_raw), 300))
for row, sample in enumerate(X_train_raw):
    # Tokenize current review
    sample_toks = preproc_tok(sample)
    for tok in sample_toks:
        # For each token check if it has a w2v representation
        # and if yes add it.
        if tok in googleModel and tok in voc:
            X_train[row] += googleModel[tok] * X_train_temp[row,voc[tok]]


# In[ ]:


# Get the vocabulary of the words in the training set 
# that contains their tf-idf value.
tfidf_vectorizer = TfidfVectorizer(analyzer = preproc_tok)
X_test_temp = tfidf_vectorizer.fit_transform(X_test_raw)
voc = tfidf_vectorizer.vocabulary_
# Do the same as before but now, we multiply each represantation by a the tf-idf of the word.
# Initialize test set
X_test = np.zeros((len(X_test_raw), 300))
for row, sample in enumerate(X_test_raw):
    # Tokenize current review
    sample_toks = preproc_tok(sample)
    for tok in sample_toks:
        # For each token check if it has a w2v representation
        # and if yes add it.
        if tok in googleModel and tok in voc:
            X_test[row] += googleModel[tok] * X_test_temp[row,voc[tok]]


# __ζ)__ Επαναλαμβάνουμε την ταξινόμηση με τις νέες αναπαραστάσεις.

# In[ ]:


# Define the clasifier
clf = LogisticRegression()
# Train the model
clf.fit(X_train, Y_train)
# Compute error on training data.
print("Training error =", zero_one_loss(Y_train, clf.predict(X_train)))
# Compute error on test data
print("Test error =", zero_one_loss(Y_test, clf.predict(X_test)))

