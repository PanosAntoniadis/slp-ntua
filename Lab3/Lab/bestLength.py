# A simple script that prints as a histogram the
# length of each sentence in our training set in
# order to choose the appropriate for our network.
import os
import warnings
import torch
import sys
import numpy as np

from utils.load_embeddings import load_word_vectors
from config import EMB_PATH
from utils.load_datasets import load_MR, load_Semeval2017A
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from ekphrasis.classes.tokenizer import SocialTokenizer

DATASET = sys.argv[1]  # options: "MR", "Semeval2017A"

# Ιf your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# Print a histogram with the length of each sentence and other
# useful statistics (mean value, variance) in order to decide a
# max sentence length.
# IMPORTANT: Implement the same tokenization as in the implementation.
social_tokenizer = SocialTokenizer().tokenize
train_data = [social_tokenizer(example) for example in X_train]
lengths_train = [len(social_tokenizer(example)) for example in X_train]
plt.figure(figsize=(10, 4))
plt.xlabel('Length')
plt.ylabel('Number of sentences')
plt.title('Train set histogram')
plt.hist(lengths_train)
print("Size of train set: " + str(len(X_train)))
print("Mean value of train set: " + str(sum(lengths_train)/float(len(lengths_train))))
print("Variance of train set is: " + str(np.std(lengths_train)))

test_data = [social_tokenizer(example) for example in X_test]
lengths_test = [len(social_tokenizer(example)) for example in X_test]
plt.figure(figsize=(10, 4))
plt.xlabel('Length')
plt.ylabel('Number of sentences')
plt.title('Test set histogram')
plt.hist(lengths_test)
print("Size of test set: " + str(len(X_test)))
print("Mean value of test set: " + str(sum(lengths_test)/float(len(lengths_test))))
print("Variance of test set is: " + str(np.std(lengths_test)))
plt.show()
