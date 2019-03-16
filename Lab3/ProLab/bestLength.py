# A simple script that prints as a scatterplot the
# length of each sentence in our training set in
# order to choose the appropriate for our network.
import os
import warnings
import torch
import sys

from utils.load_embeddings import load_word_vectors
from config import EMB_PATH
from utils.load_datasets import load_MR, load_Semeval2017A
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = sys.argv[1]  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# print a scatter plot with the length of each sentence in the
# training set in order to decide a max sentence length.
idx_temp = list(range(len(X_train)))
plt.scatter(idx_temp, [len(word_tokenize(example)) for example in X_train])
