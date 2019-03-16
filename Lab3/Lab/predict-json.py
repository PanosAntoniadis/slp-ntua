# A script that takes a model as input, predicts the
# validation set of Semeval2017A and save the results
# in a specific json format to be used as data file for
# neat-vision.
import os
import warnings
import argparse
import errno
import json

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score
import torch
from torch.utils.data import DataLoader
import numpy as np
import sys

from config import EMB_PATH
from dataloading import SentenceDataset
from models.BaselineDNN import BaselineDNN
from models.LSTMNet import LSTMNet
from models.LSTMpool import LSTMpool
from models.NN_Attention import NN_Attention
from models.LSTM_Attention import LSTM_Attention
from models.BiLSTM_Attention import BiLSTM_Attention

from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from nltk.tokenize import TweetTokenizer

from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

BATCH_SIZE = 128
EMBEDDINGS = os.path.join(EMB_PATH, 'glove.twitter.27B.50d.txt')
EMB_DIM = 50
EMB_TRAINABLE = False
DATASET = "Semeval2017A"

# If your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# lLoad word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# Load the raw data
if DATASET == "Semeval2017A":
    _, _, X_test, y_test = load_Semeval2017A()
else:
    raise ValueError("Invalid dataset")

# Convert data labels from strings to integers
# create a new label encoder
le = LabelEncoder()
# encode test set labels
y_test = le.fit_transform(y_test)  # EX1
# compute number of classes made by the encoder
n_classes = le.classes_.size

# Define our PyTorch-based Dataset
test_set = SentenceDataset(X_test, y_test, word2idx)
tweetToken = TweetTokenizer()
text = [tweetToken.tokenize(example) for example in X_test]

# Define our PyTorch-based DataLoader
# Batch size is 1.
test_loader = DataLoader(test_set)
# Load user model.
model = torch.load(sys.argv[1]).to(DEVICE)
# Define criterion for evaluation.
loss_function = torch.nn.CrossEntropyLoss()

model.eval()
# IMPORTANT: in evaluation mode, we don't want to keep the gradients
# so we do everything under torch.no_grad()
data = []
with torch.no_grad():
    for index, batch in enumerate(test_loader):
        sample = {}

        # get the inputs (batch)
        inputs, labels, lengths = batch

        # Save text in dictionary and label
        sample['text'] = text[index]
        sample['label'] = labels.item()

        # Step 1 - move the batch tensors to the right device
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Step 2 - forward pass: y' = model(x)
        y_preds = model(inputs, lengths)  # EX9
        # Step 4 - make predictions (class = argmax of posteriors)
        y_preds_arg = torch.argmax(y_preds)

        # Save prediction and id
        sample['prediction'] = y_preds_arg.item()
        sample['posterior'] = y_preds.cpu().numpy().tolist()
        param = model.state_dict()
        sample['attention'] = model.scores[0].cpu().numpy().tolist()
        sample['id'] = index
        data.append(sample)

with open(sys.argv[2], 'w') as fp:
    json.dump(data, fp)
