import os
import warnings
import argparse
import errno

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
from models.BiLSTMpool import BiLSTMpool

from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

# Convert a string to a class object.
def str_to_class(str):
    return getattr(sys.modules[__name__], str)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################

# Parse user argument to specify embeddings, embeddings size, batch size
# and epochs.
parser = argparse.ArgumentParser(description='Sentiment analysis in Semeval2017A')
parser.add_argument('-embedding', '--embedding', help='Embeddings to be used (default: glove.twitter.27B.50d.txt)',
    default='glove.twitter.27B.50d.txt')
parser.add_argument('-emb_size', '--emb_size', help='Size of the embeddings to be used (default: 50)',
    choices=[50, 100, 200], default = 50, type=int)
parser.add_argument('-epochs', '--epochs', help='number of epochs (default: 30)', default = 30, type = int)
parser.add_argument('-batch_size', '--batch_size', help='Size of each mini batch (default: 128)', default = 128,
    type = int)
parser.add_argument('-model', '--model_name', help='Model to use (default: BaselineDNN)', default = 'BaselineDNN',
    choices=['BaselineDNN', 'LSTMNet', 'LSTMpool', 'NN_Attention', 'LSTM_Attention', 'BiLSTM_Attention', 'BiLSTMpool'])
parser.add_argument('-save', '--save', help='True if you want to save the model (default: False)', default = False,
    choices=['True', 'False'])
# Get arguments from the parser in a dictionary,
args = vars(parser.parse_args())

# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# Point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, args['embedding'])

# Set the correct dimensionality of the embeddings
EMB_DIM = args['emb_size']
# Set the rest of the necessary variables.
EMB_TRAINABLE = False
BATCH_SIZE = args['batch_size']
EPOCHS = args['epochs']
DATASET = "Semeval2017A"
MODEL = args['model_name']
SAVE = args['save']

# Check inputs.
if EPOCHS <= 0:
    raise ValueError("Invalid number of epochs")
if BATCH_SIZE <= 0:
    raise ValueError("Invalid batch size")
if not os.path.exists(EMBEDDINGS):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), EMBEDDINGS)

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Print useful parameters to the user
########################################################
print('Sentiment Analysis in Semeval2017A')
print("Dataset: " + DATASET)
print("Embeddings: " + args['embedding'])
print("Embeddings dimension: " + str(EMB_DIM))
print("Batch size: " + str(BATCH_SIZE))
print("Number of epochs: " + str(EPOCHS))
print("Model to be used: " + MODEL)
print("Device available: " + str(DEVICE))

########################################################
# Load word embeddings and data to be trained
########################################################

# Load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# Load the raw data (here they are in a form of raw text).
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
else:
    raise ValueError("Invalid dataset")

########################################################
# Convert data labels from text to integers
########################################################
# Sreate a new label encoder
le = LabelEncoder()
# Encode train set labels
y_train = le.fit_transform(y_train)
# Encode test set labels
y_test = le.fit_transform(y_test)
# Compute number of classes made by the encoder
n_classes = le.classes_.size

########################################################
# Define Pytorch datasets and dataloaders
########################################################
# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=4)

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
# Convert string from input to class object.
model_obj = str_to_class(MODEL)
model = model_obj(output_size= n_classes,
                         embeddings=embeddings,
                         trainable_emb=EMB_TRAINABLE)

# Move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

criterion = torch.nn.CrossEntropyLoss()
# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
parameters = model.parameters()
optimizer = torch.optim.Adam(parameters)

#############################################################################
# Training Pipeline
#############################################################################
# Define lists for train and test loss over each epoch
total_train_losses = []
total_test_losses = []
for epoch in range(1, EPOCHS + 1):
    # Train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    # Save losses to the corresponding lists
    total_train_losses.append(train_loss)
    total_test_losses.append(test_loss)
    # Convert preds and golds in a list.
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )
    # Print metrics for current epoch
    print("My train loss is :" , train_loss)
    print("My test loss is :", test_loss)
    print("Accuracy for train:" , accuracy_score(y_train_true, y_train_pred))
    print("Accuracy for test:" , accuracy_score(y_test_true, y_test_pred))
    print("F1 score for train:", f1_score(y_train_true, y_train_pred, average='macro'))
    print("F1 score for test:", f1_score(y_test_true, y_test_pred, average='macro'))
    print("Recall score for train:", recall_score(y_train_true, y_train_pred, average='macro'))
    print("Recall score for test:", recall_score(y_test_true, y_test_pred, average='macro'))

# Plot train and test loss curve
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('Model: ' + MODEL)
plt.plot(range(1,EPOCHS+1), total_train_losses)
plt.show()
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Model: ' + MODEL)
plt.plot(range(1,EPOCHS+1), total_test_losses)
plt.show()
# Ιf asked to, save the model
if (SAVE == 'True'):
    torch.save(model, MODEL + '.pt')
    print("Model saved succesfully to " + MODEL + '.pt')
