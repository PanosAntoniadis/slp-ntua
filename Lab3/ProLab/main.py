import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score
import torch
from torch.utils.data import DataLoader
import numpy as np

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.100d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 40
DATASET = "MR"  # options: "MR", "Semeval2017A"

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

# ------------ #
#     EX1      #
# ------------ #
# Convert data labels from strings to integers
# create a new label encoder
le = LabelEncoder()
# encode train set labels
y_train = le.fit_transform(y_train)  # EX1
# encode test set labels
y_test = le.fit_transform(y_test)  # EX1
# compute number of classes made by the encoder
n_classes = le.classes_.size
#print("EX1 answer printing")
#print("First 10 unencoded labels from the training set are: ")
#print(le.inverse_transform(y_train[:10]))
#print("First 10 encoded labels from the training set are: ")
#print(y_train[:10])

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
# ------------ #
#     EX2      #
# ------------ #
# EX2
# print first 10 tokenized training examples
#print("EX2 printing")
#for i in range(10):
#    print(train_set.data[i])

# ------------ #
#     EX3      #
# ------------ #
# EX3
# print("EX3 printing")
# for i in range(5):
#    print("Initial sentence is: ")
#    print(X_train[i])
#    sent = train_set[i]
#    print("Encoded sentence is: ")
#    print(sent)
test_set = SentenceDataset(X_test, y_test, word2idx)

# ------------ #
#     EX7      #
# ------------ #
# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = BaselineDNN(output_size= n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)
print(model.parameters())

# ------------ #
#     EX8      #
# ------------ #
if (n_classes == 2):
    criterion = torch.nn.BCEWithLogitsLoss()  # EX8
else:
    criterion = torch.nn.CrossEntropyLoss()

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
parameters = model.parameters()  # EX8
optimizer = torch.optim.Adam(parameters)  # EX8

#############################################################################
# Training Pipeline
#############################################################################
total_train_losses = []
total_test_losses = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)

    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    total_train_losses.append(train_loss)
    total_test_losses.append(test_loss)
    # Convert preds and golds in a list.
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_test_true = np.concatenate( y_test_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_test_pred = np.concatenate( y_test_pred, axis=0 )
    print("My train loss is :" , train_loss)
    print("My test loss is :", test_loss)
    print("Accuracy for train:" , accuracy_score(y_train_true, y_train_pred))
    print("Accuracy for test:" , accuracy_score(y_test_true, y_test_pred))
    print("F1 score for train:", f1_score(y_train_true, y_train_pred, average='macro'))
    print("F1 score for test:", f1_score(y_test_true, y_test_pred, average='macro'))
    print("Recall score for train:", recall_score(y_train_true, y_train_pred, average='macro'))
    print("Recall score for test:", recall_score(y_test_true, y_test_pred, average='macro'))

# Plot
plt.plot(range(1,EPOCHS+1), total_train_losses)
plt.show()

plt.plot(range(1,EPOCHS+1), total_test_losses)
plt.show()
