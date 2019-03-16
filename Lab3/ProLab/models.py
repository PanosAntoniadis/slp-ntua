import torch

from torch import nn
import numpy as np


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """
        super(BaselineDNN, self).__init__()
        # ------------ #
        #     EX4      #
        # ------------ #
        # EX4
        # define some useful variables
        n_embeddings, embedding_size = np.shape(embeddings)
        # 1 - define the embedding layer
        self.embeddings = nn.Embedding(num_embeddings = n_embeddings, embedding_dim = embedding_size)

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # EX4
        self.embeddings.load_state_dict({'weight': torch.from_numpy(embeddings)})

        # 3 - define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embeddings.weight.requires_grad = False

        # ------------ #
        #     EX5      #
        # ------------ #
        # 4 - define a non-linear transformation of the representations
        # EX5
        hidden_size = 32
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        #self.linear3 = nn.Linear(256, hidden_size)
        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5
        self.linear2 = nn.Linear(hidden_size, output_size)


    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # ------------ #
        #     EX6      #
        # ------------ #
        # Useful variables
        batch_size = len(x)
        # 1 - embed the words, using the embedding layer
        embeddings = self.embeddings(x)  # EX6
        embeddings_size = embeddings.shape[2]

        # 2 - construct a sentence representation out of the word embeddings
        # by computing the mean of the word embeddings
        representations = torch.zeros([batch_size, embeddings_size])
        for i in range(batch_size):
            representations[i] = torch.sum(embeddings[i], dim=0) / lengths[i]

        # 3 - transform the representations to new ones.
        representations =  self.relu(self.linear1(representations))# EX6
        #representations =  self.relu(self.linear3(representations))# EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.linear2(representations)  # EX6

        return logits
