import torch

from torch import nn
import numpy as np


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.
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
        # Define some useful variables
        n_embeddings, self.embedding_size = np.shape(embeddings)
        # 1 - define the embedding layer
        self.embeddings = nn.Embedding(num_embeddings = n_embeddings, embedding_dim = self.embedding_size)
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embeddings.load_state_dict({'weight': torch.from_numpy(embeddings)})
        # 3 - define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embeddings.weight.requires_grad = False

        # 4 - define a non-linear transformation of the representations
        self.hidden_size = 32
        self.linear1 = nn.Linear(self.embedding_size*2, self.hidden_size)
        self.relu = nn.ReLU()
        # 5 - define the final Linear layer which maps
        # the representations to the classes
        self.linear2 = nn.Linear(self.hidden_size, output_size)


    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """

        # Obtain the model's device ID
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Useful variables
        batch_size = len(x)
        # 1 - embed the words, using the embedding layer
        embeddings = self.embeddings(x)

        # 2 - Construct a sentence representation out of the word embeddings
        # by computing the concatenation of the mean pooling and the max pooling
        # Create a representation for each sentence by computing the mean pooling.
        representations_mean = torch.zeros([batch_size, self.embedding_size]).to(DEVICE)
        for i in range(batch_size):
            representations_mean[i] = torch.sum(embeddings[i], dim=0) / lengths[i]
        # Create a representation for each sentence by computing the max pooling.
        representations_max = torch.zeros([batch_size, self.embedding_size]).to(DEVICE)
        for i in range(batch_size):
            representations_max[i],_ = torch.max(embeddings[i], dim=0)
        # Create the final representation for each sentence by concatenating mean and max pooling.
        representations = torch.cat((representations_mean, representations_max), 1)
        # 3 - transform the representations to new ones.
        representations =  self.relu(self.linear1(representations))
        # 4 - project the representations to classes using a linear layer
        logits = self.linear2(representations)

        return logits
