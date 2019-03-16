import torch

from torch import nn
import numpy as np
from .SelfAttention import SelfAttention

class NN_Attention(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer.
    2. We compute the weighted sum of word embeddings.
    3. We project with a linear layer the representation
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
        super(NN_Attention, self).__init__()
        # Define some useful variables
        n_embeddings, self.embedding_size = np.shape(embeddings)
        # Define the embedding layer
        self.embeddings = nn.Embedding(num_embeddings = n_embeddings, embedding_dim = self.embedding_size)
        # Initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.embeddings.load_state_dict({'weight': torch.from_numpy(embeddings)})
        # Define if the embedding layer will be frozen or finetuned
        if not trainable_emb:
            self.embeddings.weight.requires_grad = False

        # Define the attention layer.
        self.attention = SelfAttention(self.embedding_size)

        # Define the final Linear layer which maps
        # the representations to the classes
        self.linear = nn.Linear(self.embedding_size, output_size)


    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # Define useful variables
        batch_size = len(x)
        # Embed the words, using the embedding layer
        embeddings = self.embeddings(x)

        # Construct a sentence representation out of the word embeddings
        # by computing the weighted sum using the attention layer.
        representations, scores = self.attention(embeddings, lengths)
        self.scores = scores
        # Project the representations to classes using a linear layer
        logits = self.linear(representations)

        return logits
