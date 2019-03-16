import torch

from torch import nn
import numpy as np
from .SelfAttention import SelfAttention


class BiLSTM_Attention(nn.Module):

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """
        super(BiLSTM_Attention, self).__init__()
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
        # Define lstm layer.
        self.hidden_size = 16
        self.num_layers = 1
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional = True)
        # Define attention layer.
        self.attention = SelfAttention(2*self.hidden_size)
        # Define linear layer.
        self.linear = nn.Linear(2*self.hidden_size, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # Obtain the model's device ID
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define useful variables
        batch_size = len(x)
        # Embed the words, using the embedding layer
        embeddings = self.embeddings(x)
        # Set initial hidden and cell states for the lstm layer.
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        # Forward propagate LSTM
        # lstm_out: tensor of shape (batch_size, max_length,2* hidden_size)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        # Pass through the attention layer.
        representations,_ = self.attention(lstm_out, lengths)
        # Decode the hidden state of the last time step
        logits = self.linear(representations)

        return logits
