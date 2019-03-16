import torch

from torch import nn
import numpy as np

class LSTMpool(nn.Module):

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """
        super(LSTMpool, self).__init__()
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
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        # Define linear layer.
        # Linear layer takes as input the concatenation of the last output, the
        # mean pooling and the max pooling of all the outputs of LSTM.
        self.linear = nn.Linear(3 * self.hidden_size, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # obtain the model's device ID
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define useful variables
        batch_size = len(x)
        seq_len = x.size(1)
        # Embed the words, using the embedding layer
        embeddings = self.embeddings(x)  # EX6
        # Set initial hidden and cell states for the lstm layer.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        # Forward propagate LSTM
        # lstm_out: tensor of shape (batch_size, max_length, hidden_size)
        lstm_out, _ = self.lstm(embeddings, (h0, c0))
        # Compute last output of LSTM for each sentence.
        last = torch.zeros(batch_size, self.hidden_size).float().to(DEVICE)
        for i in range(batch_size):
            if lengths[i] > seq_len:
                last[i] = lstm_out[i, seq_len-1, :]
            else:
                last[i] = lstm_out[i, lengths[i]-1, :]
        # Comput mean of the outputs of the LSTM for each sentence.
        mean_pool = torch.zeros(batch_size, self.hidden_size).float().to(DEVICE)
        for i in range(batch_size):
            if lengths[i] > seq_len:
                mean_pool[i] = torch.sum(lstm_out[i, :seq_len, :], dim=0)
            else:
                mean_pool[i] = torch.sum(lstm_out[i, :lengths[i], :], dim=0)
        # Compute max value of the outputs in each dimension for each sentence.
        max_pool = torch.zeros(batch_size, self.hidden_size).float().to(DEVICE)
        for i in range(batch_size):
            if lengths[i] > seq_len:
                max_pool[i],_ = torch.max(lstm_out[i, :seq_len, :], dim=0)
            else:
                max_pool[i],_ = torch.max(lstm_out[i, :lengths[i], :], dim=0)

        # Concatenate above three representations
        representations = torch.cat((last, mean_pool, max_pool), 1)
        # Project representation to output
        logits = self.linear(representations)

        return logits
