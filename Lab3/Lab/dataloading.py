from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import string
from ekphrasis.classes.tokenizer import SocialTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

DATASET = "Semeval2017A"

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """
        # Tokenize each training example.
        if DATASET == "Semeval2017A":
            # Use social tokenizer from ekphrasis to tokenize tweets dataset.
            social_tokenizer = SocialTokenizer().tokenize
            self.data = [social_tokenizer(example) for example in X]
        else:
            raise ValueError("Invalid dataset")

        self.labels = y
        self.word2idx = word2idx

    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['this', 'is', 'really', 'simple']
                self.target[index] = "neutral"

            the function will have to return something like:
            ::
                example = [  533  3908  1387   649   0     0     0     0]
                label = 1
                length = 4
        """
        # Decide seq length by running bestLength.py.
        if DATASET == "Semeval2017A":
            seq_len = 30
        else:
            raise ValueError("Invalid dataset")
        # Initialize the ndarray that will contain the encoded form of a sentence.
        example = np.zeros(seq_len, dtype = np.int64)
        # Get the sentence by its index.
        data_idx = self.data[index]
        for i in range(min(seq_len, len(data_idx))):
            if data_idx[i].lower() in self.word2idx:
                example[i] = self.word2idx[data_idx[i].lower()]
            else:
                example[i] = self.word2idx["<unk>"]
        label = self.labels[index]
        length = len(data_idx)
        return example, label, length
