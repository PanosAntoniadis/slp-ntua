from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from nltk.tokenize import TweetTokenizer
import string
from nltk.corpus import stopwords

DATASET = "MR"  # options: "MR", "Semeval2017A"

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
        # ------------ #
        #     EX2      #
        # ------------ #
        # tokenize each training example.
        if DATASET == "Semeval2017A":
            # use tweet tokenizer to tokenize tweets dataset.
            tweetToken = TweetTokenizer()
            self.data = [tweetToken.tokenize(example) for example in X]
        elif DATASET == "MR":
            self.data = []
            table = str.maketrans('', '', string.punctuation)
            stop_words = set(stopwords.words('english'))
            for example in X:
                # split into tokens by white space
                tokens = example.split()
                # remove punctuation from each token
                tokens = [w.translate(table) for w in tokens]
                # remove remaining tokens that are not alphabetic
                tokens = [word for word in tokens if word.isalpha()]
                # filter out stop words
                tokens = [w for w in tokens if not w in stop_words]
                # filter out short tokens
                tokens = [word for word in tokens if len(word) > 1]
                self.data.append(tokens)
        else:
            raise ValueError("Invalid dataset")
        self.labels = y
        self.word2idx = word2idx

        # raise NotImplementedError

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
        # ------------ #
        #     EX3      #
        # ------------ #
        # Decide best length by running bestLength.py
        if DATASET == "Semeval2017A":
            bestLength = 40
        elif DATASET == "MR":
            bestLength = 50
        else:
            raise ValueError("Invalid dataset")
        # Initialize the ndarray that will contain the encoded form of a sentence.
        example = np.zeros(bestLength, dtype = np.int64)
        # Get the sentence by its index.
        data_idx = self.data[index]
        #print("Tokenized sentence")
        #print(data_idx)
        for i in range(min(bestLength, len(data_idx))):
            if data_idx[i] in self.word2idx:
                example[i] = self.word2idx[data_idx[i]]
            else:
                example[i] = self.word2idx["<unk>"]
        label = self.labels[index]
        length = len(data_idx)
        return example, label, length
        # raise NotImplementedError
