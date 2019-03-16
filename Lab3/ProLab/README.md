Preparatory Lab for NTUA Speech and Language Processing course.

## Prerequisites
The project requires **Python 3**.

#### 1 - Create a Virtual Environment (Optional)
You can use `virtualenv` but we recommend that you use `conda`.
Download the appropriate [Miniconda](https://conda.io/miniconda.html) version for your system. Then follow the installation [instructions](https://conda.io/docs/user-guide/install/linux.html).

#### 2 - Install PyTorch
Follow the instructions from the PyTorch home page: https://pytorch.org/

#### 3 - Install Requirements
```
pip install -r slp-lab3-prep/requirements.txt
```

#### 4 - Download pre-trained Word Embeddings
In order to minimize the memory requirements you can use low dimensional word embeddings,
such as the 50d Glove embeddings. However, if your computer has enough RAM you will get
better results with higher dimensional embeddings.

- [Glove 6B](http://nlp.stanford.edu/data/glove.6B.zip): Generic english word embeddings - 50d, 100d, 200d, & 300d vectors.
- [Glove Twitter](http://nlp.stanford.edu/data/glove.twitter.27B.zip): Twitter specific word embeddings - 25d, 50d, 100d, & 200d vectors

 - [fastText](https://fasttext.cc/docs/en/english-vectors.html): Generic english word embeddings - only 300d vectors.

The project expects the file(s) to be in the `/embeddings` folder.
