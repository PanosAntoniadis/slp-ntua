# Sentiment classification using Deep Neural Networks

The deep learning framework that is used is Pytorch.

## Datasets:

- __Sentence Polarity Dataset 2.__ This dataset contains 5331 positive and 5331 negative movie reviews, from Rotten Tomatoes and  it is a binary classification problem (positive, negative).
- __Semeval 2017 Task4-A 3.__ This dataset contains tweets that are classified in 3 classes (positive, negative, neutral) with 49570 training examples and 12284 test examples.

## Word embeddings:

The embeddings that are used are [Glove embeddings](<https://nlp.stanford.edu/projects/glove/>).

## Prolab:

In Prolab, a simple neural network is implemented using Pytorch as an introduction for the main part of the project. For more details see the prolab report.

## Lab:

In Lab, the following models are implemented:

- LSTM neural networks with different final representations.
- LSTM with attention layer.
- BiLSTM with and without an attention layer.

In addition, a parser has been implemented in order to help user train different models more efficiently.