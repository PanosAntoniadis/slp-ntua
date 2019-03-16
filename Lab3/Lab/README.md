# Project Structure

- __datasets:__ The dataset that are used (Semeval 2017A). Download it from [here](<https://github.com/slp-ntua/slp-lab3-prep/tree/master/datasets>).
- __embeddings:__ The embeddings that can be download them from [here](<https://nlp.stanford.edu/projects/glove/>).
- __models:__ Contains the models that are available for training.
- __report:__ The presentation and the evaluation of the models.
- __utils:__ Useful functions for loading the data and embeddings.
- __bestLength.py:__ A script that computes the fixed length of the sentences for zero padding.
- __data.json:__ Json file for neat-vision produced by the LSTM with Attention model.
- __data3-1.json:__ Json file for neat-vision produced by the model with Attention in the embedding layer.
- __labels.json:__ Short desciption of the labels. It will be used by neat-vision.
- __predict-json.py:__ Python scipt that gets as arguments a saved model and the name of the output file and  produces the json file that neat-vision needs (questions 5-2 and 5-3 of the project).
- __predict-validation.py:__ Python scipt that gets as argument a saved model and saves the predictions of the model in the validation set in a file named y_pred.txt (question 5-1 of the project).