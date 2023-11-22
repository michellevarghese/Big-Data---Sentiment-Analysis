# Big-Data---Sentiment-Analysis

# Design details:
1. Linear and nonlinear classification models were used to make predictions and the difference in their performance was analysed (from the sklearn module).
2. The preprocessing pipeline includes Tokenizer, StopWordsRemover, HashingTF and Inverse Data Frequency (from the PySpark module).

# Surface level implementation about each unit:
1. For linear classification, we used Stochastic Gradient Descent, Multinomial Naive Bayes and Perceptron whereas we used Multilayer perceptron for non-linear classification using ReLU as the activation function.
2. The dataset was sent in batches to a pipeline that applied preprocessing on the train dataset which would make changes in the tweet features and then send it to the models for training.
3. For training the models, the data was fitted using the partial fit function to exhibit incremental learning.
4. The trained models were stored in pickle files which would later on be used for predicting the classes for the test dataset.
5. The results were visualised by plotting the accuracy of the predictions for each batch streamed to the model.
6. After training, the test dataset was streamed and predictions were made for the tweets in the test dataset after applying the same preprocessing as done during the training phase.
7. F1-score, Precision, Recall and Confusion matrices were calculated on the predictions made for the test dataset.
8. K Means MiniBatch clustering was performed after training and obtaining the accuracy. The number of clusters was set to 2. The difference in the performance before and after significant preprocessing like removal of stop words was observed.

# Reason behind design decisions:
1. The ReLU activation function was used for non-linear classification as it
has predominantly given good results when compared to the other
activation functions available.
2. Tokeniser was used to break down the string tweets into tokens, which
were further passed to other functions for preprocessing.
3. StopWordsRemover was implemented as a part of the preprocessing to
filter out the nltk stop words.
4. HashingTF was used to convert the tokenized tweets into vectors of
fixed size.
5. Inverse Data Frequency (IDF) was used to determine how relevant
each word in the tweet was by associating a number with each word in
the tweets which were passed to the function.

# Take Away from the project:
1. Usage of different spark in-built functions
2. Streaming data
3. Operations that can be performed on data frames
4. Effects of parameters on predictions in terms of accuracy and time
taken to execute
5. Incremental learning
   
# Observations and conclusions:
1. After changing the different hyper parameters, it was observed that
using 128 features for HashingTF gives a good result in terms of
accuracy and time taken for execution.
2. Batch size of 15200 was feasible when compared to higher batch size
of 152000 and lower batch size of 1520.
3. ReLU as the activation function for the non-linear classification model of Multilayer Perceptron gave good results. Changing it to tanh did not
show any significant improvement in the accuracy. The difference
between the results from the two activation functions for MultilayerPerceptron is visualised below.
4. Non-linear classification gave prediction results with better accuracy
when compared to the predictions made by linear classification.
5. Decreasing order of accuracy: MLP > MNB > SGD > Perceptron.


