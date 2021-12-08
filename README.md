# DSC-Final-Project
## Introduction
The objective of this project is to demonstrate the use of machine learning to train on a data set of tweets tagged with sentiment scores, and predict the sentiment of future tweets. Our group was curious whether any semblence of accurate prediction was possible without the use of natual language processing.

## Selection of Data
The model processing and training were conducted using Scikit and Pandas using a dataset of 1.6 million tweets obtained from [Kaggle](https://www.kaggle.com/kazanova/sentiment140). The data features a sentiment value, tweet ID, date, query flag, username, and the text contained in the tweet. 
###Data Preview
![image](https://user-images.githubusercontent.com/54987160/145305379-11a54997-652c-4208-ab31-2e8888e1d511.png)

Our group used a TF-IDF Vectorizer to tokenize the words of the tweets out to a array of 1 and 0 outputs which when put through a Bernoulli Naive Bayes classifier categorizes our scores into their negative and positive weights. We picked Bernoulli Naive Bayes for its binary capabilities as well as being used exactly like this for classifying text documents.

A pipeline was used for feature extraction to feed into the model as the number of features would vary based on the amount of words in the tweets. 

The model was then saved via joblib so that it could be used to deploy the model onto a Raspberry Pi Zero W.

##Methods

##Results

##Discussion

##Summary
