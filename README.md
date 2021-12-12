# DSC-Final-Project
## Introduction
The objective of this project is to demonstrate the use of machine learning to train on a data set of tweets tagged with sentiment scores, and predict the sentiment of future tweets. Our group was curious whether any semblence of accurate prediction was possible without the use of natual language processing.

## Selection of Data
The model processing and training were conducted using Scikit and Pandas using a dataset of 1.6 million tweets obtained from [Kaggle](https://www.kaggle.com/kazanova/sentiment140). The data features a sentiment value, tweet ID, date, query flag, username, and the text contained in the tweet. 

## Data Preview

![image](https://user-images.githubusercontent.com/54987160/145305379-11a54997-652c-4208-ab31-2e8888e1d511.png)

Our group used a TF-IDF Vectorizer to tokenize the words of the tweets out to a array of 1 and 0 outputs which when put through a Bernoulli Naive Bayes classifier categorizes our scores into their negative and positive weights. We picked Bernoulli Naive Bayes for its binary capabilities as well as being used exactly like this for classifying text documents.

A pipeline was used for feature extraction to feed into the model as the number of features would vary based on the amount of words in the tweets. 

The model was then saved via joblib so that it could be used to deploy the model to other devices without recalculating the massive amount of data.

## Methods
Tools: 
-Scikit-learn (and by extension NumPy), Pandas for data analysis.
-Visual Studio Code as main IDE, Jupyter Notebook for immediate testing. 

Scikit-learn features:
-Bernoulli Naive Bayes Classifier
-TF-IDF Vectorizer for word tokenization.
-Pipeline for feature extraction and feeding the classifier the right stuff. 
## Results
![image](https://user-images.githubusercontent.com/54987160/145731986-f6d84d09-e3c7-4e22-80e0-f251640eba8d.png)

## Discussion
One of the first things that was noticed with this model is that it's accuracy is already decently high from the beginning with just a small sample set. Due to memory constraints we initially created models with only 5000 or 7000 total tweets. Despite these small numbers some of our first accuracy numbers were averaging around 70%! Once it was moved to a machine with far more ram and horsepower behind it we threw in all 1.6 million tweets into the mix and managed to process it all, it raised our accuracy score to a totally respectable 0.7817575% All of that with only a mathematical model and no natural language processing. Taking a look at other examples tackling this problem across the internet and many were only getting scores of around 50% which makes this a great turnout. 

An unexpected restraint with testing the output of the model is actually just the fact that the language of the dataset is dated, from 2009 infact. Surprisingly language has changed a lot and the things people are tweeting about too! For example, we were finding that the word car was being weighed heavily negatively due to the 2009 Toyota recall controversy. 

Our next assumption is that the remaining gap between 78 and >99% accuracy will be exactly that, rooted in language. The model is capable of implying some level of context out of raw weighting, but knowledge of context such as "This is heat" vs "Feeling the heat" would be impossible without further implementation of natural language processing. Additionally, with over 500 million tweets sent every day, is 1.6 million samples truly enough to cover all of human speech? Seeing our sample accuracy go up with sample size alone might hint that we will not see the true accuracy of our model without even more data. 

Further more, a shortcoming of our data set comes from a quality issue. The data set was advertised to have examples of neutrally scored tweets but there was actually none, leaving us no method to train the model on what "neutral" phrases look like. In this way our model is strictly binary and thus certain to lose accuracy in this way. 

Another constraint with processing such massive amounts of data became RAM usage. The final resulting model itself is very small at around 40MB, but the CSV used to train it is almost 3 times as large and once you start including processing it all we actually couldn't process all 1.6 million on the Razer Blade 15 at first. 
## Summary
This project deploys a supervised classification model to predict sentiment score of phrases/future tweets using two categorial features, but a variable number of actual features due to TF-IDF vectorization. 
## References

[Creating a pipeline for TF-IDF Vectorizers](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
[How to create a Scikit pipeline for TF-IDF Vectorizer?](https://stackoverflow.com/questions/63662308/how-to-create-a-scikit-pipeline-for-tf-idf-vectorizer)
[Kaggle Sentiment140 Dataset](https://www.kaggle.com/kazanova/sentiment140)
