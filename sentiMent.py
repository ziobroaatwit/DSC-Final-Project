import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
#we win
#This function creates the model based on the test data and then prints the accuracy score of the model.
def createAndTest():
    #Importing the data from the Kaggle 1.6m large Sentiment140 data set, then shuffling the data since it is ordered. 
    df = pd.read_csv(r'training_data.csv')
    df = shuffle(df)

    #Creating our X/y feature and output matrices. 
    X = df.iloc[:1600000,5]
    y = df.iloc[:1600000,0]

    #Splitting the test data with a train size of 0.75 
    X_train, X_test, y_train, y_test  = train_test_split(X, y, train_size=0.75, random_state=1234, shuffle=True)

    #Pipeline to vectorize the data, then apply it to a BernoulliNB classifier.
    pipe = Pipeline([('tfidf', TfidfVectorizer(analyzer = 'word', lowercase=False)), ('bernoulli', BernoulliNB())])

    #Fitting the model.
    pipe.fit(X_train, y_train)

    #Dumping our newly made model to a joblib file. Look how handsome it is. 
    from joblib import dump
    dump(pipe, 'mymodel.joblib') #save 

    #Finally printing the score. Hopefully it's a high score as high as my bill for this school.
    print(pipe.score(X_test, y_test))

#This function utilizes the joblib to predict future phrases and potential tweets and identifies them as positive or negative. 
def posOrNeg(sentence):
    from joblib import dump, load   #Impoting the joblib file
    pipe = load('mymodel.joblib')
    x = pipe.predict([sentence]) #Using the model with a passed string.
    if x == 0:
        return "Negative"
    else:
        return "Positive"

#In our main function, we can either create a new joblib or run our main loop.
#Since we'll be deploying to a Raspberry Pi Zero W with a meager 512MB of ram: lets not create a new model there okay?
def main():
    #createAndTest() latest accuracy score 0.7817575
    while True: #Loop pretty self explanatory, collect user input, if not "exit" then use the model to predict a sentiment, else thank our audience and exit. 
        senti = input("Enter text to be analyzed: ").lower()
        if(senti=="exit"):
            print("\nThanks for trying our model!\n")
            break
        print(posOrNeg(senti))

if __name__=="__main__":
    # call the main function
    main()