import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
def createAndTest():
    df = pd.read_csv(r'training_data.csv')
    df = shuffle(df)

    X = df.iloc[:1600000,5]
    y = df.iloc[:1600000,0]

    X_train, X_test, y_train, y_test  = train_test_split(X, y, train_size=0.75, random_state=1234, shuffle=True)

    pipe = Pipeline([('tfidf', TfidfVectorizer(analyzer = 'word', lowercase=False)), ('bernoulli', BernoulliNB())])

    pipe.fit(X_train, y_train)

    from joblib import dump
    dump(pipe, 'mymodel.joblib') #save 

    print(pipe.score(X_test, y_test))

def posOrNeg(sentence):
    from joblib import dump, load
    pipe = load('mymodel.joblib')
    x = pipe.predict([sentence])
    if x == 0:
        return "Negative"
    else:
        return "Positive"

def main():
    #createAndTest() latest accuracy score 0.7817575
    senti = input("Enter text to be analyzed: ")
    print(posOrNeg(senti))

if __name__=="__main__":
    # call the main function
    main()