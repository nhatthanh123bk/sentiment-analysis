import pandas as pd
from preprocess import DataSource
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from model_tfidf import Dict_Tfidf
from utils import util

PATH = "./data/train.crash"

def create_tfidf_vector(path):
    dict_tfidf = Dict_Tfidf(PATH)
    vectorizer = dict_tfidf.create_dict_tfidf()

    #load du lieu
    ds = DataSource()
    train_data = pd.DataFrame(ds.load_data(path))
    x_train = train_data.review
    y_train = train_data.label
    
    # chuan hoa lai du lieu
    x_train = x_train.tolist()
    Util = util()
    A = []
    for i in range(len(x_train)):
        text = x_train[i]
        text = Util.text_util_final(text)
        A.append(text)        

    #Chuyen ve vector tfidf    
    x_train_tfidf = vectorizer.transform(A)
    return x_train_tfidf,y_train
def turn_params():
    x_train_tfdf,y_train = create_tfidf_vector(PATH)
    parameter_candidates = [
      {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000], 'kernel': ['linear']},
    ]

    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=1,cv = 5,scoring = "f1")
    print("mode istraining....")
    clf.fit(x_train_tfdf, y_train)
    print('Best score:', clf.best_score_)
    print("Best parameter_gram:",clf.best_params_)

if __name__ == '__main__':
    turn_params()
