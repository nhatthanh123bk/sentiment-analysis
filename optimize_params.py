import pandas as pd
from preprocess import DataSource
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib
from model_tfidf import Dict_Tfidf

PATH = "./data/train.crash"

def create_tfidf_vector(path):
    dict_tfidf = Dict_Tfidf(PATH)
    vectorizer = dict_tfidf.create_dict_tfidf()

    ds = DataSource()
    train_data = pd.DataFrame(ds.load_data(path))
    x_train = train_data.review
    y_train = train_data.label

    x_train_tfdf = vectorizer.transform(x_train)
    return x_train_tfdf,y_train

def turn_params():
    x_train_tfdf,y_train = create_tfidf_vector(PATH)
    parameter_candidates = [
      {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000], 'kernel': ['linear']},
    ]

    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=1,cv = 5,scoring = "f1")
    print("mode istraining....")
    clf.fit(x_train_tfdf, y_train)
    joblib.dump(clf.best_params_, './models/best_model.pkl', compress = 1)
    print('Best score:', clf.best_score_)
    print("Best parameter_gram:",clf.best_params_)

if __name__ == '__main__':
    turn_params()
