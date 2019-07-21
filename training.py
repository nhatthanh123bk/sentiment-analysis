import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib
from preprocess import DataSource
from model_tfidf import Dict_Tfidf
PATH = "./data/train.crash"

def create_tfidf_vector(path):
    dict_tfidf = Dict_Tfidf(PATH)
    vectorizer = dict_tfidf.create_dict_tfidf()

    ds = DataSource()
    train_data = pd.DataFrame(ds.load_data(path))
    x_train = train_data.review
    y_train = train_data.label

    x_train_tfidf = vectorizer.transform(x_train)
    return x_train_tfidf,y_train

def training():
    x_train_tfidf,y_train = create_tfidf_vector(PATH)
    model = SVC(C=1,kernel='linear')
    model.fit(x_train_tfidf,y_train)
    joblib.dump(model, './models/best_model.pkl', compress = 1)

if __name__ == '__main__':
    training()