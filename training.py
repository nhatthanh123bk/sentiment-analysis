import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib
from preprocess import DataSource
from model_tfidf import Dict_Tfidf
from sklearn.model_selection import cross_val_score
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

def training():
    x_train_tfidf,y_train = create_tfidf_vector(PATH)
    model = SVC(C=1,kernel='linear')
    model.fit(x_train_tfidf,y_train)
    joblib.dump(model, './models/best_model.pkl', compress = 1)

if __name__ == '__main__':
    training()