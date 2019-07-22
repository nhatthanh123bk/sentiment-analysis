from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
from preprocess import DataSource
from utils import util

class Dict_Tfidf():
	def __init__(self,path):
		self.path = path
	
	def create_dict_tfidf(self):
		ds = DataSource()
		dict_data = pd.DataFrame(ds.load_data(self.path)).review
		dict_data = dict_data.tolist()
		Util = util()
		A = []
		for i in range(len(dict_data)):
			text = dict_data[i]
			text = Util.text_util_final(text)
			A.append(text)
		vectorizer = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
		vectorizer.fit(A)
		return vectorizer

if __name__ == '__main__':
	DT = Dict_Tfidf("./data/train.crash")
	DT.create_dict_tfidf()