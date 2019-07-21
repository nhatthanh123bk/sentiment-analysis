from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
from preprocess import DataSource

class Dict_Tfidf():
	def __init__(self,path):
		self.path = path
	
	def create_dict_tfidf(self):
		ds = DataSource()
		dict_data = pd.DataFrame(ds.load_data(self.path)).review
		vectorizer = TfidfVectorizer(max_features=100000,ngram_range=(1, 3))
		vectorizer.fit(dict_data)
		return vectorizer
