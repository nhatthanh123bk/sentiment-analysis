from model_tfidf import Dict_Tfidf
from sklearn.externals import joblib
import argparse
from utils import util
PATH = "./data/train.crash"

def classify():
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--text",help = "nhap doan text!!!")
	args = vars(ap.parse_args())
	text = args["text"]

	dict_tfidf = Dict_Tfidf(PATH)
	vectorizer = dict_tfidf.create_dict_tfidf()
	Util = util()
	text = [Util.text_util_final(text)]
	print(text)
	vector_tfidf = vectorizer.transform(text)

	model= joblib.load('./models/best_model.pkl')
	label = model.predict(vector_tfidf)
	if(label[0] == 1):
		print("Day la binh luan tieu cuc!")
	else:
		print("Day la binh luan tich cuc!")
if __name__ == '__main__':
	classify()