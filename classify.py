from model_tfidf import Dict_Tfidf
from sklearn.externals import joblib
import exception
import argparse
from utils import util
PATH = "./data/train.crash"

def classify():
	ap = argparse.ArgumentParser()
	ap.add_argument("-t", "--text",help = "nhap doan text!!!")
	args = vars(ap.parse_args())
	text = args["text"]
	Util = util()
	text = Util.text_util_final(text)

	if(text.find("nhưng",0,len(text)) > -1 or text.find("nhung",0,len(text)) > -1):
		exception.exist_nhung(text)
		exit(0)
	if((text.find("được mỗi",0,len(text) > -1)) or (text.find("được cái",0,len(text)) >-1)):
		print("Day la binh luan tieu cuc!")
		exit(0)

	text = [Util.text_util_final(text)]
	dict_tfidf = Dict_Tfidf(PATH)
	vectorizer = dict_tfidf.create_dict_tfidf()	
	vector_tfidf = vectorizer.transform(text)

	model= joblib.load('./models/best_model.pkl')
	label = model.predict(vector_tfidf)
	if(label[0] == 1):
		print("Day la binh luan tieu cuc!")
	else:
		print("Day la binh luan tich cuc!")
if __name__ == '__main__':
	classify()