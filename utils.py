from preprocess import DataSource
import pandas as pd
import re

dict = [["ship", "vận chuyển"],["shop", "cửa hàng"],["m", "mình"],["mik","mình"],["ko","không"],["k","không"],["kh","không"],["khong","không"],["kg","không"],["khg","không"],["tl","trả lời"],
["rep","trả lời"],["r","rồi"],["fb","facebook"],["face","faceook"],["thanks","cảm ơn"],["thank","cảm ơn"],["tks","cảm ơn"],["tk","cảm ơn"],["ok","tốt"],["oki","tốt"],["okie","tốt"],["sp","sản phẩm"],
["dc","được"],["vs","với"],["đt","điện thoại"],["thjk","thích"],["thik","thích"],["qá","quá"],["trể","trễ"],["bgjo","bao giờ"],["h","giờ"],["qa","quá"],["dep","đẹp"],["xau","xấu"],["ib","nhắn tin"],
["cute","dễ thương"],["sz","size"],["good","tốt"],["god","tốt"],["bt","bình thường"]]


class util():
	def remove(self,text): # remove cac ky tu keo dai vd: "cai ao nay dep quaaaaaa" : "cai ao nay dep qua"
		text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
		return text
	
	def A_cvt_a(self,text): # chuyen cac ky tu viet hoa ve cac ky tu viet thuong
		text = text.lower()
		return text

	def utils_data(self,text):
		list_text = text.split(" ")
		for i in range(len(list_text)):
			for j in range(len(dict)):
				if (list_text[i] == dict[j][0]):
					list_text[i] = dict[j][1]
		text = " ".join(list_text)
		return text
	
	def text_util_final(self,text):
		text = self.remove(text)
		text = self.A_cvt_a(text)
		text = self.utils_data(text)
		return text



		