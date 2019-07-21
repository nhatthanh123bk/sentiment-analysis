from preprocess import DataSource
import numpy as np 
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
PATH = './data/train.crash'

def virtualize_data(path):
	ds = DataSource()
	train_data = pd.DataFrame(ds.load_data(path))
	numpy_label = train_data.label.to_numpy()
	plt.xlabel("number")
	plt.ylabel("name_class")
	plt.title("su phan bo cua cac class")
	plt.hist(numpy_label,rwidth = 1,align ="left")
	plt.savefig('demo.png')

if __name__ == '__main__':
	virtualize_data(PATH)