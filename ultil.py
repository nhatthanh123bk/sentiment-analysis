from preprocess import DataSource
import pandas as pd

ds = DataSource()
train_data = pd.DataFrame(ds.load_data("./data/train.crash"))
x_src_train = train_data.review
y_train = train_data.label

x_src_train = x_src_train.tolist()

for i in range(len(x_src_train)):
	print(x_src_train[i])
	break


