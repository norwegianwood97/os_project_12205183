#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/norwegianwood97/os_project_12205183.git


import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_dataset(dataset_path):
	dataset_df = pd.read_csv(dataset_path)
	return dataset_df


def dataset_stat(dataset_df):	
	n_feats = dataset_df.shape[1] - 1
	# Considering that prof's execution page show that the datasets have 23, 13 features, I thought that I should exclude target from features.
	n_class0 = dataset_df[dataset_df.target == 0].shape[0]
	n_class1 = dataset_df[dataset_df.target == 1].shape[0]
	return n_feats, n_class0, n_class1


def split_dataset(dataset_df, testset_size):
	x_train, x_test, y_train, y_test = train_test_split(dataset_df.drop('target', axis=1), dataset_df['target'], test_size = testset_size)
	return x_train, x_test, y_train, y_test


def decision_tree_train_test(x_train, x_test, y_train, y_test):
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)
	acc = dt_cls.score(x_test, y_test)
	prec = precision_score(y_test, dt_cls.predict(x_test))
	recall = recall_score(y_test, dt_cls.predict(x_test))
	return acc, prec, recall


def random_forest_train_test(x_train, x_test, y_train, y_test):
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)
	acc = rf_cls.score(x_test, y_test)
	prec = precision_score(y_test, rf_cls.predict(x_test))
	recall = recall_score(y_test, rf_cls.predict(x_test))
	return acc, prec, recall


def svm_train_test(x_train, x_test, y_train, y_test):
	svm_pipe = make_pipeline(StandardScaler(), SVC())
	svm_pipe.fit(x_train, y_train)
	acc = svm_pipe.score(x_test, y_test)
	prec = precision_score(y_test, svm_pipe.predict(x_test))
	recall = recall_score(y_test, svm_pipe.predict(x_test))
	return acc, prec, recall


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)


if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)