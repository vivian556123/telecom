import numpy as np
import scipy.sparse as sp
from scipy.linalg import norm, solve
from scipy.optimize import check_grad
import matplotlib.pyplot as plt
import time

def load_data(file_name_matrix='tfidf_matrix.npz', file_name_feature_names='feature_names.npy',
	      file_name_labels='train_labels.npy', samples_in_train_set=10000,
	      samples_in_test_set=137562):
	# Recuperation des donnees
	TF_IDF_matrix = sp.load_npz(file_name_matrix)
	TF_IDF_feature_names = np.load(file_name_feature_names)
	train_labels = np.load(file_name_labels, allow_pickle=True)
	train_labels_numeric = ((train_labels == 'Oui') + 0)

	X = TF_IDF_matrix[:samples_in_train_set].toarray()
	y = train_labels_numeric[:samples_in_train_set] * 2 - 1

	X_test = TF_IDF_matrix[samples_in_train_set:samples_in_train_set+samples_in_test_set].toarray()
	y_test = train_labels_numeric[samples_in_train_set:samples_in_train_set+samples_in_test_set] * 2 - 1


	# Standardisation des données
	std_X = np.maximum(np.std(X, axis=0), 1e-7)
	X = X / std_X
	X_test = X_test / std_X

	n = X.shape[0]
	m = X.shape[1]

	# Ajout d'une colonne de uns
	eX = np.hstack((np.ones((n,1)), X))
	eX_test = np.hstack((np.ones((n,1)), X_test))

	return eX, y, eX_test, y_test
	
	

