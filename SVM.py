from sklearn import svm
from sklearn import model_selection
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import classification_report


X = sparse.load_npz('Xtrain_matrix.npz')
y = pd.read_csv('ytrain.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf_svm = svm.SVC( gamma = 0.01, kernel = 'poly')
clf_svm.fit(X_train, y_train['prdtypecode'])

y_pred = clf_svm.predict(X_test)
# Calcul et affichage de classification_report
print( classification_report(y_test['prdtypecode'], y_pred) )
clf_svm.score(X_test, y_test['prdtypecode'])