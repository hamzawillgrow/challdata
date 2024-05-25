import pandas as pd
import re
import spacy
import numpy as np
from scipy import sparse

X = sparse.load_npz('Xtrain_matrix.npz')
y = pd.read_csv('ytrain.csv')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn import linear_model

clf = linear_model.LogisticRegression(multi_class='multinomial',class_weight= "balanced", max_iter=1000)
clf.fit(X_train, y_train['prdtypecode'])

y_pred = clf.predict(X_test)

y_pred = pd.DataFrame(y_pred, columns=['prdtypecode'])

#  Classe classification_report pour afficher les résultats 
from sklearn.metrics import classification_report

result = classification_report(y_test['prdtypecode'], y_pred)
print("Classification report :")
print(result)

print("Score : "+str(clf.score(X_test, y_test['prdtypecode'])))