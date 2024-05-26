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

""" 0       0.00      0.00      0.00       610
           1       0.00      0.00      0.00       474
           2       0.00      0.00      0.00       352
           3       0.00      0.00      0.00       156
           4       0.00      0.00      0.00       535
           5       0.00      0.00      0.00       801
           6       0.00      0.00      0.00       138
           7       0.00      0.00      0.00       978
           8       0.00      0.00      0.00       429
           9       0.00      0.00      0.00      1051
          10       0.00      0.00      0.00       163
          11       0.00      0.00      0.00       475
          12       0.00      0.00      0.00       643
          13       0.00      0.00      0.00      1015
          14       0.00      0.00      0.00       885
          15       0.00      0.00      0.00       148
          16       0.00      0.00      0.00       990
          17       0.00      0.00      0.00       155
          18       0.00      0.00      0.00       947
          19       0.00      0.00      0.00       956
          20       0.00      0.00      0.00       287
          21       0.00      0.00      0.00      1002
          22       0.00      0.00      0.00       511
...
    accuracy                           0.12     16984
   macro avg       0.00      0.04      0.01     16984
weighted avg       0.01      0.12      0.03     16984 """