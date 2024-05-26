from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd
from scipy import sparse

X = sparse.load_npz('Xtrain_matrix.npz')
y = pd.read_csv('ytrain.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#on prend les modèles qui ont fonctionné auparavant
print("Voting Classifier")
clf1 = KNeighborsClassifier(n_neighbors=40)
clf2 = RandomForestClassifier(n_jobs = -1,max_depth=None, min_samples_leaf=1, min_samples_split=5, n_estimators=200)
clf3 = LogisticRegression(multi_class='multinomial',class_weight= "balanced", max_iter=1000)

vc = VotingClassifier(estimators=[('knn', clf1), ('rf', clf2), ('lr', clf3)], voting='hard')
vc.fit(X_train, y_train["prdtypecode"])

y_pred = vc.predict(X_test)

print( classification_report(y_test["prdtypecode"], y_pred) )
print(vc.score(X_test, y_test["prdtypecode"]
               ))

import joblib
joblib.dump(vc, 'VotingClassifier.pkl')
"""
import numpy as np
X_test = sparse.load_npz('Xtest_matrix.npz')
y_pred = vc.predict(X_test)
y_pred = y_pred.astype(int)
np.savetxt('ytest_pred.csv', y_pred.astype(int), delimiter=',')
"""
"""
 Voting Classifier
              precision    recall  f1-score   support

           0       0.40      0.67      0.50       618
           1       0.72      0.62      0.67       468
           2       0.78      0.79      0.78       329
           3       0.95      0.83      0.88       166
           4       0.72      0.82      0.77       523
           5       0.91      0.93      0.92       799
           6       0.68      0.59      0.63       152
           7       0.70      0.54      0.61       972
           8       0.69      0.48      0.57       390
           9       0.80      0.93      0.86      1023
          10       0.89      0.90      0.89       146
          11       0.79      0.75      0.77       501
          12       0.81      0.77      0.79       645
          13       0.78      0.80      0.79      1043
          14       0.87      0.90      0.89       884
          15       0.81      0.88      0.84       144
          16       0.78      0.76      0.77      1018
          17       0.89      0.80      0.84       164
          18       0.69      0.81      0.74       994
          19       0.80      0.70      0.74       956
          20       0.74      0.79      0.76       285
          21       0.92      0.91      0.92       931
          22       0.84      0.70      0.76       522
          23       0.96      0.97      0.96      2091
          24       0.85      0.72      0.78       501
          25       0.83      0.66      0.74       567
          26       0.99      0.94      0.96       152

    accuracy                           0.79     16984
   macro avg       0.80      0.78      0.78     16984
weighted avg       0.80      0.79      0.80     16984
"""