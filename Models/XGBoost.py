#importing libraries
import pandas as pd
import numpy as np
import xgboost
import scipy.sparse as sparse

X = sparse.load_npz('Xtrain_matrix.npz')
y = pd.read_csv('ytrain.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from xgboost import XGBClassifier

xgb_clf = XGBClassifier()

xgb_clf.fit(X_train, y_train['prdtypecode'])
y_pred = xgb_clf.predict(X_test)

from sklearn.metrics import classification_report
print( classification_report(y_test['prdtypecode'], y_pred) )
print( xgb_clf.score(X_test, y_test['prdtypecode']))

import joblib
joblib.dump(xgb_clf, 'XGBoost_model.pkl')
