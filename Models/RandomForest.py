from sklearn import svm
from sklearn import model_selection
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import classification_report
from sklearn import ensemble
from sklearn.metrics import classification_report   

X = sparse.load_npz('Xtrain_matrix.npz')
y = pd.read_csv('ytrain.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf_rf = ensemble.RandomForestClassifier(n_jobs = -1, random_state=654)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 20, 50, 100],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

print("Grid search with the following parameters:")
print(param_grid)
grid_search = model_selection.GridSearchCV(clf_rf, param_grid, cv=5, n_jobs=-1,verbose=1)
grid_search.fit(X_train, y_train["prdtypecode"])
print("Best parameters set found on development set:")
print(grid_search.best_params_)
print("Best score found on development set:")
print(grid_search.best_score_)
print("All scores on development set:")
print(grid_search.cv_results_)

clf_rf = ensemble.RandomForestClassifier(n_jobs = -1, random_state=654, **grid_search.best_params_)

clf_rf.fit(X_train, y_train["prdtypecode"])

y_pred = clf_rf.predict(X_test)

print(classification_report(y_test['prdtypecode'], y_pred))

# Save the model
import joblib
joblib.dump(clf_rf, 'RandomForest.pkl')

print("Best param Max_depth=NONE min_samples_leaf=1 min_samples_split=5 n_estimators=200")
print("Best score found on development set: 0.7364717235024677")
""""
           0       0.42      0.57      0.48       645
           1       0.76      0.45      0.57       492
           2       0.78      0.64      0.70       330
           3       0.95      0.78      0.86       176
           4       0.72      0.77      0.74       533
           5       0.85      0.93      0.89       762
           6       0.85      0.48      0.62       157
           7       0.66      0.57      0.61       983
           8       0.71      0.36      0.48       421
           9       0.79      0.93      0.85      1042
          10       0.98      0.75      0.85       168
          11       0.87      0.52      0.65       490
          12       0.80      0.64      0.72       625
          13       0.68      0.78      0.73      1023
          14       0.88      0.87      0.88       894
          15       0.85      0.64      0.73       145
          16       0.70      0.77      0.73      1021
          17       0.98      0.52      0.68       172
          18       0.59      0.80      0.68       904
          19       0.70      0.70      0.70       919
          20       0.63      0.78      0.70       288
          21       0.82      0.88      0.85      1008
          22       0.78      0.55      0.65       498
          23       0.82      0.98      0.89      2057
          24       0.91      0.53      0.67       499
          25       0.71      0.61      0.66       553
          26       0.98      0.91      0.94       179

    accuracy                           0.75     16984
   macro avg       0.78      0.69      0.72     16984
weighted avg       0.76      0.75      0.74     16984"""