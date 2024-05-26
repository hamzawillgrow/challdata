import pandas as pd
import re
import spacy
import numpy as np
from scipy import sparse
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn import model_selection

X = sparse.load_npz('Xtrain_matrix.npz')
y = pd.read_csv('ytrain.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = neighbors.KNeighborsClassifier(n_jobs=-1)

param_grid = {
    'n_neighbors': [30, 40, 50],
}

print("Grid search with the following parameters:")
print(param_grid)
grid_search = model_selection.GridSearchCV(knn, param_grid, cv=2, n_jobs=-1,verbose=3)

grid_search.fit(X_train, y_train["prdtypecode"])
print("Best parameters set found on development set:")
print(grid_search.best_params_)
print("Best score found on development set:")
print(grid_search.best_score_)
print("All scores on development set:")
print(grid_search.cv_results_)
knn = neighbors.KNeighborsClassifier(**grid_search.best_params_)

# Entrainement du modèle
knn.fit(X_train, y_train['prdtypecode'])

y_pred = knn.predict(X_test)
print(classification_report(y_test['prdtypecode'], y_pred))
print("best param n_neighbors=40 with accuracy 0.73")
# Save the model
import joblib

joblib.dump(knn, 'KNN.pkl')
        
"""
              precision    recall  f1-score   support

           0       0.53      0.31      0.39       640
           1       0.69      0.52      0.59       509
           2       0.66      0.56      0.61       336
           3       0.87      0.82      0.84       170
           4       0.67      0.68      0.68       560
           5       0.82      0.92      0.87       809
           6       0.67      0.52      0.59       155
           7       0.64      0.53      0.58       982
           8       0.66      0.35      0.46       396
           9       0.73      0.93      0.82       949
          10       0.82      0.90      0.86       173
          11       0.69      0.67      0.68       484
          12       0.76      0.65      0.70       617
          13       0.67      0.76      0.71       988
          14       0.81      0.90      0.85       830
          15       0.87      0.77      0.82       168
          16       0.69      0.73      0.71       995
          17       0.77      0.70      0.74       162
          18       0.64      0.65      0.65       971
          19       0.56      0.76      0.65       952
          20       0.50      0.77      0.60       269
          21       0.88      0.87      0.88      1015
          22       0.77      0.60      0.67       534
          23       0.88      0.96      0.92      2084
          24       0.77      0.56      0.65       521
          25       0.77      0.48      0.59       546
          26       0.50      0.83      0.63       169

    accuracy                           0.73     16984
   macro avg       0.72      0.69      0.69     16984
weighted avg       0.73      0.73      0.72     16984
"""