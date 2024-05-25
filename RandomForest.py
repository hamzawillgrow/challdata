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
