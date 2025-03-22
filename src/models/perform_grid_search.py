import sklearn
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
from pathlib import Path

X_train = pd.read_csv('./data/processed_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv')

# grid search on Support Vector Regressor : using R2 to evaluate and get the best model
svr = SVR()
param_grid_svr = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "epsilon": [0.01, 0.1, 0.2]
}

grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search_svr.fit(X_train, y_train)

# save best params to .pkl file
repo_path = Path(__file__).parent.parent.parent
best_params = grid_search.best_params_

with open(repo_path / "models/svr_best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

