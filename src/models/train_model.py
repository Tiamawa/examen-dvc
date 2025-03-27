import pickle
import joblib
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR

X_train = pd.read_csv('./data/scaled_data/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/y_train.csv')

repo_path = Path(__file__).parent.parent.parent

# Load best parameters from the .pkl file
with open(repo_path / "models/svr_best_params.pkl", "rb") as f:
    loaded_params = pickle.load(f)

# Train the model
best_model = SVR(**loaded_params).fit(X_train, y_train)

#--Save the trained model to a file
model_filename = repo_path / "models/trained_svr_model.joblib"
joblib.dump(best_model, model_filename)

print("Model trained and saved successfully.")


