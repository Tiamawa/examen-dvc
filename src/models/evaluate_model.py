import pandas as pd
from joblib import load
import json
from pathlib import Path

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

X_test = pd.read_csv('./data/scaled_data/X_test_scaled.csv')
y_test = pd.read_csv('./data/processed_data/y_test.csv')

def main(repo_path):
    model = load(repo_path / "models/trained_svr_model.joblib")
    predictions = model.predict(X_test)

    # saving predictions into data folder
    predictions_df = pd.DataFrame(predictions, columns=y_test.columns)
    predictions_df.to_csv('./data/predictions.csv', index=False)

    # retrieve predictions score and saving to metrics folder
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    metrics = {"R2 score": r2, "MAE Score": mae, "MSE Score": mse}
    r2_path = repo_path / "metrics/scores.json"
    r2_path.write_text(json.dumps(metrics))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)

