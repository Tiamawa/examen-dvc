import pandas as pd
from joblib import load
import json
from pathlib import Path

from sklearn.metrics import r2_score

X_test = pd.read_csv('./data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('./data/processed_data/y_test.csv')

def main(repo_path):
    model = load(repo_path / "models/trained_svr_model.joblib")
    predictions = model.predict(X_test)
    r2 = (y_test, predictions)
    metrics = {"R2 score": r2}
    r2_path = repo_path / "metrics/scores.json"
    r2_path.write_text(json.dumps(metrics))

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)

