# train_model.py
import pandas as pd, joblib, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from api.utils import normalize_columns, coerce_times, add_features, featurize_for_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datasets" / "courier_dataset.csv"

def main():
    print("🚀 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)
    df = coerce_times(df)
    df["delivery_duration_min"] = (df["sign_time"] - df["receipt_time"]).dt.total_seconds()/60
    df = df.dropna(subset=["delivery_duration_min"])
    df = add_features(df)

    print("🔧 Featurizing...")
    X, y, encoders = featurize_for_model(df, fit_encoders=True)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    print("✅ MAE:", mean_absolute_error(y_val, pred))
    print("✅ R²:", r2_score(y_val, pred))

    joblib.dump(model, BASE_DIR / "ml_models" / "model.pkl")
    joblib.dump(encoders, BASE_DIR / "ml_models" / "encoders.pkl")

    print("🎉 Model and encoders saved to ml_models/")

if __name__ == "__main__":
    main()
