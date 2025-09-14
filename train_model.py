import pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from api.utils import normalize_columns, coerce_times, clean_data, add_features, featurize_for_model
import gc

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "datasets" / "courier_dataset.csv"

def main():
    print("🚀 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    gc.collect()
    
    df = normalize_columns(df)
    df = coerce_times(df)
    
    # Clean data first to remove outliers
    df = clean_data(df)
    
    # HAPA NDIO TATIZO: Usifute trip_km kabla ya add_features!
    # Weka tu columns zote muhimu
    cols_to_keep = ['from_city_name', 'poi_lat', 'poi_lng', 'receipt_lat', 
                   'receipt_lng', 'receipt_time', 'delivery_duration_min', 'trip_km']
    df = df[cols_to_keep].copy()
    
    gc.collect()
    
    df = add_features(df)
    gc.collect()

    print("🔧 Featurizing...")
    X, y, encoders = featurize_for_model(df, fit_encoders=True)
    
    print(f"📊 Final dataset: X={X.shape}, y={y.shape}")
    
    del df
    gc.collect()

    print("🤖 Training model...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42, 
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    
    print("✅ MAE:", mae)
    print("✅ R²:", r2)
    
    # Additional diagnostics
    print(f"📈 Target stats: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}")
    print(f"📊 Prediction stats: min={pred.min():.1f}, max={pred.max():.1f}, mean={pred.mean():.1f}")

    # Save model and encoders
    (BASE_DIR / "ml_models").mkdir(exist_ok=True)
    joblib.dump(model, BASE_DIR / "ml_models" / "model.pkl")
    joblib.dump(encoders, BASE_DIR / "ml_models" / "encoders.pkl")

    print("🎉 Model and encoders saved to ml_models/")

if __name__ == "__main__":
    main()