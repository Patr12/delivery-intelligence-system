import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
import gc

# --- Haversine distance (optimized) ---
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance calculation"""
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# --- Normalize columns ---
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

# --- Time parsing ---
def coerce_times(df):
    """Convert datetime columns with proper year handling"""
    datetime_columns = ['receipt_time', 'sign_time']
    
    for c in datetime_columns:
        if c in df.columns:
            # Use 2018 as the year based on your data
            df[c] = pd.to_datetime("2018-" + df[c], 
                                  format="%Y-%m-%d %H:%M:%S", 
                                  errors="coerce")
    return df

# --- Data Cleaning ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by removing outliers and invalid entries"""
    print("🧹 Cleaning data...")
    
    # Remove rows with missing coordinates
    coord_cols = ['poi_lat', 'poi_lng', 'receipt_lat', 'receipt_lng']
    df = df.dropna(subset=coord_cols)
    
    # Calculate delivery duration
    df["delivery_duration_min"] = (df["sign_time"] - df["receipt_time"]).dt.total_seconds()/60
    
    # Remove negative durations and extreme outliers
    original_count = len(df)
    df = df[(df["delivery_duration_min"] > 0) & 
            (df["delivery_duration_min"] < 24*60)]  # Less than 24 hours
    
    # Calculate distance
    df["trip_km"] = haversine_km(
        df["poi_lat"].values, 
        df["poi_lng"].values,
        df["receipt_lat"].values, 
        df["receipt_lng"].values
    )
    
    # Remove unrealistic distances (more than 1000km within Tanzania)
    df = df[df["trip_km"] < 1000]  # Tanzania is about 1000km across
    
    print(f"📊 Removed {original_count - len(df)} outliers ({len(df)} remaining)")
    print(f"📈 Cleaned duration range: {df['delivery_duration_min'].min():.2f} to {df['delivery_duration_min'].max():.2f} minutes")
    print(f"📈 Cleaned distance range: {df['trip_km'].min():.2f} to {df['trip_km'].max():.2f} km")
    
    return df

# --- Add features robust version ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features with memory optimization"""

    print("📈 Adding features...")
    
    # Calculate distance if not already present
    if 'trip_km' not in df.columns and all(col in df.columns for col in ["poi_lat", "poi_lng", "receipt_lat", "receipt_lng"]):
        print("📊 Calculating trip_km...")
        df["trip_km"] = haversine_km(
            df["poi_lat"].values, 
            df["poi_lng"].values,
            df["receipt_lat"].values, 
            df["receipt_lng"].values
        )
    elif 'trip_km' not in df.columns:
        df["trip_km"] = 1.0  # default for user-input

    # Time features only if receipt_time exists
    if "receipt_time" in df.columns:
        df["hour"] = pd.to_datetime(df["receipt_time"]).dt.hour.astype(np.int8)
        df["dow"] = pd.to_datetime(df["receipt_time"]).dt.dayofweek.astype(np.int8)
        df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    else:
        df["hour"] = 0
        df["dow"] = 0
        df["is_weekend"] = 0

    # Speed feature if delivery_duration_min exists
    if "delivery_duration_min" in df.columns:
        df["speed_km_min"] = df["trip_km"] / df["delivery_duration_min"]
        # Remove unrealistic speeds
        df = df[(df["speed_km_min"] > 0) & (df["speed_km_min"] < 2)]
        df["speed_km_min"] = df["speed_km_min"].astype(np.float32)

    df["trip_km"] = df["trip_km"].astype(np.float32)

    return df


# --- Feature engineering for ML ---
def featurize_for_model(df: pd.DataFrame, fit_encoders=True, encoders=None):
    """Memory-efficient feature engineering"""
    
    # Make sure we keep the target variable if it exists
    feature_cols = ["trip_km", "hour", "dow", "is_weekend", "from_city_name"]
    
    # Only include speed_km_min if it exists (for training data)
    if "speed_km_min" in df.columns:
        feature_cols.append("speed_km_min")
    
    if "delivery_duration_min" in df.columns:
        feature_cols.append("delivery_duration_min")
    
    # Keep only columns that actually exist in dataframe
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    df = df[feature_cols].copy()
    
    # Define numerical columns (exclude speed_km_min for prediction)
    num_cols = ["trip_km", "hour", "dow", "is_weekend"]
    
    # Only include speed_km_min if it exists
    if "speed_km_min" in df.columns:
        num_cols.append("speed_km_min")
    
    # Keep only numerical columns that exist
    num_cols = [col for col in num_cols if col in df.columns]
    
    cat_cols = ["from_city_name"] if "from_city_name" in df.columns else []

    # Extract target variable if it exists
    y = None
    if "delivery_duration_min" in df.columns:
        y = df["delivery_duration_min"].values.astype(np.float32)
        # Remove target from features
        df_features = df.drop(columns=["delivery_duration_min"])
    else:
        df_features = df.copy()

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols))

    ct = ColumnTransformer(transformers)
    
    if fit_encoders:
        print("🏗️ Fitting transformers...")
        X = ct.fit_transform(df_features)
        encoders = {"column_transformer": ct}
    else:
        ct = encoders["column_transformer"]
        X = ct.transform(df_features)

    # Convert to sparse matrix if possible to save memory
    if hasattr(X, 'toarray'):
        X = csr_matrix(X)
    
    # Free memory
    gc.collect()
    
    return X, y, encoders
def greedy_assign(orders_df, couriers_df):
    # simple assignment: round-robin
    orders = orders_df.copy()
    couriers = couriers_df.copy()

    # Ensure courier_id exists
    if "courier_id" not in couriers.columns:
        couriers["courier_id"] = range(1, len(couriers)+1)

    assignments = []
    for i, order in orders.iterrows():
        courier = couriers.iloc[i % len(couriers)]
        assignments.append({
            "order_id": order["order_id"],
            "courier_id": courier["courier_id"]
        })
    
    return pd.DataFrame(assignments)
