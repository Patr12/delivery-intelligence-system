import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy.sparse import csr_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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


def rule_based_flags(df):
    """Apply rule-based anomaly detection flags"""
    df = df.copy()
    
    # Duration-based flags
    if 'delivery_duration_min' in df.columns:
        df['flag_duration'] = ((df['delivery_duration_min'] < 5) |  # Less than 5 minutes
                              (df['delivery_duration_min'] > 720)).astype(int)  # More than 12 hours
    else:
        df['flag_duration'] = 0
    
    # Item-based flags (if available)
    if 'num_items' in df.columns:
        df['flag_items'] = (df['num_items'] > 50).astype(int)  # More than 50 items
    else:
        df['flag_items'] = 0
    
    # Distance-based flags
    if 'trip_km' in df.columns:
        df['flag_distance'] = ((df['trip_km'] < 0.1) |  # Less than 100m
                              (df['trip_km'] > 100)).astype(int)  # More than 100km
    else:
        df['flag_distance'] = 0
    
    return df

def isolation_forest_flags(df, feature_cols, contamination=0.05):
    """Apply Isolation Forest for anomaly detection"""
    if len(df) < 10:  # Need minimum samples
        df['flag_iforest'] = 0
        df['anomaly_score'] = 0
        return df
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].fillna(0))
    
    # Apply Isolation Forest
    iso = IsolationForest(
        contamination=contamination, 
        random_state=42,
        n_estimators=100
    )
    
    iso_labels = iso.fit_predict(X_scaled)
    df['flag_iforest'] = (iso_labels == -1).astype(int)
    df['anomaly_score'] = iso.decision_function(X_scaled)
    
    return df

def hybrid_anomaly_detection(df, feature_cols):
    """Hybrid anomaly detection combining rules and machine learning"""
    df = rule_based_flags(df)
    df = isolation_forest_flags(df, feature_cols)
    
    # Combine all flags
    flag_columns = [col for col in df.columns if col.startswith('flag_')]
    df['anomaly'] = df[flag_columns].max(axis=1).astype(int)
    
    return df

def calculate_anomaly_metrics(df):
    """Calculate detailed anomaly metrics"""
    metrics = {
        'total_orders': len(df),
        'anomaly_count': df['anomaly'].sum(),
        'anomaly_percentage': (df['anomaly'].sum() / len(df) * 100) if len(df) > 0 else 0,
        'flag_breakdown': {}
    }
    
    # Count each type of flag
    flag_columns = [col for col in df.columns if col.startswith('flag_')]
    for flag in flag_columns:
        metrics['flag_breakdown'][flag] = df[flag].sum()
    
    return metrics

def export_anomalies_to_csv(df, filename='anomalies_report.csv'):
    """Export anomalies to CSV file"""
    anomaly_df = df[df['anomaly'] == 1].copy()
    if not anomaly_df.empty:
        anomaly_df.to_csv(filename, index=False)
        return filename
    return None

