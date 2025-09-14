# api/ml.py
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml_models" / "model.pkl"
ENCODERS_PATH = BASE_DIR / "ml_models" / "encoders.pkl"

# Load model mara moja wakati server inaanza
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)
