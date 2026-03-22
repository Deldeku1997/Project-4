import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_data, feature_engineering, encode_scale
from config import MODEL_PATH

try:
    # ---------------- LOAD DATA ----------------
    print("Loading data...")
    df = load_data()
    print(f"[OK] Data loaded: {df.shape}")

    # ---------------- FEATURE ENGINEERING ----------------
    print("Feature engineering...")
    df = feature_engineering(df)
    print("[OK] Features engineered")

    # ---------------- ENCODE & SCALE ----------------
    print("Encoding & scaling...")
    X, yc, yr, features = encode_scale(df, training=True)
    print(f"[OK] Data scaled: {X.shape}")

    # ---------------- SPLIT ----------------
    print("Splitting data...")
    X_train, X_test, yc_train, yc_test = train_test_split(
        X, yc, test_size=0.2, random_state=42
    )

    _, _, yr_train, yr_test = train_test_split(
        X, yr, test_size=0.2, random_state=42
    )
    print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")

    # ---------------- CLASSIFICATION ----------------
    print("Training classification model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, yc_train)
    acc = accuracy_score(yc_test, clf.predict(X_test))
    print(f"[OK] Accuracy: {acc}")

    # ---------------- REGRESSION ----------------
    print("Training regression model...")
    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train, yr_train)
    mse = mean_squared_error(yr_test, reg.predict(X_test))
    rmse = np.sqrt(mse)
    print(f"[OK] RMSE: {rmse}")

    # ---------------- ENSURE MODELS FOLDER EXISTS ----------------
    print("Creating models folder...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"[OK] Models folder: {os.path.dirname(MODEL_PATH)}")

    # ---------------- SAVE MODEL ----------------
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump((clf, reg, features), MODEL_PATH)
    print(f"[OK] Model saved successfully!")
    print(f"[OK] Training complete!")

except Exception as e:
    print(f"[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)