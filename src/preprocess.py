import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PATH, SCALER_PATH

def load_data():
    return pd.read_csv(DATA_PATH)

def feature_engineering(df):

    df = df.copy()

    df.drop_duplicates(inplace=True)
    df.ffill(inplace=True)

    df["Price_per_SqFt"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    df["Property_Age"] = 2025 - df["Year_Built"]

    median_price = df["Price_per_SqFt"].median()
    df["Good_Investment"] = (df["Price_per_SqFt"] <= median_price).astype(int)

    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1.08 ** 5)

    return df

def encode_scale(df, training=True):

    df = df.copy()

    encoders = {}

    for col in df.select_dtypes("object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(["Good_Investment", "Future_Price_5Y"], axis=1)
    y_cls = df["Good_Investment"]
    y_reg = df["Future_Price_5Y"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if training:
        # ✅ CREATE MODELS FOLDER IF NOT EXISTS
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

        joblib.dump((scaler, encoders, X.columns.tolist()), SCALER_PATH)

    return X_scaled, y_cls, y_reg, X.columns.tolist()