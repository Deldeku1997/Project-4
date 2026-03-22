import os
import sys
import joblib
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_PATH, SCALER_PATH

def predict_property(input_dict):

    clf,reg,features=joblib.load(MODEL_PATH)
    scaler,encoders,_=joblib.load(SCALER_PATH)

    df=pd.DataFrame([input_dict])

    # encode
    for col in encoders:
        if col in df:
            df[col]=encoders[col].transform(df[col].astype(str))

    df=df.reindex(columns=features,fill_value=0)
    df=scaler.transform(df)

    good=clf.predict(df)[0]
    future=reg.predict(df)[0]

    return good,future