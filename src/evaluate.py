import os
import sys
import joblib
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_PATH

clf,reg,features=joblib.load(MODEL_PATH)

imp=clf.feature_importances_

plt.figure(figsize=(8,10))
plt.barh(features,imp)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()