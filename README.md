# 🏠 Real Estate Price Prediction

**Smart tool to predict flat prices and investment potential in India**

---

## 📱 Quick Start

```bash
# 1. Train the model
python src/train.py

# 2. Run the app
streamlit run app.py
```

Open your browser → **http://localhost:8501**

---

## 🎯 Features

### 1. **EDA (Exploratory Data Analysis)**
- Visualize market trends
- Analyze price distributions
- Discover property patterns

### 2. **Prediction**
- Enter property details (BHK, Size, Price, Age, etc.)
- Get **investment recommendation** (Good/Not Good)
- See **expected price after 5 years**

### 3. **Feature Importance**
- Understand which factors drive prices
- View detailed analysis charts

### 4. **Model Info**
- Learn about the ML models
- Check accuracy metrics

---

## 📥 Input Parameters

| Parameter | Range | Example |
|-----------|-------|---------|
| BHK | 1-10 | 3 |
| Size (SqFt) | 300-5000 | 1500 |
| Current Price (Lakhs) | 10-500 | 75 |
| Property Age (Years) | 0-50 | 5 |
| Nearby Schools | 0-10 | 3 |
| Nearby Hospitals | 0-10 | 2 |
| Parking Spots | 0-5 | 1 |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| RMSE | 0.0033 |
| Algorithm | Random Forest |
| Estimators | 100 |

---

## 🔍 How to Use

### **Step 1: Explore Data**
- Go to **EDA** tab
- View market insights
- Identify trends

### **Step 2: Make Predictions**
- Go to **Prediction** tab
- Fill in property details
- Click **"Predict"**
- Get instant recommendations

### **Step 3: Understand Factors**
- Check **Feature Importance** tab
- See what matters most for pricing

---

## 📁 Project Structure

```
Flat_Price_Prediction/
├── src/
│   ├── train.py           # Train models
│   ├── predict.py         # Prediction logic
│   ├── preprocess.py      # Data processing
│   ├── config.py          # Configuration
│   ├── evaluate.py        # Model evaluation
│   └── eda.py             # Data analysis
├── data/
│   └── india_housing_prices.csv
├── models/
│   ├── model.pkl          # Trained models
│   └── scaler.pkl         # Data scaler
├── app.py                 # Main Streamlit app
└── README.md
```

---

## 🛠 Tech Stack

- **Python 3.8+**
- **Streamlit** - Web UI
- **Scikit-Learn** - ML Models
- **Pandas & NumPy** - Data processing
- **Matplotlib & Seaborn** - Visualizations

---

## ⚡ Features Used

✅ Classification (Investment Quality)
✅ Regression (Future Price)
✅ Feature Engineering
✅ Data Scaling & Encoding
✅ Interactive Visualizations

---

## 🚀 Tips

💡 **For best predictions**, use properties similar to training data
💡 **Economic factors** affect actual market prices
💡 **Historical data** improves accuracy

---

## 📞 Need Help?

1. **Model not found?** → Run `python src/train.py` first
2. **Import errors?** → Check virtual environment is activated
3. **Slow predictions?** → First load is slower (model loading)

---

## 📝 License

Educational Project - GUVI

---

**Made with ❤️ for Real Estate Analysis**
