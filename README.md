# Walmart Product Pricing Model

## 📋 Project Overview

This project implements a machine learning-based pricing model for Walmart products using Jupyter notebooks. The system analyzes various product attributes including cost, stock levels, supplier information, and product characteristics to predict optimal pricing strategies with confidence scores.

## 🎯 Features

- **XGBoost Regression Model**: Implements gradient boosting for accurate price predictions
- **Feature Engineering**: Handles missing data with mean imputation and numeric type conversion
- **Confidence Scoring**: Generates mlScore based on tree-level prediction variance
- **Model Persistence**: Saves trained models and preprocessing components
- **Price Prediction**: Generates suggested prices with confidence scores
- **Performance Metrics**: Comprehensive evaluation with MAE, R², and cross-validation

## 📁 Project Structure

```
Walmart/
├── train.ipynb                           # Main training notebook
├── Train_Pricing_Model.ipynb             # Alternative training approach
├── product_cleaned.csv                   # Cleaned product dataset
├── predicted_prices_with_score_cleaned.json  # Model predictions output
├── suggested_price_xgb_model_cleaned.pkl # Trained XGBoost model
├── imputer_cleaned.pkl                   # Data imputation model
├── confidence_scaler.pkl                 # Confidence score scaler
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
└── .gitignore                           # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab (already installed in your environment)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Walmart
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter and run the notebook**
   ```bash
   jupyter notebook
   ```
   Then open `train.ipynb` and run all cells.

## 📊 Data Description

The model uses the following features from the product dataset:

### Core Product Features

- `cost`: Product cost
- `currentPrice`: Current selling price
- `originalPrice`: Original product price
- `margin`: Profit margin

### Inventory Features

- `stock`: Current stock level
- `maxStock`: Maximum stock capacity
- `minStockLevel`: Minimum stock threshold
- `daysUntilExpiry`: Days until product expires
- `isPerishable`: Perishable product flag

### Price Factors

- `priceFactors.expirationUrgency`: Expiration urgency factor
- `priceFactors.stockLevel`: Stock level factor
- `priceFactors.timeOfDay`: Time-based pricing factor
- `priceFactors.demandForecast`: Demand prediction factor
- `priceFactors.competitorPrice`: Competitor pricing factor
- `priceFactors.seasonality`: Seasonal pricing factor
- `priceFactors.marketTrend`: Market trend factor

### Performance Metrics

- `clearanceRate`: Product clearance rate
- `wasteReduction`: Waste reduction percentage

## 🤖 Model Details

### Algorithm

- **XGBoost Regressor** with optimized hyperparameters
- **Parameters**: n_estimators=100, learning_rate=0.1, max_depth=4

### Feature Processing

- **Missing Data**: Mean imputation using SimpleImputer
- **Data Types**: Automatic conversion to numeric types
- **Feature Selection**: 18 carefully selected features

### Performance Metrics

- **MAE (Mean Absolute Error)**: ~450.89
- **R² Score**: ~0.9957
- **Cross-Validation**: 3-fold CV R² mean ~0.9212

### Confidence Scoring

The model generates an `mlScore` (0.70-0.99) based on:

1. Tree-level prediction variance across XGBoost ensemble
2. Standard deviation of predictions
3. Normalized confidence scaling
4. Power transformation for score distribution

## 💻 Usage Examples

### Training the Model

```python
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("product_cleaned.csv")
df = df[df['suggestedPrice'].notnull()]

# Prepare features
selected_features = [
    'cost', 'currentPrice', 'originalPrice', 'margin',
    'stock', 'maxStock', 'minStockLevel', 'daysUntilExpiry', 'isPerishable',
    'priceFactors.expirationUrgency', 'priceFactors.stockLevel', 'priceFactors.timeOfDay',
    'priceFactors.demandForecast', 'priceFactors.competitorPrice',
    'priceFactors.seasonality', 'priceFactors.marketTrend',
    'clearanceRate', 'wasteReduction'
]

# Train model
X = df[selected_features]
y = df['suggestedPrice']

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_imputed, y)
```

### Making Predictions

```python
# Load saved model
import joblib
model = joblib.load("suggested_price_xgb_model_cleaned.pkl")
imputer = joblib.load("imputer_cleaned.pkl")

# Prepare new data
new_data = pd.DataFrame([{
    'cost': 10.0,
    'currentPrice': 15.0,
    'stock': 100,
    # ... other features
}])

# Make prediction
X_new = imputer.transform(new_data[selected_features])
predicted_price = model.predict(X_new)[0]
print(f"Suggested Price: ${predicted_price:.2f}")
```

## ⚙️ Configuration

### Model Parameters

```python
model_params = {
    'n_estimators': 100,      # Number of boosting rounds
    'learning_rate': 0.1,     # Learning rate
    'max_depth': 4,           # Maximum tree depth
    'random_state': 42        # Reproducibility
}
```

### Confidence Score Parameters

- **Base Score**: 0.70 (minimum confidence)
- **Max Score**: 0.99 (maximum confidence)
- **Power Transform**: 0.3 (score distribution)

## 📈 Output Files

### Predictions JSON

```json
[
  {
    "productId": "12345",
    "suggestedPrice_predicted": 12.99,
    "mlScore": 0.85
  }
]
```

### Model Files

- `suggested_price_xgb_model_cleaned.pkl`: Trained XGBoost model
- `imputer_cleaned.pkl`: Data imputation model
- `confidence_scaler.pkl`: Confidence score scaler

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For questions or support, please open an issue in the repository or contact the development team.

---

**Note**: This model is designed for educational and research purposes. Always validate predictions in production environments and consider business constraints when implementing pricing strategies.
