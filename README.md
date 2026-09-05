# 🇮🇳 India House Price Predictor

> An India-wide, locality-aware machine learning application for estimating residential property prices based on location and property characteristics.

[![Python](https://img.shields.io/badge/Python-3.14+-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)](https://streamlit.io/)

---

## 📌 Overview

**India House Price Predictor** is a machine learning project designed to estimate the price of a residential property using its location and physical characteristics.

Instead of treating house prices as a simple relationship between area and price, the project incorporates location hierarchy:

```text
State
  ↓
City
  ↓
Locality
  ↓
Property Characteristics
  ↓
Machine Learning Model
  ↓
Estimated Property Price
```

The application is designed to scale across Indian states, Union Territories, cities and localities.

---

## ✨ Features

### 🔮 Price Prediction

Enter property details and receive:

* Estimated property price
* Estimated price per sq.ft
* Approximate prediction range
* Selected machine learning model

### 📍 Location-Aware Prediction

The application supports a hierarchical location structure:

```text
State / Union Territory
        ↓
      City
        ↓
    Locality
```

The repository contains a location catalogue covering India's states and Union Territories.

### 🏠 Property Features

The model can use:

* Property type
* BHK
* Area in sq.ft
* Bathrooms
* Floor
* Total floors
* Property age
* Furnishing status
* Parking
* Balcony
* State
* City
* Locality

### 📊 Market Analytics

The application provides basic dataset analytics including:

* Number of listings
* Number of states/UTs
* Number of cities
* Median property price
* City-level price comparisons
* Property dataset preview

### 🧪 Model Comparison

The training pipeline evaluates multiple regression algorithms:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Models are evaluated using:

* MAE
* RMSE
* R² Score

The best-performing model is automatically selected.

---

# 🧠 Machine Learning Pipeline

```text
                 Raw Dataset
                      │
                      ▼
              Data Validation
                      │
                      ▼
               Preprocessing
                /          \
               /            \
      Numerical Features   Categorical Features
             │                    │
             ▼                    ▼
        Imputation            Imputation
             │                    │
          Scaling             One-Hot Encoding
               \              /
                \            /
                 ▼          ▼
                ML Model
                    │
                    ▼
             Model Evaluation
                    │
                    ▼
              Best Model
                    │
                    ▼
             Saved Model
                    │
                    ▼
             Streamlit App
```

---

# 🛠️ Technology Stack

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Core programming language |
| Pandas       | Data manipulation         |
| NumPy        | Numerical operations      |
| Scikit-learn | Machine learning          |
| Joblib       | Model persistence         |
| Plotly       | Data visualization        |
| Streamlit    | Web application           |

---

# 📁 Project Structure

```text
india-house-price-predictor/
│
├── data/
│   ├── sample_housing_data.csv
│   └── processed/
│
├── models/
│   ├── house_price_model.joblib
│   ├── metadata.joblib
│   └── metrics.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   └── validate_data.py
│
├── app.py
├── locations.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/india-house-price-predictor.git
cd india-house-price-predictor
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If pip reports cache problems:

```bash
pip install --no-cache-dir -r requirements.txt
```

---

# 🧠 Train the Model

Run:

```bash
python src/train.py
```

The training pipeline:

1. Loads the dataset
2. Separates features and target
3. Detects numerical/categorical features
4. Handles missing values
5. Scales numerical features
6. One-hot encodes categorical features
7. Trains multiple regression models
8. Calculates MAE, RMSE and R²
9. Selects the best model
10. Saves the trained model

Generated files:

```text
models/
├── house_price_model.joblib
├── metadata.joblib
└── metrics.csv
```

---

# 🌐 Run the Application

After training:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 📊 Example Prediction

Example input:

```text
State: Gujarat
City: Ahmedabad
Locality: Ahmedabad Central

Property Type: Apartment
BHK: 3
Area: 1800 sq.ft
Bathrooms: 2
Floor: 5
Total Floors: 12
Property Age: 5 years
Furnishing: Semi-Furnished
Parking: 1
Balcony: 2
```

The application returns:

```text
Estimated Price
₹XX.XX Cr

Estimated Price / sq.ft
₹X,XXX

Estimated Range
₹XX.XX Cr – ₹XX.XX Cr
```

The actual prediction depends on the trained dataset and model.

---

# 📂 Dataset Format

The training CSV should contain:

```text
state
city
locality
property_type
bhk
area_sqft
bathrooms
floor
total_floors
age_years
furnished
parking
balcony
price_inr
```

Example:

```csv
state,city,locality,property_type,bhk,area_sqft,bathrooms,floor,total_floors,age_years,furnished,parking,balcony,price_inr
Gujarat,Ahmedabad,Satellite,Apartment,3,1800,2,5,12,5,Semi-Furnished,1,2,12000000
```

---

# ⚠️ Important Data Disclaimer

The repository currently contains a **synthetic demonstration dataset**.

The synthetic dataset exists so that the complete application can be trained and executed without requiring access to a proprietary or third-party property database.

It should **not** be interpreted as real Indian property-market data.

For real-world property estimation, the dataset should be replaced with a legally obtained and appropriately licensed dataset containing recent property listings or transaction data.

The quality of predictions depends heavily on:

* Dataset size
* Data freshness
* Geographic coverage
* Locality coverage
* Data accuracy
* Feature quality
* Market conditions

Predictions from this project should not be used as a substitute for professional property valuation or financial advice.

---

# 🔬 Future Improvements

Planned improvements include:

* [ ] Integrate real property-market datasets
* [ ] Add more locality-level data
* [ ] Add historical price trends
* [ ] Add price-per-square-foot prediction
* [ ] Add property similarity search
* [ ] Add undervalued/overvalued detection
* [ ] Add interactive India map
* [ ] Add geospatial features
* [ ] Add distance from metro/railway/highways
* [ ] Add neighborhood-level statistics
* [ ] Experiment with XGBoost/CatBoost
* [ ] Hyperparameter optimization
* [ ] Cross-validation
* [ ] Automated model retraining
* [ ] Deploy the application online
* [ ] Add API endpoints
* [ ] Add database support

---

# 🎯 Learning Outcomes

This project demonstrates practical knowledge of:

* Supervised machine learning
* Regression
* Feature engineering
* Data preprocessing
* Categorical encoding
* Numerical scaling
* Train/test splitting
* Model comparison
* Model evaluation
* Model persistence
* Streamlit application development
* Data visualization
* ML deployment workflow

---

# 🤝 Contributing

Contributions are welcome.

Possible contribution areas include:

* Better preprocessing
* Additional datasets
* New regression algorithms
* Improved visualizations
* Location data
* Feature engineering
* Model optimization
* UI improvements

---

## 📜 License

This project is licensed under the MIT License.

Copyright (c) 2026 Meet Kadiya

See the [LICENSE](LICENSE) file for the complete license text.

## 👨‍💻 Author

**Meet Kadiya**

Built as an applied Machine Learning project focused on understanding the complete workflow from data preprocessing to model deployment.
