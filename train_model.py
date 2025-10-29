# train_model.py
# Minimal pipeline (preprocess + LinearRegression) -> saves model.pkl

CSV_PATH = "HDBee_Cleaned.csv"   # <-- change to YOUR cleaned CSV filename
TARGET   = "resale_price"        # <-- change to YOUR price column name

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1) Load data
df = pd.read_csv(CSV_PATH)

# 2) Keep only fields that match current app UI
required = ["floor_area_sqft", "years_of_lease_left", "FlatType", TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")
df = df[required].copy()

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 3) Split
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Preprocess
num_features = ["floor_area_sqft", "years_of_lease_left"]
cat_features = ["FlatType"]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_features),
    ("cat", cat_pipe, cat_features),
])

# 5) Model (easy to swap later to RandomForest/XGBoost)
model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression()),
])

# 6) Fit + quick validation
model.fit(X_tr, y_tr)
pred = model.predict(X_va)
rmse = mean_squared_error(y_va, pred, squared=False)
r2 = r2_score(y_va, pred)
print(f"Validation R²: {r2:.3f}")
print(f"Validation RMSE: S${rmse:,.0f}")

# 7) Save full pipeline
joblib.dump(model, "model.pkl")
print("Saved model to model.pkl")