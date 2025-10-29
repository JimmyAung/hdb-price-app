Goal
-----
Provide a single file `model.pkl` that the Streamlit app can load and call `predict(...)` on.

Requirements
------------
1) Export a scikit-learn Pipeline that includes BOTH:
   - Preprocessing (imputers, encoders, scalers, etc.)
   - Final regressor (e.g., LinearRegression, RandomForestRegressor, XGBRegressor/SGBoost)
   Example: Pipeline([("preprocess", ColumnTransformer(...)), ("regressor", <model>)])

2) The pipeline must accept a pandas DataFrame with these columns (initial UI):
   - floor_area_sqft (numeric)
   - years_of_lease_left (numeric)
   - FlatType (categorical; values in the UI: "3-Room", "4-Room", "5-Room", "Executive")
   [We will add more features later; please keep `OneHotEncoder(handle_unknown="ignore")`]

3) Save with joblib:
   >>> import joblib
   >>> joblib.dump(pipeline, "model.pkl")

4) Optional but helpful:
   - List the final set of feature names your pipeline expects (raw, before preprocessing).
   - State the target column used (e.g., "resale_price").
   - Share your validation scores (R², RMSE) and random_state if relevant.

Minimal export snippet (example)
--------------------------------
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

# X: DataFrame with at least ['floor_area_sqft','years_of_lease_left','FlatType']
# y: Series with target price (e.g., 'resale_price')

num = ["floor_area_sqft","years_of_lease_left"]
cat = ["FlatType"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ]), num),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat),
])

pipe = Pipeline([
    ("preprocess", preprocess),
    ("regressor", LinearRegression()),  # swap to your chosen model if needed
])

pipe.fit(X, y)
joblib.dump(pipe, "model.pkl")
print("Saved model.pkl")

Versions used by the app (requirements.txt)
-------------------------------------------
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.4.2