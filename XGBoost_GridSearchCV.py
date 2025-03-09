#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings("ignore")


# In[6]:


calendar_df = pd.read_csv("C:/Users/ADMIN/Rakamin/Week 16/calendar.csv")
listings_df = pd.read_csv("C:/Users/ADMIN/Rakamin/Week 16/listings.csv")
reviews_df = pd.read_csv("C:/Users/ADMIN/Rakamin/Week 16/reviews.csv")


# In[8]:


# Remove currency symbols and convert 'price' to numeric in listings_df
listings_df['price'] = listings_df['price'].replace('[\$,]', '', regex=True).astype(float)

# Remove currency symbols and convert 'price' to numeric in calendar_df
calendar_df['price'] = calendar_df['price'].replace('[\$,]', '', regex=True).astype(float)

# Fill missing values with median in calendar_df
calendar_df['price'].fillna(calendar_df['price'].median(), inplace=True)

# Fill missing values in listings_df
listings_df['reviews_per_month'].fillna(listings_df['reviews_per_month'].median(), inplace=True)

# Fill missing values in reviews_df
reviews_df['comments'].fillna('No Comments', inplace=True)

# Check missing values after handling
calendar_missing_values_after = calendar_df.isnull().sum()
listings_missing_values_after = listings_df.isnull().sum()
reviews_missing_values_after = reviews_df.isnull().sum()


# Convert 'date' columns to datetime
calendar_df['date'] = pd.to_datetime(calendar_df['date'], errors='coerce')
reviews_df['date'] = pd.to_datetime(reviews_df['date'], errors='coerce')

# Standardize categorical values in 'room_type'
listings_df['room_type'] = listings_df['room_type'].str.strip().str.lower()

# Remove extra spaces in 'reviewer_name'
reviews_df['reviewer_name'] = reviews_df['reviewer_name'].str.strip().str.title()


# Drop duplicates based on the correct column name
listings_df.drop_duplicates(subset=['id'], inplace=True)
reviews_df.drop_duplicates(subset=['reviewer_id'], inplace=True)



# In[10]:


# Mengimpor library tambahan
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Memilih fitur yang relevan untuk model
# Menggunakan fitur seperti 'bedrooms', 'beds', 'host_listings_count', dan 'zipcode'
X = listings_df[['bedrooms', 'beds', 'host_listings_count', 'zipcode']]
y = listings_df['price']

# Menangani nilai kategori dalam 'zipcode' menggunakan LabelEncoder
label_encoder = LabelEncoder()
X['zipcode'] = label_encoder.fit_transform(X['zipcode'])

# Mengisi missing values pada fitur yang masih memiliki nilai kosong
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for GridSearchCV for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 150, 200],  # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate
    'max_depth': [3, 4, 5, 6],  # Depth of the trees
    'subsample': [0.7, 0.8, 0.9, 1.0],  # Fraction of samples used per boosting round
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]  # Fraction of features used per tree
}
# Generate dataset
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=4, noise=0.1, random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XGBoost Regressor model with default parameters
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict with the trained model
y_xgb_pred = xgb_model.predict(X_test)

# Create GridSearchCV with XGBoost Regressor
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb,
                               cv=3, n_jobs=-1, verbose=2)

# Train the model with GridSearchCV
grid_search_xgb.fit(X_train, y_train)

# Display the best hyperparameters from GridSearchCV
best_params_grid_xgb = grid_search_xgb.best_params_
print("Best hyperparameters using GridSearchCV (XGBoost): ", best_params_grid_xgb)

# Predict with the best model
y_xgb_pred_grid = grid_search_xgb.predict(X_test)

# Calculate MAE and RMSE for the best model
xgb_mae_grid = mean_absolute_error(y_test, y_xgb_pred_grid)
xgb_rmse_grid = np.sqrt(mean_squared_error(y_test, y_xgb_pred_grid))

print(f"Mean Absolute Error (MAE) for XGBoost after GridSearchCV: {xgb_mae_grid}")
print(f"Root Mean Squared Error (RMSE) for XGBoost after GridSearchCV: {xgb_rmse_grid}")

# Compute MAPE for the best model
xgb_mape_grid = mean_absolute_percentage_error(y_test, y_xgb_pred_grid)
print(f"Mean Absolute Percentage Error (MAPE) for XGBoost after GridSearchCV: {xgb_mape_grid}")

# K-Fold Cross Validation
kf = KFold(n_splits=5,  shuffle=True, random_state=42)

cv_mae_grid_xgb = cross_val_score(grid_search_xgb.best_estimator_, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
cv_rmse_grid_xgb = cross_val_score(grid_search_xgb.best_estimator_, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

# Evaluasi MAPE dengan k-fold cross-validation
cv_mape_grid_xgb = cross_val_score(grid_search_xgb.best_estimator_, X_train, y_train, cv=kf,
                                   scoring=lambda estimator, X, y: -mean_absolute_percentage_error(y, estimator.predict(X)))

# Display cross-validation results
print(f"K-Fold Cross-Validation MAE for XGBoost (GridSearchCV): {-cv_mae_grid_xgb.mean()}")
print(f"K-Fold Cross-Validation RMSE for XGBoost (GridSearchCV): {-cv_rmse_grid_xgb.mean()}")
print(f"K-Fold Cross-Validation MAPE for XGBoost (GridSearchCV): {-cv_mape_grid_xgb.mean()}")


# In[13]:


joblib.dump(grid_search_xgb, 'XGboost_GridSearchCV.jodlib')


# In[15]:


grid_search_xgb.predict(X_test[:1])


# In[18]:


pip install streamlit


# In[24]:


get_ipython().system('jupyter nbconvert --to script XGBoost_GridSearchCV.ipynb')


# In[ ]:




