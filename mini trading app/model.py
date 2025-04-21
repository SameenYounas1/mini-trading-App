import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from pysimfin import PySimFin
from transformation import transformation
from joblib import dump, load

MODEL_FILENAME = 'optimized_model.joblib'

def load_and_prepare_data():
    pymsimfin = PySimFin()
    companies_df = pd.read_csv('us-companies.csv', sep=';', index_col=0)
    shareprices_df = pd.read_csv('us-shareprices-daily.csv', sep=';', parse_dates=['Date'], index_col=0)
    companies_df = companies_df[~companies_df.index.isna()]
    print(companies_df.head())
    
    shareprices_df = pymsimfin.get_share_prices('TSLA', '2019-01-01', '2024-03-31')
    shareprices_df = shareprices_df.drop(columns=['Dividend Paid', 'Common Shares Outstanding'])
    shareprices_df.rename(columns={
        'Opening Price': 'Open', 
        'Highest Price': 'High', 
        'Lowest Price': 'Low', 
        'Last Closing Price': 'Close', 
        'Trading Volume': 'Volume'
    }, inplace=True)
    shareprices_df['Date'] = pd.to_datetime(shareprices_df['Date'])
    shareprices_df.set_index('Ticker', inplace=True)
    
    df = transformation(shareprices_df)
    df = df.drop(columns=['SimFinId', 'Date'])
    df['next_day_close'] = df['Close'].shift(-1)
     
    cols = list(df.columns)
    def values_imputation(df,cols):
        median_values = df[cols].median()
        filled = df[cols].fillna(median_values)
        return filled
    df_na_filled = values_imputation(df,cols)
    
    return df_na_filled

def train_initial_model(df_na_filled):
    y = df_na_filled['next_day_close']
    X2 = df_na_filled.drop('next_day_close',axis=1).copy()
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [250, 300, 350],  # Number of trees
        'learning_rate': [0.05, 0.1, 0.15],  # Step size shrinkage
        'max_depth': [3, 5, 7],  # Tree depth
        'subsample': [0.7, 0.8, 1.0],  # Fraction of samples per tree
        'colsample_bytree': [0.8, 0.9, 1.0]  # Fraction of features per tree
    }
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='r2',  # Optimize for R^2 score
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    y_pred_xgb_tuned = best_xgb.predict(X_test)
    print(X_train.columns)
    print(X_test.columns)
    return X_train,X_test,y_train,y_test,best_xgb

def train_optimized_model(X_train, y_train, X_test, y_test,important_features):
    important_features = ['Adjusted Closing Price', 'Low', 'Close', 'High', 'Open', 'EMA_10', 'EMA_26', 'Year']
    X_train_reduced = X_train[important_features]
    X_test_reduced = X_test[important_features]
    
    param_grid = {
        'n_estimators': [250, 300, 350],
        'learning_rate': [0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
    grid_search.fit(X_train_reduced, y_train)
    
    return grid_search.best_estimator_


def save_model(model, filename=MODEL_FILENAME):
    dump(model, filename)
    print(f"✅ Model saved successfully as '{filename}'.")

def load_model(filename=MODEL_FILENAME):
    return load(filename)

if __name__ == "__main__":
    df = load_and_prepare_data()
    X_train,X_test,y_train,y_test,best_xgb = train_initial_model(df)

    important_features = ['Adjusted Closing Price', 'Low', 'Close', 'High', 'Open', 'EMA_10', 'EMA_26', 'Year']
    new_best_xgb = train_optimized_model(X_train, y_train, X_test, y_test, important_features)

    save_model(new_best_xgb)
    print("Optimized model trained and saved.")