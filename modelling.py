"""
modelling.py
Basic Model Training with MLflow Autolog (Kriteria 2 - Basic)

Melatih model RandomForestClassifier pada Wine Quality dataset
menggunakan MLflow autolog untuk logging otomatis.

Author: ardir
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(data_dir='wine_quality_preprocessing'):
    """Memuat data yang sudah dipreproses."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_val = pd.read_csv(os.path.join(data_dir, 'X_val.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_val = pd.read_csv(os.path.join(data_dir, 'y_val.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_basic_model():
    """
    Melatih model RandomForest dengan MLflow autolog.
    Logging dilakukan secara otomatis oleh MLflow.
    """
    # Set MLflow tracking URI ke lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Wine_Quality_Basic")
    
    # Load data
    print("[INFO] Loading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    
    print(f"[INFO] Training data: {X_train.shape}")
    print(f"[INFO] Validation data: {X_val.shape}")
    print(f"[INFO] Test data: {X_test.shape}")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RF_Basic_Autolog"):
        # Definisikan model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Training
        print("[INFO] Training model...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n[RESULTS]")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Val Accuracy:   {val_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        
        print(f"\n[Classification Report - Test Set]")
        print(classification_report(y_test, y_pred_test))
        
        print(f"[INFO] MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"[INFO] Model dan artefak disimpan di MLflow Tracking UI.")


if __name__ == '__main__':
    train_basic_model()
