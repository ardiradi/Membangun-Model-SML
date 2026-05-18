"""
modelling_tuning.py
Advanced Model Training with Manual Logging & Hyperparameter Tuning (Kriteria 2 - Skilled/Advanced)

Melatih model ML dengan hyperparameter tuning menggunakan GridSearchCV,
lalu melakukan manual logging ke MLflow (bukan autolog).
Termasuk artefak tambahan: confusion matrix, feature importance, ROC curve, dll.

Jika DagsHub tersedia, artefak akan disimpan secara online.

Author: ardir
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    log_loss
)
from sklearn.preprocessing import label_binarize
import joblib
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


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Membuat dan menyimpan confusion matrix sebagai gambar."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix disimpan: {save_path}")


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    """Membuat dan menyimpan plot feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    plt.barh(range(top_n), importances[indices][::-1], color=colors)
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Feature importance disimpan: {save_path}")


def plot_roc_curves(y_true, y_proba, classes, save_path):
    """Membuat dan menyimpan ROC curves untuk multi-class."""
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    
    plt.figure(figsize=(8, 6))
    colors = ['#2196F3', '#4CAF50', '#FF5722']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                 label=f'Class {classes[i]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Multi-class)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] ROC curves disimpan: {save_path}")


def plot_learning_metrics(metrics_history, save_path):
    """Membuat plot perbandingan metrik train vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metric_names = list(metrics_history.keys())
    for idx, metric in enumerate(metric_names[:2]):
        ax = axes[idx]
        values = metrics_history[metric]
        ax.bar(['Train', 'Validation', 'Test'], values, 
               color=['#2196F3', '#4CAF50', '#FF5722'])
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Learning metrics disimpan: {save_path}")


def train_with_tuning(use_dagshub=False, dagshub_owner=None, dagshub_repo=None):
    """
    Melatih model dengan hyperparameter tuning dan manual logging ke MLflow.
    
    Parameters
    ----------
    use_dagshub : bool
        Jika True, simpan artefak ke DagsHub (Advanced).
    dagshub_owner : str
        Username pemilik repo DagsHub.
    dagshub_repo : str
        Nama repo DagsHub.
    """
    # Setup MLflow
    if use_dagshub and dagshub_owner and dagshub_repo:
        try:
            import dagshub
            dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
            print(f"[INFO] DagsHub terinisialisasi: {dagshub_owner}/{dagshub_repo}")
        except ImportError:
            print("[WARNING] dagshub package belum terinstall. Menggunakan lokal MLflow.")
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    mlflow.set_experiment("Wine_Quality_Tuning")
    
    # Load data
    print("[INFO] Loading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data()
    feature_names = X_train.columns.tolist()
    classes = sorted(np.unique(y_train))
    class_labels = ['low', 'medium', 'high']
    
    # ==============================================
    # HYPERPARAMETER TUNING dengan GridSearchCV
    # ==============================================
    print("\n[INFO] Menjalankan Hyperparameter Tuning (GridSearchCV)...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    print(f"\n[INFO] Best Parameters: {best_params}")
    print(f"[INFO] Best CV Score: {grid_search.best_score_:.4f}")
    
    # ==============================================
    # MANUAL LOGGING ke MLflow
    # ==============================================
    artifacts_dir = "mlflow_artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with mlflow.start_run(run_name="RF_Tuned_ManualLog"):
        # ----- LOG PARAMETERS -----
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("test_size", len(X_test))
        
        # ----- PREDICTIONS -----
        y_pred_train = best_model.predict(X_train)
        y_pred_val = best_model.predict(X_val)
        y_pred_test = best_model.predict(X_test)
        y_proba_test = best_model.predict_proba(X_test)
        
        # ----- LOG METRICS (sama dengan autolog + tambahan) -----
        # Train metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        train_prec = precision_score(y_train, y_pred_train, average='weighted')
        train_rec = recall_score(y_train, y_pred_train, average='weighted')
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_precision", train_prec)
        mlflow.log_metric("train_recall", train_rec)
        mlflow.log_metric("train_f1_score", train_f1)
        
        # Validation metrics
        val_acc = accuracy_score(y_val, y_pred_val)
        val_prec = precision_score(y_val, y_pred_val, average='weighted')
        val_rec = recall_score(y_val, y_pred_val, average='weighted')
        val_f1 = f1_score(y_val, y_pred_val, average='weighted')
        
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_precision", val_prec)
        mlflow.log_metric("val_recall", val_rec)
        mlflow.log_metric("val_f1_score", val_f1)
        
        # Test metrics
        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, average='weighted')
        test_rec = recall_score(y_test, y_pred_test, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        test_logloss = log_loss(y_test, y_proba_test)
        
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_precision", test_prec)
        mlflow.log_metric("test_recall", test_rec)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_log_loss", test_logloss)
        
        # CV score
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # ----- LOG ARTEFAK TAMBAHAN (Advanced: minimal 2 artefak tambahan) -----
        
        # Artefak 1: Confusion Matrix
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_test, y_pred_test, class_labels, cm_path)
        mlflow.log_artifact(cm_path)
        
        # Artefak 2: Feature Importance
        fi_path = os.path.join(artifacts_dir, "feature_importance.png")
        plot_feature_importance(best_model, feature_names, fi_path)
        mlflow.log_artifact(fi_path)
        
        # Artefak 3: ROC Curves
        roc_path = os.path.join(artifacts_dir, "roc_curves.png")
        plot_roc_curves(y_test, y_proba_test, classes, roc_path)
        mlflow.log_artifact(roc_path)
        
        # Artefak 4: Classification Report (JSON)
        report = classification_report(y_test, y_pred_test, output_dict=True)
        report_path = os.path.join(artifacts_dir, "classification_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        
        # Artefak 5: Metrics Comparison Plot
        metrics_history = {
            'Accuracy': [train_acc, val_acc, test_acc],
            'F1-Score': [train_f1, val_f1, test_f1],
        }
        metrics_path = os.path.join(artifacts_dir, "metrics_comparison.png")
        plot_learning_metrics(metrics_history, metrics_path)
        mlflow.log_artifact(metrics_path)
        
        # Artefak 6: Best params (JSON)
        params_path = os.path.join(artifacts_dir, "best_params.json")
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2, default=str)
        mlflow.log_artifact(params_path)
        
        # ----- LOG MODEL -----
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        # Print results
        print(f"\n{'='*60}")
        print(f"  MODEL TRAINING RESULTS (Tuned)")
        print(f"{'='*60}")
        print(f"  Train Accuracy:  {train_acc:.4f}")
        print(f"  Val Accuracy:    {val_acc:.4f}")
        print(f"  Test Accuracy:   {test_acc:.4f}")
        print(f"  Test F1-Score:   {test_f1:.4f}")
        print(f"  Test Log Loss:   {test_logloss:.4f}")
        print(f"  Best CV Score:   {grid_search.best_score_:.4f}")
        print(f"{'='*60}")
        print(f"\n[Classification Report - Test Set]")
        print(classification_report(y_test, y_pred_test, target_names=class_labels))
        
        run_id = mlflow.active_run().info.run_id
        print(f"\n[INFO] MLflow Run ID: {run_id}")
        print(f"[INFO] Semua artefak berhasil di-log ke MLflow.")
        
        return best_model, run_id


if __name__ == '__main__':
    # Untuk Advanced: gunakan DagsHub
    # train_with_tuning(use_dagshub=True, dagshub_owner='<username>', dagshub_repo='<repo_name>')
    
    # Untuk Skilled: gunakan lokal
    train_with_tuning(use_dagshub=False)
