# 📦 Heart Disease MLOps — Model Building

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MLflow-2.19.0-blue?logo=mlflow&logoColor=white" alt="MLflow">
  <img src="https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Accuracy-91.8%25-brightgreen" alt="Accuracy">
</p>

## 📋 Overview

Tahap **Membangun Model** dari end-to-end MLOps pipeline. Melakukan training, hyperparameter tuning, dan experiment tracking menggunakan **MLflow** untuk model klasifikasi penyakit jantung.

## 🗂️ Struktur Direktori

```
├── modelling.py                    # Training script (basic)
├── modelling_tuning.py             # Hyperparameter tuning (GridSearchCV)
├── DagsHub.txt                     # DagsHub integration info
├── requirements.txt                # Python dependencies
├── screenshoot_artifak.jpg         # Bukti MLflow artifacts
├── screenshoot_dashboard.jpg       # Bukti MLflow dashboard
├── heart_disease_preprocessing/    # Input data (preprocessed)
└── mlartifacts/                    # MLflow logged artifacts
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── feature_importance.png
    ├── metrics_comparison.png
    ├── best_params.json
    └── random_forest_model/        # Saved model
```

## 🧠 Model Architecture

### Training Script (`modelling.py`)
- **Algorithm**: Random Forest Classifier
- **Experiment Tracking**: MLflow (`Heart_Disease_Basic`)
- **Metrics Logged**: Accuracy, Precision, Recall, F1-Score

### Tuning Script (`modelling_tuning.py`)
- **Method**: GridSearchCV (5-fold cross-validation)
- **Algorithm**: Random Forest Classifier
- **Experiment Tracking**: MLflow (`Heart_Disease_Tuning`)
- **Artifacts**: Confusion Matrix, ROC Curves, Feature Importance, Classification Report

### Hyperparameter Search Space

| Parameter | Values |
|-----------|--------|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 5, 10, 15, None |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

## 📊 Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **91.80%** |
| Test Precision | 91.87% |
| Test Recall | 91.80% |
| Test F1-Score | 91.81% |

## 🚀 Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/ardiradi/Membangun-Model-SML.git
cd Membangun-Model-SML

# Install dependencies
pip install -r requirements.txt

# Training dasar
python modelling.py

# Training dengan hyperparameter tuning
python modelling_tuning.py

# Buka MLflow UI untuk melihat experiment
mlflow ui
```

## 🔗 Related Repositories

| Component | Repository |
|-----------|------------|
| 🔬 Experimentation | [Eksperimen_SML_ardir](https://github.com/ardiradi/Eksperimen_SML_ardir) |
| 🔄 CI/CD Workflow | [Workflow-CI](https://github.com/ardiradi/Workflow-CI) |
| 📊 Monitoring | [Monitoring-Logging-SML](https://github.com/ardiradi/Monitoring-Logging-SML) |

---

<p align="center">
  <b>Part of the Heart Disease MLOps Pipeline</b><br>
  Built as part of Dicoding — Membangun Sistem Machine Learning
</p>
