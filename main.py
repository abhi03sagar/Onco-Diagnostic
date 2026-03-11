import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, roc_auc_score, RocCurveDisplay)
from sklearn.metrics import recall_score, precision_score

sns.set_theme(style="whitegrid")

def load_and_prep_data():
    """Loads dataset and performs initial split."""
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    return X_train, X_test, y_train, y_test, data

def build_pipelines():
    """Defines pipelines and hyperparameter grids for multiple models."""
    
    #SVM Pipeline with Scaling
    pipe_svm = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    
    param_grid_svm = {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__gamma': ['scale', 'auto']
    }

    #Random Forest Pipeline
    pipe_rf = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid_rf = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    return [
        ("SVM", pipe_svm, param_grid_svm),
        ("Random Forest", pipe_rf, param_grid_rf)
    ]

def evaluate_model(name, model, X_test, y_test, target_names):
    """Generates a professional evaluation report and plots."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*10} {name} Results {'='*10}")
    print(f"Best Params: {model.best_params_}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    #Confusion Matrix Visualization
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
 
def main():
    X_train, X_test, y_train, y_test, data = load_and_prep_data()
    models = build_pipelines()
    
    results = {}
    
    #Single ROC-AUC Plot for Comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, pipeline, params in models:
        print(f"Tuning {name}...")
        
        grid = GridSearchCV(pipeline, params, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        results[name] = grid
        evaluate_model(name, grid, X_test, y_test, data.target_names)
        
        RocCurveDisplay.from_estimator(grid, X_test, y_test, name=name, ax=ax)

    ax.set_title("ROC-AUC Curve Comparison")
    plt.show()

    #Feature Importance for Random Forest
    best_rf = results["Random Forest"].best_estimator_.named_steps['classifier']
    importances = pd.Series(best_rf.feature_importances_, index=data.feature_names)
    
    plt.figure(figsize=(10, 6))
    importances.sort_values().tail(10).plot(kind='barh', color='teal')
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.show()

if __name__ == "__main__":
    main()