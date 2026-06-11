"""
Classifier training and evaluation pipeline.

Implements the four classifiers defined in the TCC methodology:
    1. Decision Tree
    2. Support Vector Machine (SVM)
    3. XGBoost
    4. Multilayer Perceptron (Neural Network)

All are evaluated with stratified k-fold cross-validation. In the VeHIF
detection context, recall (sensitivity to faults) is particularly important
because missed faults carry higher risk than false alarms.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def get_classifiers(random_state: int = 42) -> dict:
    """Return the four classifiers with default hyperparameters.

    Each entry is a sklearn-compatible estimator (or Pipeline where
    scaling is required by the algorithm).

    Returns
    -------
    dict mapping classifier name to estimator.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                early_stopping=True,
                random_state=random_state,
            )),
        ]),
    }


def evaluate_classifier(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """Evaluate a classifier with stratified k-fold cross-validation.

    Parameters
    ----------
    clf          : sklearn-compatible classifier (fit/predict interface)
    X            : feature matrix
    y            : label vector
    cv           : number of folds (default 5)
    random_state : seed for reproducibility

    Returns
    -------
    dict with mean and std of: accuracy, precision, recall, f1
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    results = cross_validate(clf, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    return {
        "accuracy_mean":   float(np.mean(results["test_accuracy"])),
        "accuracy_std":    float(np.std(results["test_accuracy"])),
        "precision_mean":  float(np.mean(results["test_precision_weighted"])),
        "precision_std":   float(np.std(results["test_precision_weighted"])),
        "recall_mean":     float(np.mean(results["test_recall_weighted"])),
        "recall_std":      float(np.std(results["test_recall_weighted"])),
        "f1_mean":         float(np.mean(results["test_f1_weighted"])),
        "f1_std":          float(np.std(results["test_f1_weighted"])),
    }


def run_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run all four classifiers and return a comparative results table.

    Parameters
    ----------
    X  : feature matrix
    y  : label vector (string labels)
    cv : number of cross-validation folds

    Returns
    -------
    pd.DataFrame with one row per classifier and columns:
        accuracy_mean, accuracy_std, precision_mean, precision_std,
        recall_mean, recall_std, f1_mean, f1_std
    """
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), index=y.index)

    classifiers = get_classifiers(random_state=random_state)
    rows = []
    for name, clf in classifiers.items():
        metrics = evaluate_classifier(clf, X, y_enc, cv=cv, random_state=random_state)
        metrics["classifier"] = name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("classifier")
    return df[["accuracy_mean", "accuracy_std",
               "precision_mean", "precision_std",
               "recall_mean", "recall_std",
               "f1_mean", "f1_std"]]
