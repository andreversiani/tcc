"""
Feature extraction aggregation and selection utilities.

Combines all domain-specific feature modules into a single pipeline and
provides feature selection methods to identify the most discriminative
attributes for the VeHIF classification problem.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from src import data_loader
from src.features import time_domain, frequency, time_frequency


def extract_all_features(
    record: dict,
    channel: str = "lf",
    fundamental: float = 60.0,
) -> pd.Series:
    """Extract the full feature vector from a single VeHIF record.

    Uses one channel (default: 'lf') of the full record signal without
    windowing. Call preprocessing.segment_by_cycles() first if per-window
    features are needed.

    Parameters
    ----------
    record   : output of data_loader.load_record()
    channel  : 'lf' or 'hf'
    fundamental: fundamental frequency (Hz)

    Returns
    -------
    pd.Series with all time-domain, frequency-domain, and time-frequency
    features, plus the record label.
    """
    sig = data_loader.get_channel(record, channel)
    fs = record[f"fs_{channel}"]

    features = {}
    features.update(time_domain.extract_all(sig, fs, fundamental))
    features.update(frequency.extract_all(sig, fs, fundamental))
    features.update(time_frequency.extract_all(sig, fs))
    features["label"] = record["label"]
    return pd.Series(features)


def build_feature_matrix(
    dataset: pd.DataFrame,
    channel: str = "lf",
    fundamental: float = 60.0,
) -> tuple:
    """Build feature matrix X and label vector y from a dataset DataFrame.

    Parameters
    ----------
    dataset   : output of data_loader.load_dataset()
    channel   : 'lf' or 'hf'
    fundamental: fundamental frequency (Hz)

    Returns
    -------
    (X, y) where X is pd.DataFrame of features and y is pd.Series of labels.
    """
    rows = []
    for _, row in dataset.iterrows():
        record = data_loader.load_record(row["path"])
        rows.append(extract_all_features(record, channel, fundamental))
    df = pd.DataFrame(rows)
    y = df.pop("label")
    return df, y


def select_top_k_anova(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 20,
) -> list:
    """Select top-k features by ANOVA F-score.

    Uses univariate F-test (scipy.stats.f_oneway via sklearn) to rank
    features by their ability to discriminate between classes.

    Returns
    -------
    List of column names of the top-k selected features, ranked by score.
    """
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    scores = pd.Series(selector.scores_, index=X.columns)
    return list(scores.nlargest(k).index)


def select_top_k_rf(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 20,
    n_estimators: int = 200,
    random_state: int = 42,
) -> list:
    """Select top-k features by Random Forest importance.

    Trains a Random Forest and uses mean decrease in impurity (Gini) as the
    feature importance measure. Robust to feature scale and interactions.

    Returns
    -------
    List of column names of the top-k features by importance, ranked.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X, y)
    importance = pd.Series(rf.feature_importances_, index=X.columns)
    return list(importance.nlargest(k).index)
