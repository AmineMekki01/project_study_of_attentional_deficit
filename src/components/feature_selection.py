"""

Description : In this file, we will implement the feature selection algorithms.

"""
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import LinearSVC
from typing import Tuple, Optional
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.feature_selection import RFE


def perform_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    model: Optional[BaseEstimator] = None
) -> Tuple[np.ndarray, BaseEstimator]:
    """Utility function to perform feature selection."""
    if model is None:
        raise ValueError("Model for feature selection is not provided.")

    try:
        feature_selector = SelectFromModel(model)
        feature_selector.fit(X, y)
        X_new = feature_selector.transform(X)
        return X_new, feature_selector
    except Exception as e:
        print(f"Error during feature selection: {e}")
        return X, None


def feature_selection_RFC(
    X: np.ndarray,
    y: np.ndarray,
    model: Optional[RFC] = None
) -> Tuple[np.ndarray, BaseEstimator]:
    """
    Performs feature selection using RandomForestClassifier.

    Parameters:
    - X: Features
    - y: Labels
    - model: Optional RandomForestClassifier model. If None, a default model is used.

    Returns:
    - Transformed features and the feature selection model.
    """
    if model is None:
        model = RFC(n_estimators=100, random_state=0)

    return perform_feature_selection(X, y, model)


def feature_selection_RFC(
    X: np.ndarray,
    y: np.ndarray,
    model: Optional[RFC] = None
) -> Tuple[np.ndarray, BaseEstimator]:
    """
    Performs feature selection using RandomForestClassifier.

    Parameters:
    - X: Features
    - y: Labels
    - model: Optional RandomForestClassifier model. If None, a default model is used.

    Returns:
    - Transformed features and the feature selection model.
    """
    if model is None:
        model = RFC(n_estimators=100, random_state=0)

    return perform_feature_selection(X, y, model)


def feature_selection_LDA(
    X: np.ndarray,
    y: np.ndarray,
    model: Optional[LDA] = None
) -> Tuple[np.ndarray, BaseEstimator]:
    """
    Performs feature selection using Linear Discriminant Analysis (LDA).

    Parameters:
    - X: np.ndarray
        Features
    - y: np.ndarray
        Labels
    - model: Optional LDA model
        If None, a default model is used.

    Returns:
    - Tuple[np.ndarray, BaseEstimator]
        Transformed features and the feature selection model.
    """

    if model is None:
        model = LDA()

    return perform_feature_selection(X, y, model)


def feature_selection_RFE(
    X: np.ndarray,
    y: np.ndarray,
    model: Optional[LinearSVC] = None
) -> Tuple[np.ndarray, BaseEstimator]:
    """
    Performs feature selection using Recursive Feature Elimination (RFE).

    Parameters:
    - X: np.ndarray
        Features
    - y: np.ndarray
        Labels
    - model: Optional LinearSVC model
        If None, a default model is used.

    Returns:
    - Tuple[np.ndarray, BaseEstimator]
        Transformed features and the feature selection model.
    """

    if model is None:
        model = LinearSVC()

    try:
        feature_selector = RFE(model)
        feature_selector.fit(X, y)
        X_new = feature_selector.transform(X)
        return X_new, feature_selector
    except Exception as e:
        print(f"Error during feature selection: {e}")
        return X, None
