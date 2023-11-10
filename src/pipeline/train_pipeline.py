
import mne
import pandas as pd
from typing import Tuple, Dict, Union
from joblib import dump

from src.components.feature_engineering import get_features
from src.components.model_trainer import train
from src.components.evaluate import evaluate_on_new_data


def read_epochs_from_file(training_data_path: str) -> mne.Epochs:
    """Read EEG epochs data from a file."""
    return mne.io.read_epochs_eeglab(training_data_path)


def perform_feature_selection(X: pd.DataFrame, y: pd.Series, feature_selector) -> Tuple[pd.DataFrame, any]:
    """Perform feature selection on the data."""
    X_new, fitted_feature_selector = feature_selector(X, y)
    return X_new, fitted_feature_selector


def train_models(X: pd.DataFrame, y: pd.Series, models: Dict, evaluation_method: str) -> Tuple[Dict, Dict]:
    """Train models on the data."""
    return train(X, y, models, cv_type=evaluation_method)


def training_pipeline(training_data_path: str, feature_selection_methods: Dict, models: Dict, evaluation_method: str) -> pd.DataFrame:
    """Main function to run the training pipeline."""

    # Step 1: Read data
    epochs = read_epochs_from_file(training_data_path)
    features, target, featured_df = get_features(epochs)

    # Initialize data structures to store metrics and models
    feature_selection_metrics = {"RFC": {}, "LDA": {}, "RFE": {}}
    hashmap_fitted_feature_selector = {"RFC": None, "LDA": None, "RFE": None}
    models_registry_per_FSM = {}

    # Step 2: Feature selection and model training
    for method in feature_selection_methods:

        print(f"Feature Selection Method: {method}")
        print('xxxxxasdxxx', feature_selection_methods[method])
        X_new, fitted_feature_selector = perform_feature_selection(
            features, target, feature_selection_methods[method])

        hashmap_fitted_feature_selector[method] = fitted_feature_selector

        model_metrics, model_registry = train_models(
            X_new, target, models, evaluation_method)

        feature_selection_metrics[method] = model_metrics
        models_registry_per_FSM[method] = model_registry

    # Step 3: Create metrics Dataframe
    metrics_df = pd.DataFrame.from_dict({(i, j): feature_selection_metrics[i][j]
                                         for i in feature_selection_metrics.keys()
                                         for j in feature_selection_metrics[i].keys()},
                                        orient='index')

    metrics_df.reset_index(inplace=True)
    metrics_df.columns = ['Method', 'Model', 'Accuracy',
                          'Precision', 'Recall', 'F1', 'ROC AUC', 'Confusion Matrix']

    print("The final metrics for each feature selection method and each model. \n")
    print(metrics_df)

    metrics_df.to_csv("./artifacts/scores/training/metrics.csv", index=False)

    # Step 4: Save models and feature selectors

    best_feature_selection_method = list(feature_selection_methods.keys())[0]
    best_model = list(
        models_registry_per_FSM[best_feature_selection_method].keys())[0]
    best_trained_model = models_registry_per_FSM[best_feature_selection_method][best_model]
    best_fitted_feature_selector = hashmap_fitted_feature_selector[best_feature_selection_method]

    best_trained_model_path = f'./artifacts/models/{best_feature_selection_method}_{best_model}.joblib'
    best_fitted_feature_selector_path = f'./artifacts/feature_selectors/{best_feature_selection_method}.joblib'

    dump(best_trained_model, best_trained_model_path)
    dump(fitted_feature_selector, best_fitted_feature_selector_path)

    return metrics_df, best_trained_model_path, best_fitted_feature_selector_path
