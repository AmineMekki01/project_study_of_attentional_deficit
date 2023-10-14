
import mne
import pandas as pd
from typing import Tuple, Dict

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


def evaluate_best_model_on_test_data(best_model, best_feature_selector, TEST_data_path: str, include_auc: str) -> pd.DataFrame:
    """Evaluate the best model on new test data."""
    return evaluate_on_new_data(best_model, TEST_data_path, feature_selector=best_feature_selector, include_auc=include_auc)


def training_pipeline(training_data_path: str, feature_selection_methods: Dict, models: Dict, evaluation_method: str, include_auc: str, TEST_data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main function to run the training pipeline."""

    # Step 1: Read data
    epochs = read_epochs_from_file(training_data_path)
    features, target, featured_df = get_features(epochs)
    print(featured_df)

    # Initialize data structures to store metrics and models
    feature_selection_metrics = {"RFC": {}, "LDA": {}, "RFE": {}}
    hashmap_fitted_feature_selector = {"RFC": None, "LDA": None, "RFE": None}
    models_registry_per_FSM = {}

    # Step 2: Feature selection and model training
    for method in feature_selection_methods:
        print(f"Feature Selection Method: {method}")
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
                          'Precision', 'Recall', 'F1', 'ROC AUC']

    print("The final metrics for each feature selection method and each model. \n")
    print(metrics_df)

    metrics_df.to_csv("./artifacts/scores/training/metrics.csv", index=False)

    # Step 4: Evaluate the best model on test data
    best_feature_selection_method = "RFE"
    best_model = "Random Forest"
    best_trained_model = models_registry_per_FSM[best_feature_selection_method][best_model]
    best_fitted_feature_selector = hashmap_fitted_feature_selector[best_feature_selection_method]
    test_metrics = evaluate_best_model_on_test_data(
        best_trained_model, best_fitted_feature_selector, TEST_data_path, include_auc)

    # Step 5: Save the test metrics

    test_metrics_df = pd.DataFrame.from_dict({
        "Feature Selection Method": [best_feature_selection_method],
        "Model": [best_model],
        "Accuracy": test_metrics[0],
        "Precision": test_metrics[1],
        "Recall": test_metrics[2],
        "F1": test_metrics[3],
        "ROC AUC": test_metrics[4]
    })

    test_metrics_df.to_csv(
        "./artifacts/scores/testing/metrics.csv", index=False)

    return metrics_df, test_metrics_df
