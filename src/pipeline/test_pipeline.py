from src.components.evaluate import evaluate_on_new_data
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator
from typing import Tuple


def load_model(model_path: str) -> BaseEstimator:
    return load(model_path)


def load_feature_selector(feature_selector_path: str):
    return load(feature_selector_path)


def testing_pipeline(
    testing_data_path: str,
    model_path: str,
    feature_selector_path: str
) -> Tuple[pd.DataFrame]:

    loaded_model = load_model(model_path)

    loaded_feature_selector = load_feature_selector(feature_selector_path)

    metrics_df = evaluate_on_new_data(
        loaded_model, testing_data_path, loaded_feature_selector, include_auc=True)

    return metrics_df


# (test_data_path: str, chosen_model: str, chosen_feature_selector: str) -> pd.DataFrame:
#     # Load the chosen model and feature selector
#     loaded_model = load(
#         f'./artifacts/models/#{chosen_feature_selector}_{chosen_model}.joblib')
#     loaded_feature_selector = load(
#         f'./artifacts/feature_selectors/{chosen_feature_selector}.joblib')

#     metrics_df = evaluate_on_new_data(
#         loaded_model, test_data_path, loaded_feature_selector)

#     metrics_df['Model'] = chosen_model
#     metrics_df['Method'] = chosen_feature_selector
#     return metrics_df
