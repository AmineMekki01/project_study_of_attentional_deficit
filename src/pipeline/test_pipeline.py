import pandas as pd
from joblib import load

from src.components.evaluate import evaluate_on_new_data


def testing_pipeline(test_data_path: str, chosen_model: str, chosen_feature_selector: str) -> pd.DataFrame:
    # Load the chosen model and feature selector
    loaded_model = load(
        f'./artifacts/models/{chosen_feature_selector}_{chosen_model}.joblib')
    loaded_feature_selector = load(
        f'./artifacts/feature_selectors/{chosen_feature_selector}.joblib')

    metrics_df = evaluate_on_new_data(
        loaded_model, test_data_path, loaded_feature_selector)

    metrics_df['Model'] = chosen_model
    metrics_df['Method'] = chosen_feature_selector
    return metrics_df
