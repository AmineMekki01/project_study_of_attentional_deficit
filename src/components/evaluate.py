from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mne
from src.components.feature_engineering import get_features
from typing import Union, Tuple
import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluate_on_new_data(model, path_to_new_epochs, feature_selector, include_auc=True) -> Tuple[float, float, float, float, Union[float, str]]:
    """
    This function takes the model, the path to the new epochs, the feature selector and the include_auc flag as input. It returns the accuracy, precision, recall, f1 and roc_auc scores.

    Parameters
    ----------
    model : object
        The model.  

    path_to_new_epochs : str    
        The path to the new epochs.

    feature_selector : object   
        The feature selector.

    include_auc : str   
        The include_auc flag.

    Returns 
    -------
    accuracy : float
        The accuracy score.

    precision : float   
        The precision score.

    recall : float  
        The recall score.

    f1 : float
        The f1 score.

    roc_auc : float 
        The roc_auc score.

    """
    try:
        new_epochs = mne.io.read_epochs_eeglab(path_to_new_epochs)
    except Exception as e:
        print(f"Error reading epochs: {str(e)}")
        return 0, 0, 0, 0, "Not available"

    try:
        new_features, new_target, _ = get_features(new_epochs)
        new_features_transformed = feature_selector.transform(new_features)
        new_predictions = model.predict(new_features_transformed)
    except Exception as e:
        print(f"Error during feature extraction or prediction: {str(e)}")
        return 0, 0, 0, 0, "Not available"

    cm = confusion_matrix(new_target, new_predictions)
    accuracy = accuracy_score(new_target, new_predictions)
    precision = precision_score(
        new_target, new_predictions, average='weighted')
    recall = recall_score(new_target, new_predictions, average='weighted')
    f1 = f1_score(new_target, new_predictions, average='weighted')

    roc_auc = "Not available"
    if include_auc and hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(new_features_transformed)[:, 1]
            roc_auc = roc_auc_score(new_target, proba)
        except ValueError as e:
            print(str(e))

    test_metrics_df = pd.DataFrame(
        {'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1': [f1], 'ROC AUC': [roc_auc], 'confusion_matrix': [cm]})

    return test_metrics_df
