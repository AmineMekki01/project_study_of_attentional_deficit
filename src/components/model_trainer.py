"""

This file will contain the model trainer function. This function will take in the data and the model and train the model on the data. It will return the trained model.

"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
import numpy as np
from typing import Dict, List, Union, Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from src.utils.plotting import plot_confusion_matrix


def evaluate_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    include_auc: bool = True
) -> Tuple[float, float, float, float, Union[float, str], np.ndarray]:
    """
    This function takes the model, the training data, the testing data and the include_auc flag as input. It returns the accuracy, precision, recall, f1 and roc_auc scores.

    Parameters
    ----------
    model : object
        The model.

    X_train : numpy.ndarray 
        The training data.

    X_test : numpy.ndarray  
        The testing data.

    y_train : numpy.ndarray 
        The training labels.

    y_test : numpy.ndarray  
        The testing labels.

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
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    if cm.shape == (1, 1):
        if (y_test[0] == 2) and (y_pred[0] == 2):
            cm = np.array([[0, 0], [0, cm[0][0]]])
        elif (y_test[0] == 1) and (y_pred[0] == 1):
            cm = np.array([[cm[0][0], 0], [0, 0]])
        elif (y_test[0] == 1) and (y_pred[0] == 2):
            cm = np.array([[0, cm[0][0]], [0, 0]])
        elif (y_test[0] == 2) and (y_pred[0] == 1):
            cm = np.array([[0, 0], [cm[0][0], 0]])
        else:
            print("Something wrong with the confusion matrix")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted')

    if include_auc and hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)[:, 1]
        try:
            roc_auc = roc_auc_score(y_test, proba)
        except ValueError as e:
            print(str(e))
            roc_auc = "Not available"
    else:
        roc_auc = "Not available"

    return accuracy, precision, recall, f1, roc_auc, cm, model


def compute_average_metrics(metrics: Dict[str, List[Union[float, str, np.ndarray]]]) -> Dict[str, Union[float, np.ndarray]]:
    avg_metrics = {}
    for key, values in metrics.items():
        if key != 'confusion_matrix':
            if len([x for x in values if x != "Not available"]) > 0:
                avg_metrics[key] = np.mean(
                    [x for x in values if x != "Not available"])
            else:
                avg_metrics[key] = "Not available"
        else:
            avg_metrics[key] = np.sum(
                [x for x in values if isinstance(x, np.ndarray)], axis=0)

    return avg_metrics


def train(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, BaseEstimator],
    cv_type: str = 'stratifiedKFold',
    n_splits: int = 5
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, BaseEstimator]]:
    """
    This function takes the data, the models, the cv_type and the n_splits as input. It returns the model metrics and the models registry.

    Parameters
    ----------
    X : numpy.ndarray
        The data.

    y : numpy.ndarray   
        The labels.

    models : dict   
        The dictionary of the models.   

    cv_type : str   
        The cv_type.

    n_splits : int  
        The n_splits.   

    Returns 
    ------- 
    model_metrics : dict
        The dictionary of the model metrics.    

    models_registry : dict  
        The dictionary of the models registry.  
    """
    model_metrics = {}
    models_registry = {}

    include_auc = (cv_type != 'loo')

    for model_name, model_instance in models.items():
        model_metrics[model_name] = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': [],
            'confusion_matrix': []
        }

        if (cv_type == 'loo' or model_name in ['Random Forest', 'Support Vector Machine']) and (model_name != 'Linear Discriminant Analysis'):
            model_instance.set_params(class_weight='balanced')

        cv = StratifiedKFold(
            n_splits=n_splits) if cv_type == 'stratifiedKFold' else LeaveOneOut()

        for train_index, test_index in cv.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            metrics = evaluate_model(
                model_instance,
                X_train,
                X_test,
                y_train,
                y_test,
                include_auc=include_auc
            )

            for key, value in zip(list(model_metrics[model_name].keys()), metrics[:-2]):
                if isinstance(value, (float, int)):
                    model_metrics[model_name][key].append(
                        round(value * 100, 2))
                else:
                    model_metrics[model_name][key].append(value)

            model_metrics[model_name]['confusion_matrix'].append(metrics[-2])

        models_registry[model_name] = metrics[-1]
        model_metrics[model_name] = compute_average_metrics(
            model_metrics[model_name])

    return model_metrics, models_registry
