import os
from typing import Dict, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from src.components.feature_engineering import get_features
from src.components.feature_selection import feature_selection_RFC, feature_selection_LDA, feature_selection_RFE
from src.components.model_trainer import train
from src.components.evaluate import evaluate_on_new_data
# Make sure this import path is correct
from src.pipeline import training_pipeline

# Set environment variable to hide pygame support prompt
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Constants
DATA_PATH_S11 = os.getenv(
    "DATA_PATH_S11", "./artifacts/data/raw/AEP/Subject_11_H_AEP_Run_01.set")
TEST_DATA_PATH = os.getenv(
    "TEST_DATA_PATH", "./artifacts/data/raw/AEP/Subject_15_H_AEP_Run_01.set")
EVALUATION_METHOD = os.getenv("EVALUATION_METHOD", 'stratifiedKFold')
INCLUDE_AUC = EVALUATION_METHOD != 'loo'

# Feature selection methods
feature_selection_methods: Dict[str, Callable] = {
    "RFC": feature_selection_RFC,
    "LDA": feature_selection_LDA,
    "RFE": feature_selection_RFE
}

# Models for training
models: Dict[str, Callable] = {
    "Random Forest": RandomForestClassifier(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Support Vector Machine": SVC(probability=True)
}

if __name__ == "__main__":
    try:
        # Run the training pipeline
        training_pipeline(
            DATA_PATH_S11,
            feature_selection_methods,
            models,
            EVALUATION_METHOD,
            INCLUDE_AUC,
            TEST_DATA_PATH
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
