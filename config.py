"""
Configuration file for breast cancer ensemble learning project.
Contains all constants, paths, and hyperparameters.
"""

import numpy as np

# Random seed for reproducibility
RANDOM_STATE = 42

# Data parameters
TEST_SIZE = 0.25
CV_FOLDS = 10

# Feature engineering
TARGET_COLUMN = 'diagnosis'
DROP_COLUMNS = ['id', 'gender']
MALIGNANT_LABEL = 'M'
BENIGN_LABEL = 'B'

# SMOTE parameters
SMOTE_NEIGHBORS = 5

# RFE parameters
RFE_STEP = 1
RFE_N_FEATURES = 15

# Model hyperparameters
KNN_PARAMS = {
    'n_neighbors': list(range(1, 31)),
    'cv': 10,
    'scoring': 'accuracy'
}

DT_PARAMS = {
    'max_depth': list(range(1, 31)),
    'cv': 10,
    'scoring': 'accuracy'
}

XGBOOST_PARAMS = {
    'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 
                      0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.2, 0.5, 0.7, 1],
    'n_estimators': [100],
    'cv': 10
}

# Voting classifier weights
VOTING_WEIGHTS = [2, 1, 1]  # [LogisticRegression, DecisionTree, KNN]

# Visualization parameters
FIGURE_SIZE = (12, 8)
HEATMAP_SIZE = (21, 21)
COLOR_PALETTE = 'BrBG'

# File paths
DATA_PATH = 'breastCancer.csv'
RESULTS_PATH = 'results/'