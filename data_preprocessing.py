"""
Data preprocessing pipeline including cleaning, encoding, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import (TARGET_COLUMN, DROP_COLUMNS, MALIGNANT_LABEL, 
                    BENIGN_LABEL, TEST_SIZE, RANDOM_STATE, SMOTE_NEIGHBORS)


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline."""
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.numeric_features = None
        self.target_mapping = {MALIGNANT_LABEL: 1, BENIGN_LABEL: 0}
    
    def remove_unnecessary_columns(self):
        """Drop columns that don't contribute to prediction."""
        existing_drops = [col for col in DROP_COLUMNS if col in self.df.columns]
        self.df.drop(columns=existing_drops, inplace=True)
        print(f" Removed columns: {existing_drops}")
        return self
    
    def convert_datatypes(self):
        """Convert object columns to appropriate numeric types."""
        # Identify numeric feature columns
        self.numeric_features = [col for col in self.df.columns 
                                  if col != TARGET_COLUMN]
        
        for col in self.numeric_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f" Converted {len(self.numeric_features)} features to numeric")
        return self
    
    def encode_target(self):
        """Encode target variable to binary."""
        self.df[TARGET_COLUMN] = self.df[TARGET_COLUMN].map(self.target_mapping)
        print(f" Target encoded: {self.target_mapping}")
        return self
    
    def handle_outliers_and_missing(self):
        """Clean data by handling negatives, outliers, and missing values."""
        features = [col for col in self.df.columns if col != TARGET_COLUMN]
        
        # Replace impossible negative values
        for col in features:
            self.df.loc[self.df[col] < 0, col] = np.nan
        
        # Impute with median
        self.df[features] = self.df[features].fillna(self.df[features].median())
        
        # Cap extreme outliers at 1st and 99th percentile
        for col in features:
            lower_bound = self.df[col].quantile(0.01)
            upper_bound = self.df[col].quantile(0.99)
            self.df[col] = self.df[col].clip(lower_bound, upper_bound)
        
        print(f" Cleaned outliers and imputed missing values")
        return self
    
    def get_processed_data(self):
        """Return cleaned dataframe."""
        return self.df
    
    def preprocess(self):
        """Execute full preprocessing pipeline."""
        return (self
                .remove_unnecessary_columns()
                .convert_datatypes()
                .encode_target()
                .handle_outliers_and_missing()
                .get_processed_data())


class DataSplitter:
    """Handle train-test splitting and SMOTE oversampling."""
    
    def __init__(self, dataframe, target_col=TARGET_COLUMN):
        self.df = dataframe
        self.target_col = target_col
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(self, test_size=TEST_SIZE, stratify=True):
        """Split data into train and test sets."""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        stratify_param = y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify_param
        )
        
        print(f" Data split: Train={len(self.X_train)}, Test={len(self.X_test)}")
        return self
    
    def apply_smote(self, k_neighbors=SMOTE_NEIGHBORS):
        """Apply SMOTE to balance training data."""
        original_counts = self.y_train.value_counts().to_dict()
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
        new_counts = self.y_train.value_counts().to_dict()
        print(f"SMOTE applied: {original_counts} → {new_counts}")
        return self
    
    def get_splits(self):
        """Return all train-test splits."""
        return self.X_train, self.X_test, self.y_train, self.y_test


def prepare_dataset(dataframe):
    """Convenience function for full preprocessing."""
    preprocessor = DataPreprocessor(dataframe)
    clean_data = preprocessor.preprocess()
    
    splitter = DataSplitter(clean_data)
    splitter.split_data().apply_smote()
    
    return splitter.get_splits()