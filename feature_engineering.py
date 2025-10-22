"""
Feature selection using Recursive Feature Elimination (RFE).
"""

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from config import RANDOM_STATE, RFE_STEP


class FeatureSelector:
    """RFE-based feature selection wrapper."""
    
    def __init__(self, n_features_to_select=None, step=RFE_STEP):
        self.n_features = n_features_to_select
        self.step = step
        self.base_estimator = RandomForestClassifier(random_state=RANDOM_STATE)
        self.rfe = None
        self.selected_features = None
        self.feature_rankings = None
    
    def fit_rfe(self, X_train, y_train):
        """Fit RFE on training data."""
        self.rfe = RFE(
            estimator=self.base_estimator,
            n_features_to_select=self.n_features,
            step=self.step
        )
        self.rfe.fit(X_train, y_train)
        
        # Store selected feature names and rankings
        self.selected_features = X_train.columns[self.rfe.support_].tolist()
        self.feature_rankings = pd.DataFrame({
            'Feature': X_train.columns,
            'Ranking': self.rfe.ranking_
        }).sort_values('Ranking')
        
        return self
    
    def transform_data(self, X):
        """Transform dataset using selected features."""
        if self.rfe is None:
            raise ValueError("RFE not fitted yet. Call fit_rfe() first.")
        return self.rfe.transform(X)
    
    def evaluate_baseline(self, X_train, X_test, y_train, y_test):
        """Compare model performance with and without feature selection."""
        # Without RFE
        baseline = self.base_estimator.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)
        baseline_acc = accuracy_score(y_test, y_pred_baseline)
        baseline_f1 = f1_score(y_test, y_pred_baseline)
        
        # With RFE
        X_train_rfe = self.transform_data(X_train)
        X_test_rfe = self.transform_data(X_test)
        rfe_model = self.base_estimator.fit(X_train_rfe, y_train)
        y_pred_rfe = rfe_model.predict(X_test_rfe)
        rfe_acc = accuracy_score(y_test, y_pred_rfe)
        rfe_f1 = f1_score(y_test, y_pred_rfe)
        
        results = {
            'baseline': {'accuracy': baseline_acc, 'f1_score': baseline_f1},
            'rfe': {'accuracy': rfe_acc, 'f1_score': rfe_f1}
        }
        
        print(f"Baseline: Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
        print(f"With RFE: Acc={rfe_acc:.4f}, F1={rfe_f1:.4f}")
        
        return results
    
    def get_rfe_transformer(self):
        """Return fitted RFE object for pipeline integration."""
        return self.rfe


def select_best_features(X_train, X_test, y_train, y_test, n_features=None):
    """Convenience function for feature selection."""
    selector = FeatureSelector(n_features_to_select=n_features)
    selector.fit_rfe(X_train, y_train)
    selector.evaluate_baseline(X_train, X_test, y_train, y_test)
    
    X_train_selected = selector.transform_data(X_train)
    X_test_selected = selector.transform_data(X_test)
    
    return X_train_selected, X_test_selected, selector