"""
Comprehensive model evaluation and comparison.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score, KFold
from config import CV_FOLDS, RANDOM_STATE


class ModelEvaluator:
    """Evaluate model performance with multiple metrics."""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.predictions = None
        self.metrics = {}
    
    def train_model(self, X_train, y_train):
        """Train the model."""
        self.model.fit(X_train, y_train)
        print(f"{self.model_name} trained")
        return self
    
    def predict(self, X_test):
        """Generate predictions."""
        self.predictions = self.model.predict(X_test)
        return self.predictions
    
    def calculate_metrics(self, y_true, y_pred=None):
        """Calculate all performance metrics."""
        if y_pred is None:
            y_pred = self.predictions
        
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        return self.metrics
    
    def get_confusion_matrix(self, y_true, y_pred=None):
        """Generate confusion matrix."""
        if y_pred is None:
            y_pred = self.predictions
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true, y_pred=None):
        """Generate detailed classification report."""
        if y_pred is None:
            y_pred = self.predictions
        return classification_report(y_true, y_pred, 
                                    target_names=['Benign', 'Malignant'])
    
    def cross_validate(self, X, y, scoring='f1'):
        """Perform k-fold cross-validation."""
        kfold = KFold(n_splits=CV_FOLDS, random_state=RANDOM_STATE, shuffle=True)
        cv_scores = cross_val_score(self.model, X, y, cv=kfold, 
                                     scoring=scoring, n_jobs=-1)
        
        cv_results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        print(f"{self.model_name} CV {scoring}: {cv_results['mean']:.4f} (±{cv_results['std']:.4f})")
        return cv_results
    
    def full_evaluation(self, X_train, X_test, y_train, y_test):
        """Perform complete evaluation pipeline."""
        self.train_model(X_train, y_train)
        predictions = self.predict(X_test)
        metrics = self.calculate_metrics(y_test, predictions)
        cm = self.get_confusion_matrix(y_test, predictions)
        report = self.get_classification_report(y_test, predictions)
        cv_results = self.cross_validate(X_train, y_train)
        
        return {
            'predictions': predictions,
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'cross_validation': cv_results
        }


class ModelComparator:
    """Compare multiple models side-by-side."""
    
    def __init__(self, models_dict):
        self.models = models_dict
        self.results = {}
    
    def compare_models(self, X_train, y_train, scoring='f1'):
        """Compare all models using cross-validation."""
        comparison = {}
        
        for name, model in self.models.items():
            evaluator = ModelEvaluator(model, name)
            cv_results = evaluator.cross_validate(X_train, y_train, scoring)
            comparison[name] = cv_results
        
        self.results = comparison
        return comparison
    
    def get_best_model(self):
        """Identify the best performing model."""
        if not self.results:
            raise ValueError("No comparison results. Run compare_models() first.")
        
        best_model_name = max(self.results, key=lambda k: self.results[k]['mean'])
        return best_model_name, self.results[best_model_name]


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Convenience function for single model evaluation."""
    evaluator = ModelEvaluator(model, model_name)
    return evaluator.full_evaluation(X_train, X_test, y_train, y_test)


def compare_all_models(models_dict, X_train, y_train):
    """Convenience function for model comparison."""
    comparator = ModelComparator(models_dict)
    results = comparator.compare_models(X_train, y_train)
    best_model, best_score = comparator.get_best_model()
    return results, best_model