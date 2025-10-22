"""
Build and tune ensemble learning models.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from config import (RANDOM_STATE, KNN_PARAMS, DT_PARAMS, 
                    XGBOOST_PARAMS, VOTING_WEIGHTS)


class HyperparameterTuner:
    """Grid search for hyperparameter optimization."""
    
    @staticmethod
    def tune_knn(X_train, y_train):
        """Find optimal K for KNN."""
        knn = KNeighborsClassifier()
        param_grid = {'n_neighbors': KNN_PARAMS['n_neighbors']}
        
        grid = GridSearchCV(
            knn, param_grid,
            cv=KNN_PARAMS['cv'],
            scoring=KNN_PARAMS['scoring'],
            return_train_score=False,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        print(f" KNN Best params: {grid.best_params_}, Score: {grid.best_score_:.4f}")
        return grid.best_estimator_
    
    @staticmethod
    def tune_decision_tree(X_train, y_train):
        """Find optimal max_depth for Decision Tree."""
        dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
        param_grid = {'max_depth': DT_PARAMS['max_depth']}
        
        grid = GridSearchCV(
            dt, param_grid,
            cv=DT_PARAMS['cv'],
            scoring=DT_PARAMS['scoring'],
            return_train_score=False,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        print(f" DT Best params: {grid.best_params_}, Score: {grid.best_score_:.4f}")
        return grid.best_estimator_
    
    @staticmethod
    def tune_xgboost(X_train, y_train):
        """Find optimal learning rate for XGBoost."""
        xgb_clf = xgb.XGBClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'learning_rate': XGBOOST_PARAMS['learning_rate'],
            'n_estimators': XGBOOST_PARAMS['n_estimators']
        }
        
        grid = GridSearchCV(
            xgb_clf, param_grid,
            cv=XGBOOST_PARAMS['cv'],
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        print(f" XGB Best params: {grid.best_params_}, Score: {grid.best_score_:.4f}")
        return grid.best_estimator_


class EnsembleModelFactory:
    """Factory for creating ensemble model pipelines."""
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.tuner = HyperparameterTuner()
    
    def build_voting_classifier(self):
        """Build soft voting ensemble with tuned base models."""
        # Tune base classifiers
        best_knn = self.tuner.tune_knn(self.X_train, self.y_train)
        best_dt = self.tuner.tune_decision_tree(self.X_train, self.y_train)
        lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('dt', best_dt),
                ('knn', best_knn)
            ],
            voting='soft',
            weights=VOTING_WEIGHTS
        )
        
        # Build pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', RFE(estimator=RandomForestClassifier(random_state=RANDOM_STATE))),
            ('classification', voting_clf)
        ])
        
        print(" Voting Classifier pipeline created")
        return pipeline
    
    def build_xgboost_model(self):
        """Build XGBoost pipeline with tuned hyperparameters."""
        best_xgb = self.tuner.tune_xgboost(self.X_train, self.y_train)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', RFE(estimator=RandomForestClassifier(random_state=RANDOM_STATE))),
            ('classification', best_xgb)
        ])
        
        print(" XGBoost pipeline created")
        return pipeline
    
    def build_all_models(self):
        """Build all ensemble models."""
        models = {
            'Voting Classifier': self.build_voting_classifier(),
            'XGBoost': self.build_xgboost_model()
        }
        return models


def create_ensemble_models(X_train, y_train):
    """Convenience function to create all models."""
    factory = EnsembleModelFactory(X_train, y_train)
    return factory.build_all_models()