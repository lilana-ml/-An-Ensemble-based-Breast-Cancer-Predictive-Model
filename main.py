"""
Main execution pipeline for breast cancer ensemble learning project.
Orchestrates the entire workflow from data loading to model evaluation.
"""

import warnings
warnings.filterwarnings('ignore')

from config import DATA_PATH, TARGET_COLUMN
from data_loader import DataLoader
from data_preprocessing import DataPreprocessor, DataSplitter
from feature_engineering import FeatureSelector
from model_builder import EnsembleModelFactory
from model_evaluator import ModelEvaluator, ModelComparator
from visualization import DataVisualizer, ModelVisualizer
from utils import (ResultsLogger, ProgressTracker, print_section_header, 
                   print_metrics_table, validate_data_integrity, 
                   format_confusion_matrix_summary)


class BreastCancerPipeline:
    """End-to-end pipeline for breast cancer classification."""
    
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.progress = ProgressTracker(total_steps=7)
        self.logger = ResultsLogger()
        
        # Pipeline components
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def run(self):
        """Execute complete pipeline."""
        print_section_header("Breast Cancer Ensemble Learning Pipeline")
        
        # Step 1: Load Data
        self.progress.update("Loading data")
        self._load_data()
        
        # Step 2: Preprocess Data
        self.progress.update("Preprocessing and cleaning data")
        self._preprocess_data()
        
        # Step 3: Split and Balance Data
        self.progress.update("Splitting data and applying SMOTE")
        self._split_and_balance()
        
        # Step 4: Feature Selection
        self.progress.update("Performing feature selection with RFE")
        self._select_features()
        
        # Step 5: Build Models
        self.progress.update("Building and tuning ensemble models")
        self._build_models()
        
        # Step 6: Evaluate Models
        self.progress.update("Evaluating and comparing models")
        self._evaluate_models()
        
        # Step 7: Visualize Results
        self.progress.update("Generating visualizations")
        self._visualize_results()
        
        self.progress.complete()
        self._save_results()
        
        return self.results
    
    def _load_data(self):
        """Load dataset from file."""
        loader = DataLoader(self.data_path)
        self.raw_data = loader.load_data()
        
        if self.raw_data is not None:
            print(f"Dataset shape: {self.raw_data.shape}")
            validate_data_integrity(self.raw_data, TARGET_COLUMN)
    
    def _preprocess_data(self):
        """Clean and prepare data."""
        preprocessor = DataPreprocessor(self.raw_data)
        self.processed_data = preprocessor.preprocess()
        
        print(f"Processed data shape: {self.processed_data.shape}")
        print(f"Target distribution:\n{self.processed_data[TARGET_COLUMN].value_counts()}")
    
    def _split_and_balance(self):
        """Split data and apply SMOTE."""
        splitter = DataSplitter(self.processed_data)
        splitter.split_data(stratify=True).apply_smote()
        
        self.X_train, self.X_test, self.y_train, self.y_test = splitter.get_splits()
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print(f"Class balance after SMOTE: {self.y_train.value_counts().to_dict()}")
    
    def _select_features(self):
        """Perform feature selection using RFE."""
        selector = FeatureSelector(n_features_to_select=15)
        selector.fit_rfe(self.X_train, self.y_train)
        
        print(f"\nSelected features ({len(selector.selected_features)}):")
        print(selector.selected_features)
        
        # Evaluate baseline vs RFE
        selector.evaluate_baseline(self.X_train, self.X_test, 
                                   self.y_train, self.y_test)
        
        # Store feature rankings for later
        self.feature_rankings = selector.feature_rankings
    
    def _build_models(self):
        """Build all ensemble models."""
        factory = EnsembleModelFactory(self.X_train, self.y_train)
        self.models = factory.build_all_models()
        
        print(f"\nBuilt {len(self.models)} ensemble models")
    
    def _evaluate_models(self):
        """Evaluate and compare all models."""
        print_section_header("Model Evaluation")
        
        # Evaluate each model individually
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}")
            print('='*60)
            
            evaluator = ModelEvaluator(model, model_name)
            eval_results = evaluator.full_evaluation(
                self.X_train, self.X_test, 
                self.y_train, self.y_test
            )
            
            # Print metrics
            print_metrics_table(eval_results['metrics'])
            
            # Print confusion matrix summary
            cm_summary = format_confusion_matrix_summary(eval_results['confusion_matrix'])
            print("\nConfusion Matrix Summary:")
            print_metrics_table(cm_summary)
            
            # Print classification report
            print("\nClassification Report:")
            print(eval_results['classification_report'])
            
            # Store results
            self.results[model_name] = eval_results
        
        # Compare models
        print_section_header("Model Comparison")
        comparator = ModelComparator(self.models)
        comparison = comparator.compare_models(self.X_train, self.y_train, scoring='f1')
        
        best_model_name, best_score = comparator.get_best_model()
        
        print(f"\n{'='*60}")
        print(f" BEST MODEL: {best_model_name}")
        print(f"   Mean F1 Score: {best_score['mean']:.4f} (±{best_score['std']:.4f})")
        print('='*60)
        
        self.comparison_results = comparison
        self.best_model_name = best_model_name
    
    def _visualize_results(self):
        """Generate all visualizations."""
        print_section_header("Generating Visualizations")
        
        # EDA visualizations
        data_viz = DataVisualizer()
        
        data_viz.plot_target_distribution(self.processed_data[TARGET_COLUMN])
        
        data_viz.plot_correlation_heatmap(self.processed_data)
            
        data_viz.plot_feature_rankings(self.feature_rankings)
        
        # Model evaluation visualizations
        model_viz = ModelVisualizer()
        
        model_viz.plot_model_comparison_boxplot(self.comparison_results)
        
        # Train models on full training data for ROC/DET curves
        trained_models = {}
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            trained_models[name] = model
        
        model_viz.plot_roc_and_det_curves(trained_models, self.X_test, self.y_test)
        
        # Confusion matrices for each model
        for model_name, results in self.results.items():
            print(f"Creating confusion matrix for {model_name}...")
            model_viz.plot_confusion_matrix(
                results['confusion_matrix'], 
                model_name
            )
    
    def _save_results(self):
        """Save all results to files."""
        print_section_header("Saving Results")
        
        # Save metrics for each model
        for model_name, results in self.results.items():
            filename = f"{model_name.replace(' ', '_')}_metrics.json"
            self.logger.save_metrics(results['metrics'], filename)
        
        # Save best model
        best_model = self.models[self.best_model_name]
        best_model.fit(self.X_train, self.y_train)  # Ensure it's trained
        self.logger.save_model(best_model, self.best_model_name)
        
        # Save feature rankings
        self.logger.save_dataframe(
            self.feature_rankings, 
            'feature_rankings.csv'
        )
        
        # Generate summary report
        self.logger.generate_summary_report(
            self.comparison_results, 
            self.best_model_name
        )
        

def main():
    """Main entry point."""
    pipeline = BreastCancerPipeline(data_path=DATA_PATH)
    results = pipeline.run()
    
    print("\n" + "="*70)
    print("Pipeline execution completed")
    print("Check the 'results/' directory for saved outputs.")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()