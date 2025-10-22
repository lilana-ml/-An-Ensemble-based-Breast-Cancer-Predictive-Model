"""
Utility functions for logging, saving results, and helper operations.
"""

import json
import pickle
import pandas as pd
from datetime import datetime
import os


class ResultsLogger:
    """Log and save model results."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    
    def save_metrics(self, metrics_dict, filename=None):
        """Save metrics to JSON file."""
        if filename is None:
            filename = f"metrics_{self.timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to Python native types
        clean_metrics = self._clean_metrics_for_json(metrics_dict)
        
        with open(filepath, 'w') as f:
            json.dump(clean_metrics, f, indent=4)
        
        print(f"Metrics saved to {filepath}")
        return filepath
    
    def save_model(self, model, model_name):
        """Save trained model using pickle."""
        filename = f"{model_name.replace(' ', '_')}_{self.timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model Saved to {filepath}")
        return filepath
    
    def save_dataframe(self, df, filename):
        """Save DataFrame to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f" DataFrame saved to {filepath}")
        return filepath
    
    @staticmethod
    def _clean_metrics_for_json(obj):
        """Recursively convert numpy types to Python native types."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: ResultsLogger._clean_metrics_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ResultsLogger._clean_metrics_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def generate_summary_report(self, comparison_results, best_model_name):
        """Generate a text summary report."""
        filename = f"summary_report_{self.timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("BREAST CANCER CLASSIFICATION - MODEL COMPARISON REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-" * 70 + "\n")
            
            for model_name, results in comparison_results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Mean F1 Score: {results['mean']:.4f}\n")
                f.write(f"  Std Deviation: {results['std']:.4f}\n")
                f.write(f"  CV Scores: {[f'{s:.4f}' for s in results['scores']]}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"BEST MODEL: {best_model_name}\n")
            f.write(f"   F1 Score: {comparison_results[best_model_name]['mean']:.4f}\n")
            f.write("=" * 70 + "\n")
        
        print(f" Summary report saved to {filepath}")
        return filepath


class ProgressTracker:
    """Track and display pipeline progress."""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_descriptions = []
    
    def update(self, description):
        """Update progress with step description."""
        self.current_step += 1
        self.step_descriptions.append(description)
        
        progress_bar = self._create_progress_bar()
        print(f"\n[Step {self.current_step}/{self.total_steps}] {description}")
        print(progress_bar)
    
    def _create_progress_bar(self):
        """Create visual progress bar."""
        progress = self.current_step / self.total_steps
        bar_length = 50
        filled_length = int(bar_length * progress)
        
        bar = '' * filled_length + '' * (bar_length - filled_length)
        percentage = progress * 100
        
        return f"|{bar}| {percentage:.1f}%"
    
    def complete(self):
        """Mark pipeline as complete."""
        print("\n" + "=" * 70)
        print(" Pipeline completed successfully!")
        print("=" * 70)


def load_model(filepath):
    """Load a saved model from pickle file."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f" Model loaded from {filepath}")
    return model


def print_section_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70 + "\n")


def print_metrics_table(metrics_dict):
    """Print metrics in a formatted table."""
    print("\n" + "-" * 50)
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 50)
    
    for metric_name, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"{metric_name:<20} {value:<15.4f}")
        else:
            print(f"{metric_name:<20} {value:<15}")
    
    print("-" * 50 + "\n")


def create_feature_summary(X_train, feature_rankings):
    """Create summary statistics for features."""
    summary = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean': X_train.mean(),
        'Std': X_train.std(),
        'Min': X_train.min(),
        'Max': X_train.max(),
        'RFE_Ranking': feature_rankings['Ranking'].values
    })
    
    return summary.sort_values('RFE_Ranking')


def validate_data_integrity(dataframe, target_column):
    """Perform basic data validation checks."""
    checks = {
        'null_values': dataframe.isnull().sum().sum(),
        'duplicate_rows': dataframe.duplicated().sum(),
        'target_classes': dataframe[target_column].nunique(),
        'negative_values': (dataframe.select_dtypes(include=['float64', 'int64']) < 0).sum().sum()
    }
    
    print("Data Integrity Checks:")
    print("-" * 50)
    for check_name, result in checks.items():
        status = "PASS" if result == 0 or check_name == 'target_classes' else " WARNING"
        print(f"{check_name:<30} {result:<10} {status}")
    print("-" * 50 + "\n")
    
    return checks


def format_confusion_matrix_summary(cm):
    """Extract and format confusion matrix metrics."""
    tn, fp, fn, tp = cm.ravel()
    
    summary = {
        'True Positives (TP)': int(tp),
        'True Negatives (TN)': int(tn),
        'False Positives (FP)': int(fp),
        'False Negatives (FN)': int(fn),
        'Sensitivity (Recall)': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0
    }
    
    return summary