"""
Create visualizations for EDA and model evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, DetCurveDisplay
from config import FIGURE_SIZE, HEATMAP_SIZE, COLOR_PALETTE

sns.set_style("darkgrid")


class DataVisualizer:
    """Visualization utilities for EDA."""
    
    @staticmethod
    def plot_target_distribution(y, title="Target Distribution"):
        """Visualize target variable distribution."""
        fig, ax = plt.subplots(figsize=(8, 6))
        y_counts = y.value_counts()
        
        bars = ax.bar(['Benign (0)', 'Malignant (1)'], y_counts.values, 
                      color=['skyblue', 'salmon'], edgecolor='black')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=12)
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(dataframe, figsize=HEATMAP_SIZE):
        """Generate correlation heatmap."""
        correlation = dataframe.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(correlation, cmap=COLOR_PALETTE, annot=True,
                   linewidths=0.5, center=0, square=True,
                   cbar_kws={"shrink": 0.5}, ax=ax, fmt='.2f')
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_feature_rankings(feature_df, top_n=15):
        """Visualize feature importance rankings."""
        top_features = feature_df.nsmallest(top_n, 'Ranking')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(top_features['Feature'], top_features['Ranking'], 
                       color='steelblue', edgecolor='black')
        
        ax.set_xlabel('Ranking (1 = Most Important)', fontsize=12)
        ax.set_title(f'Top {top_n} Features by RFE Ranking', 
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()


class ModelVisualizer:
    """Visualization utilities for model evaluation."""
    
    @staticmethod
    def plot_confusion_matrix(cm, model_name, labels=['Benign', 'Malignant']):
        """Display confusion matrix with annotations."""
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_model_comparison_boxplot(comparison_results, metric='f1'):
        """Create boxplot comparing model performances."""
        names = list(comparison_results.keys())
        scores = [comparison_results[name]['scores'] for name in names]
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        bp = ax.boxplot(scores, labels=names, patch_artist=True, showmeans=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Annotate means
        means = [comparison_results[name]['mean'] for name in names]
        for i, mean in enumerate(means):
            ax.text(i + 1, mean + 0.005, f"{mean:.3f}", 
                   ha='center', fontsize=10, fontweight='bold')
        
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=12)
        ax.set_title(f'Model Comparison - {metric.upper()} Scores', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_and_det_curves(models_dict, X_test, y_test):
        """Plot ROC and DET curves for model comparison."""
        fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(14, 6))
        
        for name, model in models_dict.items():
            RocCurveDisplay.from_estimator(model, X_test, y_test, 
                                          ax=ax_roc, name=name)
            DetCurveDisplay.from_estimator(model, X_test, y_test, 
                                          ax=ax_det, name=name)
        
        ax_roc.set_title("ROC Curves", fontsize=14, fontweight='bold')
        ax_det.set_title("DET Curves", fontsize=14, fontweight='bold')
        ax_roc.grid(linestyle="--", alpha=0.7)
        ax_det.grid(linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        plt.show()


def visualize_eda(dataframe, target_column='diagnosis'):
    """Run complete EDA visualization pipeline."""
    viz = DataVisualizer()
    viz.plot_target_distribution(dataframe[target_column])
    viz.plot_correlation_heatmap(dataframe)


def visualize_model_results(models, comparison_results, X_test, y_test):
    """Run complete model visualization pipeline."""
    viz = ModelVisualizer()
    viz.plot_model_comparison_boxplot(comparison_results)
    viz.plot_roc_and_det_curves(models, X_test, y_test)