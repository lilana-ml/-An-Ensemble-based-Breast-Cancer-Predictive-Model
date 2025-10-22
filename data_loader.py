"""
Data loading and initial inspection utilities.
"""

import pandas as pd
import warnings
from config import DATA_PATH, DROP_COLUMNS

warnings.filterwarnings('ignore')


class DataLoader:
    """Handle all data loading operations."""
    
    def __init__(self, filepath=DATA_PATH, separator="\t"):
        self.filepath = filepath
        self.separator = separator
        self.data = None
    
    def load_data(self):
        """Load dataset from CSV file."""
        try:
            self.data = pd.read_csv(self.filepath, sep=self.separator)
            print(f"✓ Data loaded successfully: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def get_data_info(self):
        """Display comprehensive data information."""
        if self.data is None:
            print("No data loaded.")
            return
        
        info_dict = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.value_counts().to_dict(),
            'missing_values': self.data.isnull().sum().sum(),
            'duplicates': self.data.duplicated().sum()
        }
        return info_dict
    
    def preview_data(self, n=5):
        """Show first n rows of dataset."""
        return self.data.head(n) if self.data is not None else None


def fetch_dataset(filepath=DATA_PATH):
    """Convenience function to quickly load data."""
    loader = DataLoader(filepath)
    return loader.load_data()