import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset_name="cifar", file_path=None):
    """
    Load dataset based on dataset_name.
    Get CIFAR-S dataset with openml.
    Get HAM10000 dataset with link: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download&select=hmnist_28_28_RGB.csv
    """
    if dataset_name == "cifar":
        dataset = openml.datasets.get_dataset(40926)  # CIFAR-10
        df, _, _, _ = dataset.get_data()
    elif dataset_name == "ham10000":
        if file_path is None:
            raise ValueError("file_path must be provided for HAM10000")
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported dataset_name")
    return df

def preprocess_data(df, dataset_name="cifar"):
    """Preprocess data based on dataset_name."""
    if dataset_name == "cifar":
        df['class'] = df['class'].astype(int)
        X = df.drop(columns=['class']).values  # 3072 features (32x32x3)
        y = df['class'].values
    elif dataset_name == "ham10000":
        X = df.iloc[:, :2352].values  # First 2352 columns are pixel values
        y = df.iloc[:, 2352].values   # Last column is the label

        # Map labels to integers
        if not np.issubdtype(y.dtype, np.integer):
            unique_labels = np.unique(y)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
    else:
        raise ValueError("Unsupported dataset_name")
    return X, y

def split_data(X, y, test_size=0.2):
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)