import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    dataset = openml.datasets.get_dataset(40926)
    df, _, _, _ = dataset.get_data()
    return df

def preprocess_data(df):
    # Convert the 'class' column to integers
    df['class'] = df['class'].astype(int)
    # Separate features and labels
    X = df.drop(columns=['class']).values
    y = df['class'].values
    return X, y

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def get_image(row):
    # Reshape a flattened row to a 32x32 RGB image
    img = row.reshape(3, 32, 32).transpose(1, 2, 0)
    return img
