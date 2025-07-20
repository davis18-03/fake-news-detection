import os
from typing import Tuple

import pandas as pd


def load_data(data_dir: str) -> pd.DataFrame:
    """Load and label fake and true news data from CSV files in the specified directory."""
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")
    if not os.path.isfile(fake_path) or not os.path.isfile(true_path):
        raise FileNotFoundError(
            f"Required data files not found in {data_dir}. Please add 'Fake.csv' and 'True.csv'."
        )
    try:
        fake = pd.read_csv(fake_path)
        true = pd.read_csv(true_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV files: {e}")
    if "text" not in fake.columns or "text" not in true.columns:
        raise ValueError("Both CSV files must contain a 'text' column.")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    return df


def shuffle_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Shuffle the dataframe and reset index using the provided random seed."""
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
