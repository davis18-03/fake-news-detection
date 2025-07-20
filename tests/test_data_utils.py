import os
import pandas as pd
import pytest
from src.data_utils import load_data, shuffle_data

def test_load_data(tmp_path):
    # Create fake and true CSVs
    fake = pd.DataFrame({'text': ['fake news 1', 'fake news 2']})
    true = pd.DataFrame({'text': ['real news 1', 'real news 2']})
    data_dir = tmp_path
    fake.to_csv(os.path.join(data_dir, 'Fake.csv'), index=False)
    true.to_csv(os.path.join(data_dir, 'True.csv'), index=False)
    df = load_data(str(data_dir))
    assert set(df['label']) == {0, 1}
    assert len(df) == 4
    assert 'text' in df.columns

def test_load_data_missing_files(tmp_path):
    # No files present
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path))

def test_load_data_malformed_csv(tmp_path):
    # Create malformed CSV
    data_dir = tmp_path
    with open(os.path.join(data_dir, 'Fake.csv'), 'w') as f:
        f.write('not,a,valid,csv\n')
    with open(os.path.join(data_dir, 'True.csv'), 'w') as f:
        f.write('not,a,valid,csv\n')
    with pytest.raises(ValueError):
        load_data(str(data_dir))

def test_load_data_missing_text_column(tmp_path):
    # Create CSVs without 'text' column
    fake = pd.DataFrame({'foo': ['bar']})
    true = pd.DataFrame({'foo': ['baz']})
    data_dir = tmp_path
    fake.to_csv(os.path.join(data_dir, 'Fake.csv'), index=False)
    true.to_csv(os.path.join(data_dir, 'True.csv'), index=False)
    with pytest.raises(ValueError):
        load_data(str(data_dir))

def test_shuffle_data():
    df = pd.DataFrame({'text': [str(i) for i in range(10)], 'label': [0, 1]*5})
    shuffled = shuffle_data(df, seed=123)
    assert set(shuffled['text']) == set(df['text'])
    assert not all(shuffled['text'] == df['text'])
    assert list(shuffled.index) == list(range(10)) 