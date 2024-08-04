# test_file_cases.py

import pytest
import pandas as pd

# Test case for file not found error
def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        pd.read_csv('nonexistent_file.csv')

# Test case for empty file
def test_empty_file(tmp_path):
    # Create an empty file
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    # Check if reading an empty file raises an EmptyDataError
    with pytest.raises(pd.errors.EmptyDataError):
        pd.read_csv(empty_file)
