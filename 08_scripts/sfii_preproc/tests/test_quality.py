import pytest
from core.quality import verify_crs, verify_dataset_integrity

def test_verify_crs_invalid():
    # Providing a non-existent file will result in False due to exception handling
    assert verify_crs("non_existent_file.tif") == False

def test_verify_dataset_integrity_invalid():
    # Providing a non-existent file will fail integrity check
    assert verify_dataset_integrity("non_existent_file.tif") == False
