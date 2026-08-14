import pytest
import xarray as xr
import numpy as np
from core.features import compute_indices, generate_annual_summaries

@pytest.fixture
def sample_dataset():
    # Create a small dataset with synthetic bands
    time = [np.datetime64('2018-01-01'), np.datetime64('2018-02-01')]
    
    # 2x2 grid, 2 time steps
    shape = (2, 2, 2)
    
    ds = xr.Dataset(
        {
            "B2": (("time", "y", "x"), np.random.rand(*shape)),
            "B3": (("time", "y", "x"), np.random.rand(*shape)),
            "B4": (("time", "y", "x"), np.random.rand(*shape)),
            "B8": (("time", "y", "x"), np.random.rand(*shape)),
            "B11": (("time", "y", "x"), np.random.rand(*shape)),
            "B12": (("time", "y", "x"), np.random.rand(*shape))
        },
        coords={
            "time": time,
            "y": [10.5, 10.6],
            "x": [76.0, 76.1]
        }
    )
    return ds

def test_compute_indices(sample_dataset):
    ds_out = compute_indices(sample_dataset)
    
    # Check that indices were added
    assert 'NDVI' in ds_out.data_vars
    assert 'NBR' in ds_out.data_vars
    assert 'EVI2' in ds_out.data_vars
    assert 'NDWI' in ds_out.data_vars
    assert 'TCW' in ds_out.data_vars
    
    # Check shape
    assert ds_out['NDVI'].shape == (2, 2, 2)
    
def test_generate_annual_summaries(sample_dataset):
    # Compute indices first
    ds_with_indices = compute_indices(sample_dataset)
    
    # Generate summaries
    summary = generate_annual_summaries(ds_with_indices)
    
    # Check that summary has _mean, _min, _max, _std, _amp for each var
    assert 'NDVI_mean' in summary.data_vars
    assert 'NDVI_max' in summary.data_vars
    assert 'NDVI_amp' in summary.data_vars
    
    # Time dimension should now be reduced to years (just 2018 in this case)
    assert summary.sizes['year'] == 1
