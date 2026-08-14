from setuptools import setup, find_packages

setup(
    name="sfii_preproc",
    version="1.0.0",
    description="Geospatial preprocessing pipeline for SFII Research Laboratory",
    author="SFII Research Laboratory",
    packages=find_packages(),
    install_requires=[
        "rasterio>=1.3.8",
        "xarray>=2023.7.0",
        "rioxarray>=0.14.2",
        "dask[complete]>=2023.7.1",
        "numpy>=1.24.3",
        "scipy>=1.11.1",
        "pyyaml>=6.0.1",
        "pandas>=2.0.3",
        "geopandas>=0.13.2",
        "zarr>=2.16.0",
        "pyarrow>=12.0.0",
        "torch>=2.0.0",
        "pytest>=7.3.1"
    ],
    entry_points={
        "console_scripts": [
            "sfii-preproc=pipeline:run_pipeline",
        ],
    },
)
