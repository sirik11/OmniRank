"""
data_utils.py
-------------

Helper functions for reading and writing data in the extended feed ranking project.  These
functions abstract away some of the boilerplate for working with Parquet files and big
data stores.  In a real system, you could extend them to support S3, GCS, or Delta
Lake tables.
"""

import pandas as pd
import pyarrow.parquet as pq


def read_parquet(path: str) -> pd.DataFrame:
    """Read a Parquet file into a Pandas DataFrame."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: str):
    """Write a Pandas DataFrame to a Parquet file."""
    df.to_parquet(path)
