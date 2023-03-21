"""
Large Numpy arrays and Pandas DataFrames.

Source: https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html
"""
from typing import overload
from collections.abc import Sequence
import pandas as pd
import geopandas as gpd
import numpy as np

from .shared_numpyarray import SharedNumpyArray


class PositionIndexer:
    def __init__(self, array, index, geometry):
        self._array = array
        self._index = index
        self._geometry = geometry

    def __getitem__(self, key):
        labels = self._index[key]
        return gpd.GeoDataFrame(
            self._array[key], index=labels, geometry=self._geometry
        )


class LabelIndexer:
    def __init__(self, array, index, columns, geometry):
        self._array = array
        self._index = index
        self._geometry = geometry
        self._columns = columns

    @overload
    def __getitem__(self, key: Sequence | np.ndarray | pd.Series) -> gpd.GeoDataFrame: ...

    @overload
    def __getitem__(self, key: int | str) -> pd.Series: ...

    def __getitem__(self, key) -> pd.Series | gpd.GeoDataFrame:
        if np.isscalar(key):
            index = list(self._index).index(key)
            return pd.Series(
                self._array[index, :], name=key, index=self._columns
            )
        else:
            if pd.api.types.is_bool_dtype(key):
                index = key.values
                labels = self._index[index]
            else:
                index = self._index.get_indexer(key)
                labels = key
                if -1 in index:
                    raise KeyError("Index not in Shared DataFrame")
            try:
                return gpd.GeoDataFrame(
                    self._array[index, :],
                    index=labels,
                    geometry=self._geometry,
                    columns=self._columns,
                )
            except ValueError as e:
                print("index", index)
                print("index size asked", index.shape)
                print("index size in memory", self._index.shape)
                raise e


class SharedGeoDataFrame:
    """Wraps a GeoDataFrame so that it can be shared quickly among processes, avoiding unnecessary copying and (de)serializing."""

    def __init__(self, df):
        """Create the shared memory and copies the dataframe therein."""
        self._values = SharedNumpyArray(df.values)
        self._index = df.index
        self._columns = df.columns
        self._geometry = df.geometry.name
        self.iloc = PositionIndexer(
            self._values.array, self._index, self._geometry
        )
        self.loc = LabelIndexer(
            self._values.array, self._index, self._columns, self._geometry
        )

    def read(self):
        """Read the dataframe from the shared memory without unnecessary copying."""
        return gpd.GeoDataFrame(
            self._values.read(),
            index=self._index,
            columns=self._columns,
            geometry=self._geometry,
        )

    def copy(self):
        """Return a new copy of the dataframe stored in shared memory."""
        return pd.DataFrame(
            self._values.copy(), index=self._index, columns=self._columns
        )

    def unlink(self):
        """
        Release the allocated memory.

        Call when finished using the data, or when the data was copied somewhere else.
        """
        self._values.unlink()

    def __getitem__(self, columns):
        if np.isscalar(columns):
            result = np.where(self._columns == columns)[0]
            if not result:
                available_cols = ", ".join(self._columns)
                raise IndexError(f"Column '{columns}' not found. Available columns: {available_cols}")
            indexer = int(result)
            if self._geometry == columns:
                return gpd.GeoSeries(
                    self._values.array[:, indexer],
                    name=columns,
                    index=self._index,
                )
            return pd.Series(
                self._values.array[:, indexer],
                name=columns,
                index=self._index,
            )
        else:
            indexer = np.where(self._columns.isin([columns]))[0]
            if self._geometry in columns:
                return gpd.GeoDataFrame(
                    self._values.array[:, indexer],
                    columns=columns,
                    index=self.index,
                    geometry=self._geometry,
                )
            return pd.DataFrame(
                self._values.array[:, indexer],
                columns=columns,
                index=self.index,
            )
