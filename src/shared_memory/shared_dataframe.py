"""
Large Numpy arrays and Pandas DataFrames.

Source: https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html
"""
import pandas as pd
import numpy as np

from .shared_numpyarray import SharedNumpyArray


class PositionIndexer:
    def __init__(self, array, index: pd.Index):
        self._array = array
        self._index = index

    def __getitem__(self, key):
        labels = self._index[key]
        return pd.DataFrame(self._array[key], index=labels)


class LabelIndexer:
    def __init__(self, array, index: pd.Index):
        self._array = array
        self._index = index

    def __getitem__(self, index):
        if np.isscalar(index):
            index = [index]
        indexer = self._index.get_indexer(index)
        if -1 in indexer:
            raise KeyError("Index not in Shared DataFrame")
        return pd.DataFrame(self._array[indexer], index=index)


class SharedDataFrame:
    """Wraps a pandas dataframe so that it can be shared quickly among processes, avoiding unnecessary copying and (de)serializing."""

    def __init__(self, df):
        """Create the shared memory and copies the dataframe therein."""
        self._values = SharedNumpyArray(df.values)
        self._index = df.index
        self._columns = df.columns
        self.iloc = PositionIndexer(self._values.array, self._index)
        self.loc = LabelIndexer(self._values.array, self._index)

    def read(self):
        """Read the dataframe from the shared memory without unnecessary copying."""
        return pd.DataFrame(
            self._values.read(), index=self._index, columns=self._columns
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
            if result.shape[0] == 0:
                available_cols = ", ".join(self._columns)
                raise IndexError(f"Column '{columns}' not found. Available columns: {available_cols}")
            indexer = int(result)
            return pd.Series(
                self._values.array[:, indexer],
                name=columns,
                index=self._index,
            )
        else:
            indexer = np.where(self._columns.isin([columns]))[0]
            return pd.DataFrame(
                self._values.array[:, indexer],
                columns=columns,
                index=self.index,
            )
