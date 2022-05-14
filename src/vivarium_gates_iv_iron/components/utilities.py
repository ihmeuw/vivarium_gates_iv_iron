import pandas as pd

from vivarium.framework.engine import Builder


def load_and_unstack(builder: Builder, data_key: str, unstack_col: str) -> pd.DataFrame:
    data = builder.data.load(data_key)
    idx_cols = data.columns.difference(['value', unstack_col])
    data = data.pivot(index=idx_cols, columns=unstack_col, values='value')
    return data.reset_index()
