from typing import List, Any
import pandas as pd

def apply_rolling_window(list : List[Any], window_size: int):
    series = pd.Series(list)
    rolling_window = series.rolling(window=window_size, min_periods=1)
    return rolling_window.mean().to_list()