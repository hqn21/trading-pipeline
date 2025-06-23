import numpy as np
import pandas as pd

def print_result(returns: pd.Series):
    yearly_roi = returns.resample("Y").last() / returns.resample("Y").first() - 1
    for year in yearly_roi.index:
        print(f"{year.year} ROI: {yearly_roi[year]}")
    print("avg ROI:", (returns.iloc[-1] / returns.iloc[0]) ** (240/returns.shape[0]) - 1)
    pct = returns.pct_change().dropna()
    print("sharpe:", pct.mean() / pct.std() * np.sqrt(240), "\n\n")
    return 