import numpy as np
import pandas as pd
from dataclasses import dataclass
@dataclass
class Result:
    returns: pd.Series

def print_result(returns: pd.Series):
    yearly_roi = returns.resample("Y").last() / returns.resample("Y").first() - 1
    for year in yearly_roi.index:
        print(f"{year.year} ROI: {yearly_roi[year]}")
    print("avg ROI:", (returns.iloc[-1] / returns.iloc[0]) ** (240/returns.shape[0]) - 1)
    pct = returns.pct_change().dropna()
    print("sharpe:", pct.mean() / pct.std() * np.sqrt(240), "\n\n")
    return 

def get_benchmark_result(result_dir: str, start_date: pd.DatetimeIndex, end_date: pd.DatetimeIndex):
    import os
    import finlab
    from finlab import data
    finlab.login('ntSS3778pZi2FfkeYxXP0p+S0iI4AggkcphAUxh/lTVrWqT2FreKQsDkTA92CM7d#vip_m')
    dir = os.listdir(result_dir)[0]
    pred_df = pd.read_csv(os.path.join(result_dir, dir, "test/pred_pct.csv"), index_col="date")
    close_price = data.get('etl:adj_close')[pred_df.columns]
    close_price = close_price[(close_price.index >= start_date) & (close_price.index <= end_date)]
    for col in close_price.columns:
        close_price[col] = close_price[col] / close_price[col].iloc[0]
    benchmark_returns = close_price.mean(axis=1)
    benchmark_result = Result(benchmark_returns)
    return benchmark_result
