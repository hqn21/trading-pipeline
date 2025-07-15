import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import finlab
from finlab import data
from evaluation.stats import sharpe
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

def get_equal_weight_baseline_result(result_dir: str, start_date: pd.DatetimeIndex, end_date: pd.DatetimeIndex):
    dir = os.listdir(result_dir)[0]
    pred_df = pd.read_csv(os.path.join(result_dir, dir, "test/pred_pct.csv"), index_col="date")
    close_price = data.get('etl:adj_close')[pred_df.columns]
    close_price = close_price[(close_price.index >= start_date) & (close_price.index <= end_date)]
    for col in close_price.columns:
        close_price[col] = close_price[col] / close_price[col].iloc[0]
    baseline_returns = close_price.mean(axis=1)
    baseline_result = Result(baseline_returns)
    return baseline_result

def get_sharpe_top10_baseline_result(result_dir: str, start_date: pd.DatetimeIndex, end_date: pd.DatetimeIndex):
    dir = os.listdir(result_dir)[0]
    pred_df = pd.read_csv(os.path.join(result_dir, dir, "test/pred_pct.csv"), index_col="date")
    stock_ids = pred_df.columns
    close_price = data.get('etl:adj_close')[stock_ids]

    validation_start = start_date - pd.Timedelta(days=960)

    validation_close_price = close_price[(close_price.index >= validation_start) & (close_price.index <= start_date)]

    sharpe_scores = {}
    for column in validation_close_price.columns:
        sharpe_scores[column] = sharpe(validation_close_price[column])

    sharpe_scores = pd.Series(sharpe_scores)
    top_10_stocks = sharpe_scores.nlargest(10).index
    top_10_close_price = close_price[(close_price.index >= start_date) & (close_price.index <= end_date)][top_10_stocks]

    for col in top_10_close_price.columns:
        top_10_close_price[col] = top_10_close_price[col] / top_10_close_price[col].iloc[0]
    baseline_returns = top_10_close_price.mean(axis=1)
    baseline_returns
