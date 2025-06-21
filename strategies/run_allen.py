from finlab import data
import finlab
from pandas import MultiIndex
import pandas as pd
import json
import yaml
import os
import generate_signal
from backtest.allen import *
import numpy as np

#  Input: Target = [stock ids]
# Output: Target = [stock ids] but filtered out those we do not have their data.
def filter_bad_targets(Target, cfg):
    bad = []
    for target in Target:
        try:
            df_stock = pd.read_csv(f"{cfg['root_path']}/{target}.csv")
        except:
            print(f"{target} is bad")
            bad.append(target)
            continue
        df_stock["high_grow"] = df_stock['etl:adj_high'] / df_stock['etl:adj_close'].shift(1)
        df_stock[ "low_grow"] =  df_stock['etl:adj_low'] / df_stock['etl:adj_close'].shift(1)
    for b in bad:
        Target.remove(b)
    return Target

#  Input: Given Target = [stock ids]
# Output: df with the format for backtest
def get_price_df(Target):
    open = data.get('price:開盤價')[Target]
    close = data.get('price:收盤價')[Target]
    
    close = pd.DataFrame(close)
    open = pd.DataFrame(open)
    
    open.columns = pd.MultiIndex.from_product([open.columns, ['open']], names=['Ticker', 'Price Type'])
    close.columns = pd.MultiIndex.from_product([close.columns, ['close']], names=['Ticker', 'Price Type'])
    
    price_df = pd.concat([close, open], axis=1)
    price_df = price_df.sort_index(axis=1, level=1) 
    price_df = price_df.sort_index(axis=1, level=0, sort_remaining=True) 
    
    price_df.columns = MultiIndex.from_tuples(
        [(col[0], 'Close' if col[1] == 'close' else 'Open') for col in price_df.columns]
    )
    price_df.fillna(method="bfill", inplace=True)
    return price_df

def print_result(returns):
    yearly_roi = returns.resample("Y").last() / returns.resample("Y").first() - 1
    for year in yearly_roi.index:
        print(f"{year.year} ROI: {yearly_roi[year]}")
    print("avg ROI:", (returns.iloc[-1] / returns.iloc[0]) ** (240/returns.shape[0]) - 1)
    pct = returns.pct_change().dropna()
    print("sharpe:", pct.mean() / pct.std() * np.sqrt(240), "\n\n")
    return 


if __name__ == "__main__":
    f = open("config/backtest.yaml")
    cfg_dict_backtest = yaml.safe_load(f)
    f = open("config/main_training.yaml")
    cfg_dict_train = yaml.safe_load(f)
    cfg = cfg_dict_backtest | cfg_dict_train
    
    result_dir = "results/Example_Result"
    buy_dfs, sell_dfs = pd.DataFrame(), pd.DataFrame()
    for dir in os.listdir(result_dir):
        pred_df = pd.read_csv(os.path.join(result_dir, dir, "test/pred_pct.csv"), index_col="date")
        val_df = pd.read_csv(os.path.join(result_dir, dir, "train_val/pred_pct.csv"), index_col="date")
        buy_dfs  = pd.concat([ buy_dfs, generate_signal.generate_buy_signal(pred_df, "allen", val_df)] , axis=0)
        sell_dfs = pd.concat([sell_dfs, generate_signal.generate_sell_signal(pred_df, "allen", val_df)], axis=0)
    buy_dfs  =  buy_dfs.sort_index()
    sell_dfs = sell_dfs.sort_index()

    Target = filter_bad_targets(buy_dfs.columns, cfg)
        
    finlab.login('ntSS3778pZi2FfkeYxXP0p+S0iI4AggkcphAUxh/lTVrWqT2FreKQsDkTA92CM7d#vip_m')
    price_df = get_price_df(Target)
    price_df = price_df[(price_df.index >= buy_dfs.index[0]) & (price_df.index <= buy_dfs.index[-1])]
    
    backtest = Backtest(Strategy, price_df, commission=cfg["commission"], cash=1e9)
    result           = backtest.run(buy_dfs, sell_dfs, targets=Target, max_positions=len(Target), is_benchmark = False)
    result_benchmark = backtest.run(buy_dfs, sell_dfs, targets=Target, max_positions=len(Target), is_benchmark = True)

    print("If trading by our strategy:")
    print_result(result.returns)
    print("If trading by benchmark:")
    print_result(result_benchmark.returns)
