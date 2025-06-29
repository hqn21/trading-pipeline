import pandas as pd


def compute_surge_metrics_from_buy_signals(test_pred: pd.DataFrame, test_truth: pd.DataFrame, buy_signals: pd.DataFrame, threshold=0.3):
    """
    Compute bottom-surge prediction metrics.

    Parameters:
    -----------
    test_pred : pandas.DataFrame
        Predicted scores for each stock and date.
    test_truth : pandas.DataFrame
        True scores for each stock and date.
    buy_signals : pandas.DataFrame or Series
        Boolean buy signal mask (same shape/index/columns as test_pred).
    threshold : float, optional
        Threshold above which a value is considered a "surge".

    Returns:
    --------
    dict
        {
            'precision_on_buy_signals': float,
            'actual_surge_rate_on_buy_signals': float,
            'overall_actual_surge_rate': float,
            'num_predicted_surges_on_buy': int,
            'num_correct_predicted_surges_on_buy': int,
            'num_actual_surges_on_buy': int,
            'num_buy_signals': int,
            'total_actual_surges': int,
            'total_elements': int
        }
    """
    if type(buy_signals.index) != pd.DatetimeIndex:
        buy_signals.index = pd.to_datetime(buy_signals.index)
    if type(test_pred.index) != pd.DatetimeIndex:
        test_pred.index = pd.to_datetime(test_pred.index)
    if type(test_truth.index) != pd.DatetimeIndex:
        test_truth.index = pd.to_datetime(test_truth.index)
        
    # 1. 构建掩码
    pred_surge_mask   = (test_pred  > threshold)
    actual_surge_mask = (test_truth > threshold)
    buy_mask          = buy_signals.astype(bool)

    # 2. 对齐检查
    assert buy_mask.shape == pred_surge_mask.shape == actual_surge_mask.shape
    assert buy_mask.index.equals(pred_surge_mask.index)
    assert buy_mask.columns.equals(pred_surge_mask.columns)

    # 3. 筛选 buy==True 位置
    pred_on_buy   = pred_surge_mask   & buy_mask
    actual_on_buy = actual_surge_mask & buy_mask

    # 4. 统计计数
    num_predicted_surges_on_buy      = int(pred_on_buy.sum().sum())
    num_correct_predicted_surges_on_buy = int((pred_on_buy & actual_on_buy).sum().sum())
    num_actual_surges_on_buy         = int(actual_on_buy.sum().sum())
    num_buy_signals                  = int(buy_mask.sum().sum())

    total_actual_surges = int(actual_surge_mask.sum().sum())
    total_elements      = actual_surge_mask.size

    # 5. 计算指标
    precision_on_buy_signals         = (num_correct_predicted_surges_on_buy / num_predicted_surges_on_buy
                                        if num_predicted_surges_on_buy else float("nan"))
    actual_surge_rate_on_buy_signals = (num_actual_surges_on_buy / num_buy_signals
                                        if num_buy_signals else float("nan"))
    overall_actual_surge_rate        = total_actual_surges / total_elements

    # 6. 返回结果
    return {
        'precision_on_buy_signals': precision_on_buy_signals,
        'actual_surge_rate_on_buy_signals': actual_surge_rate_on_buy_signals,
        'overall_actual_surge_rate': overall_actual_surge_rate,
        'num_predicted_surges_on_buy': num_predicted_surges_on_buy,
        'num_correct_predicted_surges_on_buy': num_correct_predicted_surges_on_buy,
        'num_actual_surges_on_buy': num_actual_surges_on_buy,
        'num_buy_signals': num_buy_signals,
        'total_actual_surges': total_actual_surges,
        'total_elements': total_elements
    }