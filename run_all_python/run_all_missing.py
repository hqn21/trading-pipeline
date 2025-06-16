#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import subprocess
import yaml
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# 4 種 split 設定（和原本完全一致）
split_sets = [
    ["2018-01-01","2020-01-01","2021-01-01","2022-01-01"],  # idx 0 → test start 2021
    ["2019-01-01","2021-01-01","2022-01-01","2023-01-01"],  # idx 1 → test start 2022
    ["2020-01-01","2022-01-01","2023-01-01","2024-01-01"],  # idx 2 → test start 2023
    ["2021-01-01","2023-01-01","2024-01-01","2025-01-01"],  # idx 3 → test start 2024
]

# 只列出「跑失敗」、需要重跑的 (category, data, loss, [split_idx, ...])
missing_tasks = [
    ('OptoTSE', 'Dataset_Abs', 'mse', [1]),
    ('SelectedVer2', 'Dataset_Abs', 'rank', [0]),
]

# 載入一次 base config
with open("config/training_config.yaml") as f:
    base_cfg = yaml.safe_load(f)

def run_task(category, data, loss, splits):
    """生成臨時 config 並呼叫 run.py。內部捕獲錯誤、不會中斷其他任務。"""
    cfg = base_cfg.copy()
    cfg["category"]         = category
    cfg["data"]             = data
    cfg["loss"]             = loss
    cfg["split_dates"]      = splits
    cfg["result_file_name"] = f"{category}_{data}_{loss}"
    
    if data == "Dataset_Pct":
        cfg["goal"] == 'max_roi'
    elif data == "Dataset_Abs":
        cfg["goal"] == 'max_price'

    fd, tmp_path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w") as tmpf:
            yaml.safe_dump(cfg, tmpf)

        env = os.environ.copy()
        # env["CUDA_VISIBLE_DEVICES"] = "2" # 在上面設定
        subprocess.run(
            ["python", "run.py", "--config", tmp_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        print(f"[ OK ] {category} | {data}_{loss} | splits={splits[2][:4]}")
    except subprocess.CalledProcessError as e:
        print(f"[ERR ] {category} | {data}_{loss} | splits={splits[2][:4]} | exit {e.returncode}")
        print(e.stderr.decode().strip())
    except Exception as e:
        print(f"[ERR ] {category} | {data}_{loss} | splits={splits[2][:4]} | exception {e}")
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    # 組成要執行的所有任務列表
    tasks = []
    for category, data, loss, idxs in missing_tasks:
        for idx in idxs:
            tasks.append((category, data, loss, split_sets[idx]))

    # 最多 4 個 worker 平行
    with ThreadPoolExecutor(max_workers=2) as exe:
        futures = [exe.submit(run_task, cat, dt, ls, sp) for cat, dt, ls, sp in tasks]
        for _ in as_completed(futures):
            pass

    print("\n—— 所有缺失任務已完成 ——")
