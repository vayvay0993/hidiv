import pandas as pd
import numpy as np
import os
from copy import deepcopy
from typing import Union


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
    printEnd: str = "\r",
) -> None:
    """
    Call in a loop to create terminal progress bar
    Args:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    Returns:
        None
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    if iteration == total:
        print()


def log(path, *args):
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join([x for x in args]) + "\n")


def find_left_right(
    a: pd.Series = None, v: pd.Series = None, q: pd.Series = None, way: str = "left"
):
    """
    Args:
        a: 被對照組, ex: np.array([1, 2, 3, 4, 5])
        v: 主要array找對應的a日期, 產出長度與v相同, np.array([-1000, -10, 1, 2, 3, 5, 10, 999, 4, 1.5])
        q: 其他欄位
        way: 找大於等於或是小於等於
    Returns:
    """

    # right_idx = np.searchsorted(a, v, side='right')
    # left = np.where(v < a.min(), np.NaN, a[right_idx - 1])

    a = np.array(a)
    v = np.array(v)
    if way == "left":
        right_idx = np.searchsorted(a, v, side="right")
        if q is not None:
            q = np.array(q)
            left = np.where(v < a.min(), np.NaN, q[right_idx - 1])
        else:
            left = np.where(v < a.min(), np.NaN, a[right_idx - 1])
        return left
    elif way == "right":
        right_idx = np.searchsorted(a, v, side="left")
        right_idx = np.clip(right_idx, 0, len(a) - 1)
        if q is not None:
            q = np.array(q)
            right = np.where(v > a.max(), np.NaN, q[right_idx])
        else:
            right = np.where(v > a.max(), np.NaN, a[right_idx])
        return right


def fill_na_until_last_valid(dataFrame: pd.DataFrame) -> pd.DataFrame:
    """fill na values with last non na values when there is non na value under it

    Args:
        dataFrame (pd.DataFrame): dataFrame that will be filled, should be a pivot table data
    Returns:
        pd.DataFrame
    """
    result = dataFrame.copy()
    result.loc[: result.last_valid_index()] = result.loc[
        : result.last_valid_index()
    ].fillna(method="ffill")

    return result


def create_name(prefix: str = None, suffix: dict = None) -> str:
    """
        用於創建資料夾名稱

    Args:
        prefix (string): 資料夾名稱前墜, 如model
        suffix (dictionary): 資料夾名稱後墜, 如 {'subsample': 1, 'seed': 1}
    Returns:
        string: 如: model_subsample_1_seed_1

    """
    suffix_string = "_".join([f"{key}_{suffix[key]}" for key in suffix])
    n = f"{prefix}_{suffix_string}"
    return n


if __name__ == "__main__":
    print(create_name("model", {"subsample": 1, "seed": 1}))
    # send notification
    # notification.notify("Notification Title","Notification Message", timeout=10    )
