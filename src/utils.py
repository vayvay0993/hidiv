import pandas as pd
from copy import deepcopy
import json


def get_rebalance_date(date_list, start_date, end_date, freq="Q"):
    next_four_seasons = [start_date]
    rebalance_date = start_date
    i = 0
    while rebalance_date < end_date:
        if freq == "Q":
            rebalance_date = start_date + pd.DateOffset(months=3 * (i + 1))
        elif freq == "M":
            rebalance_date = start_date + pd.DateOffset(months=1 * (i + 1))
        elif freq == "Y":
            rebalance_date = start_date + pd.DateOffset(years=1 * (i + 1))

        # check if rebalance_date is in date_list
        if rebalance_date in date_list:
            next_four_seasons.append(rebalance_date)
        else:
            # if not, find the next date in date_list
            for date in date_list:
                if date > rebalance_date:
                    # chang np.datetime64 to pd.Timestamp
                    rebalance_date = pd.Timestamp(date)
                    next_four_seasons.append(rebalance_date)
                    break
        i += 1
    return next_four_seasons


def get_monthly_end_rebalance_date(date_list, start_date, freq="Q"):
    # convert date_list to a pandas series for fast lookups
    date_series = pd.Series(date_list)

    # Map frequencies to their respective offsets
    freq_to_offset = {"Q": 3, "M": 1, "Y": 12}
    offset = freq_to_offset.get(freq, 1)  # default to 1 if frequency not recognized

    # Check if start_date or day before is in date_series
    if (start_date in date_series.values) or (
        (start_date - pd.DateOffset(days=1)) in date_series.values
    ):
        # Find the index of the start_date in date_series, if not found, get the last date before start_date
        start_date = date_series[date_series <= start_date].iat[-1]

    monthly_end_rebalance_date_lst = [start_date]

    rebalance_date = start_date
    i = 0
    while rebalance_date < date_series.iat[-1]:
        rebalance_date = start_date + pd.DateOffset(months=offset * (i + 1))
        rebalance_date = rebalance_date + pd.offsets.MonthEnd(0)

        # check if rebalance_date or day before is in date_series
        if (rebalance_date in date_series.values) or (
            (rebalance_date - pd.DateOffset(days=1)) in date_series.values
        ):
            # Find the index of the rebalance_date in date_series, if not found, get the last date before rebalance_date
            rebalance_date = date_series[date_series <= rebalance_date].iat[-1]
            monthly_end_rebalance_date_lst.append(rebalance_date)
        i += 1

    return monthly_end_rebalance_date_lst


def get_data_freezed_date(date_list, rebalance_date_list, days_shift=5):
    data_freezed_date_lst = []

    for rebalance_date in rebalance_date_list:
        # find the index for rebalance_date in date_list
        index = list(date_list).index(rebalance_date)
        # find the data_freezed_date by shifting the index by days_shift
        data_freezed_date = date_list[index - days_shift]
        data_freezed_date_lst.append(data_freezed_date)

    return data_freezed_date_lst


def concatenate_excel_sheets(excel_file_path: str) -> pd.DataFrame:
    # Load the Excel file
    xlsx = pd.read_excel(excel_file_path, sheet_name=None, engine="openpyxl")

    # Initialize an empty DataFrame to store the concatenated data
    concatenated_dataframe = pd.DataFrame()

    # Iterate through each sheet and append its data to the concatenated DataFrame
    for sheet_name, sheet_data in xlsx.items():
        concatenated_dataframe = pd.concat(
            [concatenated_dataframe, sheet_data], ignore_index=True
        )

    return concatenated_dataframe


def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from file: {file_path}")
        return None


def cmoney_data_clean_up(df):
    df = deepcopy(df)

    df = df.drop_duplicates()

    # handle the column name that has \u3000
    df.columns = [x.replace("\u3000", "") for x in df.columns]

    # turn the column of 日期 into datetime format YYYY-MM-DD
    if "日期" in df.columns:
        try:
            df["日期"] = pd.to_datetime(df["日期"], format="%Y%m%d")
        except:
            print("date format is not YYYYMMDD")
            try:
                df["日期"] = pd.to_datetime(df["日期"], format="%Y/%m/%d")
            except:
                print("date format is not YYYY/MM/DD")
                try:
                    df["日期"] = pd.to_datetime(df["日期"], format="%Y-%m-%d")
                except:
                    print("date format is not YYYY-MM-DD")
    return df


def tej_data_clean_up(df):
    df = deepcopy(df)

    df = df.drop_duplicates()

    # if '公司' is in the columns, split it into 股票代號 and 公司名稱
    if "公司" in df.columns:
        try:
            df.insert(0, "股票代號", df["公司"].apply(lambda x: x.split(" ")[0]))
            df.insert(0, "股票名稱", df["公司"].apply(lambda x: x.split(" ")[1]))
        except:
            print("column is already exist")

    if "證券代碼" in df.columns:
        try:
            df.insert(0, "股票代號", df["證券代碼"].apply(lambda x: x.split(" ")[0]))
            df.insert(0, "股票名稱", df["證券代碼"].apply(lambda x: x.split(" ")[1]))
        except:
            print("column is already exist")

    if "公司簡稱" in df.columns:
        try:
            df.insert(0, "股票代號", df["公司簡稱"].apply(lambda x: x.split(" ")[0]))
            df.insert(0, "股票名稱", df["公司簡稱"].apply(lambda x: x.split(" ")[1]))
        except:
            print("column is already exist")

    # turn the column of 年月日 into datetime format YYYY-MM-DD
    if "年月日" in df.columns:
        try:
            df["年月日"] = pd.to_datetime(df["年月日"], format="%Y%m%d")
        except:
            print("date format is not YYYYMMDD")
            try:
                df["年月日"] = pd.to_datetime(df["年月日"], format="%Y/%m/%d")
            except:
                print("date format is not YYYY/MM/DD")
                try:
                    df["年月日"] = pd.to_datetime(df["年月日"], format="%Y-%m-%d")
                except:
                    print("date format is not YYYY-MM-DD")
    return df


class CDF:
    def __init__(self, df):
        self.df = df

    def find_ticker(self, ticker):
        if "股票代號" in self.df.columns:
            df_copy = deepcopy(self.df[self.df["股票代號"] == ticker])

        if "ticker" in self.df.columns:
            df_copy = deepcopy(self.df[self.df["ticker"] == ticker])
        return CDF(df_copy)

    def not_na(self, column_name):
        df_copy = deepcopy(self.df[self.df[column_name].notna()])
        return CDF(df_copy)

    def __getattr__(self, attr):
        return getattr(self.df, attr)
