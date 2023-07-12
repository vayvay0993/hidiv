import pandas as pd
from copy import deepcopy
import json


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
