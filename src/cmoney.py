import subprocess
import os
import pandas as pd
import chardet
from typing import List
from datetime import datetime
import time
import concurrent.futures


# subprocess.call([r"../cmoney/price_19810101_19901231.bat"])
def get_cmoney_folder_path() -> str:
    current_path = os.getcwd().replace("\\", r"/")
    return f"{current_path}/src/cmoney"


def create_cmoney_bat(
    sql_query: str = None,
    table_name: str = None,
    start_date: str = None,
    end_date: str = None,
) -> None:
    cmoney_path = get_cmoney_folder_path()
    adj_table_name = table_name.replace("(", "_").replace(")", "")

    # if table_name and start_date and end_date is all None, then bat_file_name will be cmoney.bat
    if table_name and start_date and end_date is not None:
        bat_file_name = f"{adj_table_name}_{start_date}_{end_date}.bat"
        data_file_name = f"{adj_table_name}_{start_date}_{end_date}.txt"
    else:
        bat_file_name = "cmoney.bat"
        data_file_name = "cmoney_data.txt"

    # create a bat folder in cmoney folder if not exist
    if not os.path.exists(f"{cmoney_path}/bat"):
        os.makedirs(f"{cmoney_path}/bat")

    # create a folder by table_name in bat if not exist
    if not os.path.exists(f"{cmoney_path}/bat/{adj_table_name}"):
        os.makedirs(f"{cmoney_path}/bat/{adj_table_name}")

    # create a data folder in cmoney folder if not exist
    if not os.path.exists(f"{cmoney_path}/data"):
        os.makedirs(f"{cmoney_path}/data")

    # create a folder by table_name in data if not exist
    if not os.path.exists(f"{cmoney_path}/data/{adj_table_name}"):
        os.makedirs(f"{cmoney_path}/data/{adj_table_name}")

    with open(f"{cmoney_path}/bat/{adj_table_name}/{bat_file_name}", "w") as bat:
        if sql_query is not None:  # 依照需要欄位傳入sql query
            write_str = f'cd src \n cd cmoney \n  "CMTrans.exe" "SQL1; {sql_query};,;{cmoney_path}/data/{adj_table_name}/{data_file_name}"'

        if sql_query is None:  # sql_query 不傳入任何值，測試用
            write_str = (
                f'cd src \n cd cmoney \n "CMTrans.exe" "SQL1; select [日期], [股票代號], [股票名稱], [收盤價] from [日收盤表排行] where ([日期] between'
                + " '20200101' and '20200130' "
                + ') and ([股票代號] in <CM代號,X1> or [股票代號] in <CM代號,X2> or [股票代號] in <CM代號,1> or [股票代號] in <CM代號,2>) order by [日期], [股票代號] asc  ;,;cmoney_data.txt"'
            )

        bat.write(write_str)  # 使用 utf-8 寫入，方便後續閱讀

    # return the bat file path
    return f"{cmoney_path}/bat/{adj_table_name}/{bat_file_name}"


def create_cmoney_query(
    col_names: List[str] = None,
    table_name: str = None,
    top_5: bool = False,
    start_date: str = None,
    end_date: str = None,
):
    """
    回傳Cmoney SQL Query
    Args:
        col_names (list[str], optional): ex: ["日期", "股票代號", "股票名稱", "開盤價"]. Defaults to None 抓取所有欄位
        table_name (str, optional): "日收盤表排行". Defaults to None.
        top_5 (bool, optional): 是否僅回傳前五ROW, 供測試用. Defaults to None.
        start_date (str, optional): Start date for filtering records in the format 'YYYYMMDD'. Defaults to None.
        end_date (str, optional): End date for filtering records in the format 'YYYYMMDD'. Defaults to None.
    """

    if col_names is not None:
        col_names_str = ",".join(
            [f"[{col}]" for col in col_names]
        )  # "[col_A], [col_B], [col_C]"
    else:
        col_names_str = "*"

    cmoney_query = f"SELECT {col_names_str} FROM [{table_name}]"

    if top_5 == True:
        cmoney_query = f"SELECT TOP 5 {col_names_str} FROM [{table_name}]"

    where_conditions = []
    if start_date is not None:
        where_conditions.append(f"[日期] >= '{start_date}'")
    if end_date is not None:
        where_conditions.append(f"[日期] <= '{end_date}'")

    where_conditions.append("LEN([股票代號]) = 4")
    where_conditions.append("CHARINDEX('T', [股票代號]) = 0")

    if where_conditions:
        cmoney_query += " WHERE " + " AND ".join(where_conditions)

    cmoney_query

    return cmoney_query


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        result = chardet.detect(file.read())
    return result


def read_cmoney_data_txt(table_name: str = None) -> pd.DataFrame:
    cmoney_path = get_cmoney_folder_path()

    file_folder = f"{cmoney_path}/data/{table_name}"

    # get a list of all txt file in the folder
    file_list = [f for f in os.listdir(file_folder) if f.endswith(".txt")]

    # prepare a empty dataframe
    df = pd.DataFrame()

    # iterate through the list
    for file in file_list:
        # create the full input path and read the file
        file_path = os.path.join(file_folder, file)
        # read the file into a dataframe
        df_file = pd.read_csv(file_path, sep=",", encoding="ansi")
        # concatenate the dataframes
        df = pd.concat([df, df_file], axis=0)

    if "日期" in df.columns:
        df["日期"] = df["日期"].astype("str")

    if "股票代號" in df.columns:
        df["股票代號"] = df["股票代號"].astype("str")

    return df


#  return date by given start_date to end_date, and frequency, ex: daily, weekly, monthly, quarterly, yearly
def get_date_range(
    start_date: str = None, end_date: str = None, freq: str = None
) -> List[str]:
    """回傳日期區間, 預設為1980-01-01至今日, 頻率為日, 可自訂起始日期, 結束日期, 頻率, ex: D, W, M, Q, Y, 回傳格式為YYYYMMDD, 且為頻率的最後一天

    Args:
        start_date (str, optional): 起始日期, 格式為YYYYMMDD. Defaults to 19800101.
        end_date (str, optional): 結束日期, 格式為YYYYMMDD. Defaults to 今日.
        freq (str, optional): 頻率, ex: D, W, M, Q, Y or daily, weekly, monthly, quarterly, yearly. Defaults to D.

    Returns:
        List[str]: _description_
    """

    # if start_date is None, set start_date to 1980-01-01
    if start_date is None:
        start_date = "19800101"
    # if end_date is None, set end_date to today
    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")
    # if freq is None, set freq to daily
    if freq is None:
        freq = "D"

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    date_range = [date.strftime("%Y%m%d") for date in date_range]

    # add end_date to the date range if the last element is not end_date
    if date_range[-1] != end_date:
        date_range.append(end_date)

    return date_range


# define a function to process a date subset
def process_date_subset(date_subset, table_name):
    # iterate through the date range, set start_date as n and end_date as n+1
    for i in range(len(date_subset)):
        # if i is not equal to the len of date range, set start_date as n and end_date as n+1

        start_date = date_subset[i]
        end_date = date_subset[i + 1]

        # if end_date is the next day of start_date, set end_date as start_date
        if (
            datetime.strptime(end_date, "%Y%m%d")
            - datetime.strptime(start_date, "%Y%m%d")
        ).days == 1:
            end_date = start_date

        # create a cmoney query
        cmoney_query = create_cmoney_query(
            table_name=table_name, start_date=start_date, end_date=end_date
        )
        # create a bat file for cmoney
        create_cmoney_bat(cmoney_query, table_name, start_date, end_date)


def custom_split_date_range(date_range, num_splits):
    if num_splits >= len(date_range):
        return [[date] for date in date_range]

    avg_subset_size = len(date_range) / num_splits
    date_subsets = []
    idx = 0
    for _ in range(num_splits):
        start_idx = idx
        end_idx = idx + round(avg_subset_size)
        end_idx = min(end_idx, len(date_range))
        date_subsets.append(date_range[start_idx:end_idx])
        idx = end_idx

    if date_subsets[-1][-1] != date_range[-1]:
        date_subsets[-1].append(date_range[-1])

    # Ensure that all date_subsets have at least one element
    while any(len(subset) == 0 for subset in date_subsets):
        for i in range(len(date_subsets) - 1):
            if len(date_subsets[i + 1]) == 0:
                date_subsets[i + 1].append(date_subsets[i].pop())

    return date_subsets


def get_CMoney_data(
    table_name: str,
    start_date: str = None,
    end_date: str = None,
    freq: str = None,
    replace=True,
    run_bat=True,
) -> None:
    cmoney_path = get_cmoney_folder_path()
    # get today's date
    today = datetime.today().strftime("%Y%m%d")

    # if start_date today:
    if start_date == today:
        # create a cmoney query
        end_date = start_date
        cmoney_query = create_cmoney_query(
            table_name=table_name, start_date=start_date, end_date=end_date
        )
        # create a bat file for cmoney
        bat_file_path = create_cmoney_bat(
            cmoney_query, table_name, start_date, end_date
        )
        start_time = time.time()
        subprocess.call(
            [bat_file_path],
            stdout=subprocess.DEVNULL,
        )
        end_time = time.time()
        print(f"{table_name}下載完成, 花費時間: {round(end_time - start_time, 0)}秒")
    else:
        # if start_date, end_date, freq is not None, get date range
        if start_date is not None or end_date is not None or freq is not None:
            date_range = get_date_range(
                start_date=start_date, end_date=end_date, freq=freq
            )

            print(date_range)

            # define the number of threads to use
            num_threads = 6

            if len(date_range) < num_threads:
                num_threads = len(date_range)

            date_subsets = custom_split_date_range(date_range, num_threads)
            print("date_subsets", date_subsets)
            # create a thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # submit each date subset to a separate thread
                futures = [
                    executor.submit(process_date_subset, subset, table_name)
                    for subset in date_subsets
                ]
                # wait for all threads to complete
                concurrent.futures.wait(futures)

            # calculate the time to run the loop
            start_time = time.time()
            adj_table_name = table_name.replace("(", "_").replace(")", "")

            # get a list of all bat file in the folder
            bat_folder = f"{cmoney_path}/bat/{adj_table_name}"
            bat_list = [f for f in os.listdir(bat_folder) if f.endswith(".bat")]

            # get a list of all txt file in the data folder
            data_folder = f"{cmoney_path}/data/{adj_table_name}"
            txt_list = [f for f in os.listdir(data_folder) if f.endswith(".txt")]

            if replace == False:
                # create a set of bat file names
                bat_set = set([bat.split(".")[0] for bat in bat_list])
                # create a set of txt file names
                txt_set = set([txt.split(".")[0] for txt in txt_list])
                # get the difference between the two sets
                bat_list = list(bat_set - txt_set).sort()
                # add .bat to the file names if bat_list is not empty
                if bat_list:
                    bat_list = [bat + ".bat" for bat in bat_list]

            if run_bat:
                if bat_list:
                    # create a thread pool
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # submit each bat file to a separate thread
                        futures = [
                            executor.submit(
                                subprocess.call,
                                [os.path.join(bat_folder, bat)],
                                stdout=subprocess.DEVNULL,
                            )
                            for bat in bat_list
                        ]
                        # wait for all threads to complete
                        concurrent.futures.wait(futures)
                else:
                    print("沒有bat檔案需要執行")
            else:
                if bat_list:
                    print(f"共有{len(bat_list)}個bat檔案需要執行: {bat_list}")
                else:
                    print("沒有bat檔案需要執行")

            end_time = time.time()

            print(f"{table_name}下載完成, 花費時間: {round(end_time - start_time, 0)}秒")

        else:
            cmoney_query = create_cmoney_query(table_name=table_name)
            bat_file_path = create_cmoney_bat(cmoney_query, table_name)
            start_time = time.time()

            subprocess.call(
                [bat_file_path],
                stdout=subprocess.DEVNULL,
            )
            end_time = time.time()
            print(f"{table_name}下載完成, 花費時間: {round(end_time - start_time, 0)}秒")


if __name__ == "__main__":
    current_path = os.getcwd().replace("\\", r"/")
    # print(current_path)
    # print([f"{current_path}/src/cmoney/cmoney.bat"])

    #  following code is for testing download data from cmoney
    # cmoney_query = create_cmoney_query(table_name="股利政策表")
    # create_cmoney_bat(cmoney_query)
    # subprocess.call([f"{current_path}/src/cmoney/cmoney_test.bat"])
    # df = read_cmoney_data_txt()
    # df.to_feather("./data/股利政策表.feather")
    # df.to_excel("./data/股利政策表.xlsx")

    # get_CMoney_data(table_name="日收盤還原表排行", start_date="20000101", freq="Y")
    # get_CMoney_data(table_name="日收盤表排行", start_date="19900101", freq="Y")

    # get_CMoney_data(table_name="股利政策表")
    # get_CMoney_data(table_name="季股利政策表")
    # get_CMoney_data(table_name="季IFRS財報(資產負債)")
    # get_CMoney_data(table_name="季IFRS財報(損益累計)")
    # get_CMoney_data(table_name="季IFRS財報(損益單季)")
    # get_CMoney_data(table_name="季IFRS財報(現金流量單季)")
    # get_CMoney_data(table_name="上市櫃公司基本資料")
    get_CMoney_data(table_name="下市櫃公司基本資料")
    # get_CMoney_data(table_name="月董監股權與設質統計表")
    print("done")
