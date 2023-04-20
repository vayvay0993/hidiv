from src.cmoney import get_CMoney_data
import subprocess
import os
import pandas as pd

import threading

import time


# Define the function to be run in each thread
def call_get_CMoney_data(table_name):
    get_CMoney_data(table_name=table_name)


# Create a list of table names
table_names = [
    "股利政策表",
    "季股利政策表",
    "季IFRS財報(資產負債)",
    "季IFRS財報(損益累計)",
    "季IFRS財報(損益單季)",
    "季IFRS財報(現金流量單季)",
    "上市櫃公司基本資料",
    "下市櫃公司基本資料",
    "月董監股權與設質統計表",
]

# start_time = time.time()
# Create and start the threads
# threads = []
# for table_name in table_names:
#     thread = threading.Thread(target=call_get_CMoney_data, args=(table_name,))
#     thread.start()
#     threads.append(thread)

# # Wait for all threads to finish
# for thread in threads:
#     thread.join()
# end_time = time.time()
# print(f"花費時間: {round(end_time - start_time, 0)}秒")


if __name__ == "__main__":
    # get_CMoney_data(table_name="日收盤表排行", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="日收盤還原表排行", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="日收盤還原表排行(還原分紅)", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="個股券商分點進出明細", start_date="20200101", freq="D")
    # get_CMoney_data(table_name="日個股事件", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="集保庫存表", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="個股機構績效評等", start_date="19900101", freq="Y")
    print("done")
