from src.cmoney import get_CMoney_data
import subprocess
import os
import pandas as pd

import threading

import time
import concurrent.futures


def process_txt_file(txt_file):
    # check if the feather file exists
    feather_file = txt_file.replace(".txt", ".feather")
    if feather_file not in os.listdir(os.path.join(feather_data_folder, folder)):
        # read the txt file
        df = pd.read_csv(
            os.path.join(txt_data_folder, folder, txt_file),
            encoding="ansi",
            dtype=str,
        )
        df = df.reset_index(drop=True)
        # save as feather file
        df.to_feather(os.path.join(feather_data_folder, folder, feather_file))


# Define the function to be run in each thread
def call_get_CMoney_data(table_name):
    get_CMoney_data(table_name=table_name)


# Create a list of table names
table_names = [
    # "股利政策表",
    # "季股利政策表",
    # "季IFRS財報(資產負債)",
    # "季IFRS財報(損益累計)",
    # "季IFRS財報(損益單季)",
    # "季IFRS財報(現金流量單季)",
    # "上市櫃公司基本資料",
    # "下市櫃公司基本資料",
    # "月董監股權與設質統計表",
    # "日收盤還原表排行(還原分紅)"
]

# --------------follow code is for turn txt to feather----------------

start_time = time.time()

# get the folder structure of src > cmoney > data
txt_data_folder = os.path.join(os.getcwd(), "src", "cmoney", "data")
# get the list of folder in data folder
txt_data_folder_list = os.listdir(txt_data_folder)
print(txt_data_folder_list)


feather_data_folder = os.path.join(os.getcwd(), "src", "cmoney", "data")
# create the same folder structure in data folder
for folder in txt_data_folder_list:
    # create the folder if it doesn't exist
    feather_data_folder = os.path.join(os.getcwd(), "data")
    if not os.path.exists(os.path.join(feather_data_folder, folder)):
        os.makedirs(os.path.join(feather_data_folder, folder))

    # read the list of txt files in the folder
    txt_file_list = os.listdir(os.path.join(txt_data_folder, folder))

    for txt_file in txt_file_list:
        process_txt_file(txt_file)

end_time = time.time()
print("Time taken: ", end_time - start_time)

# --------------upper code is for turn txt to feather----------------

if __name__ == "__main__":
    # get_CMoney_data(table_name="日收盤表排行", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="日收盤還原表排行", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="日收盤還原表排行(還原分紅)", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="個股券商分點進出明細", start_date="20200101", freq="D")
    # get_CMoney_data(table_name="日個股事件", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="集保庫存表", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="個股機構績效評等", start_date="19900101", freq="Y")
    # get_CMoney_data(table_name="季股利政策表")
    print("done")
