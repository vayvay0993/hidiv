import subprocess
import os
import pandas as pd

# subprocess.call([r"../cmoney/price_19810101_19901231.bat"])
def get_cmoney_path() -> str:
    current_path = os.getcwd().replace("\\",r"/")
    return f"{current_path}/src/cmoney"

def write_cmoney_bat(sql_query = None) -> None:
    cmoney_path = get_cmoney_path()
    with open(f"{cmoney_path}/cmoney_test.bat",'w') as bat:
        if sql_query is not None: # 依照需要欄位傳入sql query
            write_str = f'cd src \n cd cmoney \n  "CMTrans.exe" "SQL1; {sql_query};,;cmoney_data.txt"'

        if sql_query is None: # sql_query 不傳入任何值，測試用
            write_str = f'cd src \n cd cmoney \n "CMTrans.exe" "SQL1; select [日期], [股票代號], [股票名稱], [開盤價], [最高價], [最低價], [收盤價] from [日收盤表排行] where ([日期] between' + " '20200101' and '20200130' " + ') and ([股票代號] in <CM代號,X1> or [股票代號] in <CM代號,X2>  or [股票代號] in <CM代號,1> or [股票代號] in <CM代號,2>) order by [日期], [股票代號] asc  ;,;cmoney_data.txt"'

        bat.write(write_str) # 使用 utf-8 寫入，方便後續閱讀

def get_cmoney_query(col_names: list[str] = None, table_name: str = None, top_5: bool = False):
    """回傳Cmoney SQL Query
    Args:
        col_names (list[str], optional): ex: ["日期", "股票代號", "股票名稱", "開盤價"]. Defaults to None.
        table_name (str, optional): "日收盤表排行". Defaults to None.
        top_5 (bool, optional): 是否僅回傳前五ROW, 供測試用. Defaults to None.
    """
    if col_names is not None:
        col_names_str = ",".join([f"[{col}]" for col in col_names]) # "[col_A], [col_B], [col_C]"
    else:
        col_names_str = "*"

    if top_5 == False:
        cmoney_query = f"SELECT {col_names_str} FROM [{table_name}]"
    else:
        cmoney_query = f"SELECT TOP 5 {col_names_str} FROM [{table_name}]"

    return cmoney_query

def read_cmoney_data_txt() -> pd.DataFrame:
    cmoney_path = get_cmoney_path()
    df = pd.read_csv(f'{cmoney_path}/cmoney_data.txt', encoding='cp950')
    if '日期' in df.columns:
        df['日期'] = df['日期'].astype('str')

    if '股票代號' in df.columns:
        df['股票代號'] = df['股票代號'].astype('str')

    return df
    # with open(f'{cmoney_path}/cmoney_data.txt','rb')as f:

    #     result = f.read().decode('cp950')
    #     print(len(result))
    #     print(result[:5])

    

if __name__ == '__main__':
    current_path = os.getcwd().replace("\\",r"/")
    # print(current_path)
    # print([f"{current_path}/src/cmoney/cmoney.bat"])
    # write_cmoney_bat()
    cmoney_query = get_cmoney_query(["日期", "股票代號", "股票名稱", "開盤價"], table_name = "日收盤表排行", top_5= True)
    # print(cmoney_query)
    # write_cmoney_bat()
    # subprocess.call([f"{current_path}/src/cmoney/cmoney_test.bat"])
    df = read_cmoney_data_txt()
    print(df.head())
