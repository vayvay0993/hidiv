import pyodbc
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text as sql_text
from datetime import datetime


# from sqlalchemy.engine import URL
# from sqlalchemy import create_engine

def commit(sql_query: str, server: str, database: str, server_uid: str, server_pwd: str, local: bool=False):
    if local:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"
    else:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"

    con = pyodbc.connect(conn)
    cursor = con.cursor()
    rowcount = cursor.execute(sql_query).rowcount
    con.commit()
    con.close()
    return rowcount


def get_one(sql_query, server, database, server_uid, server_pwd, local=False):

    if local:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"
    else:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"

    con = pyodbc.connect(conn)
    cursor = con.cursor()
    result = cursor.execute(sql_query).fetchone()
    con.close()
    return result

def get_column_names(server: str, database: str, server_uid: str, server_pwd: str, table: str, local: bool =False) -> list[str]:
    
    if local:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"

    else:
        conn =  f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"

    con = pyodbc.connect(conn)
    cursor = con.cursor()
    result = cursor.execute(f'SELECT TOP 5 * FROM {table}')
    col_names = list(map(lambda x: x[0], result.description))
    con.close()
    return col_names

# Following function is deprecated due to fix the bug when SQLAlchemy 2.0.0 version is released, please use the new one instead
# def pd_read_mssql_data(sql_query, server, database, server_uid, server_pwd, local=False):
#     if local:
#         conn = 'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database
#     else:
#         conn = 'DRIVER={};SERVER={};DATABASE={};UID={};PWD={}'.format('SQL Server', server, database, server_uid,
#                                                                       server_pwd)
#     con = pyodbc.connect(conn)
#     df = pd.read_sql_query(sql_query, con)
#     con.close()
#     return df

def pd_read_mssql_data(sql_query: str, server: str, database: str, server_uid: str, server_pwd: str, local=False) -> pd.DataFrame:

    if local == True:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"
        con = pyodbc.connect(conn)
        df = pd.read_sql_query(sql_query, con)
        con.close()
        return df
    
    if local == False:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"
        quoted = quote_plus(conn)
        new_con = 'mssql+pyodbc:///?odbc_connect={}'.format(quoted)
        engine = create_engine(new_con)
        df = pd.read_sql(sql = sql_text(sql_query), con = engine.connect())
        
        return df

# Following function is deprecated 
# def pd_write_mssql(df: pd.DataFrame, server: str, database: str, server_uid: str, server_pwd: str, table: str, local: bool =False) -> pd.DataFrame:
#     if local:
#         conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"
#     else:
#         conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"
        
#     quoted = quote_plus(conn)
#     new_con = 'mssql+pyodbc:///?odbc_connect={}'.format(quoted)
#     engine = create_engine(new_con)

#     df.to_sql(table, con = engine, if_exists='append', index=False)


def pd_write_mssql(df: pd.DataFrame, server: str, database: str, server_uid: str, server_pwd: str, table: str, local: bool =False) -> None:
    if local:
        conn_str = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"

    else:
        conn_str =  f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"

    con = pyodbc.connect(conn_str)
    cursor = con.cursor()
    col_names = get_column_names(server, database, server_uid, server_pwd, table, local) # ["col_A", "col_B", "col_C"]
    col_names_str = ",".join([f"[{col}]" for col in col_names]) # "[col_A], [col_B], [col_C]"
    def row_to_value_str(row):
        values_str = ",".join([f"'{row[col]}'" for col in col_names])
        return values_str

    for index, row in df.iterrows():
        cursor.execute(f'INSERT INTO dbo.{table}({col_names_str}) values ({row_to_value_str(row)})')
        con.commit()

    cursor.close()
    con.close()

if __name__ == '__main__':

    # db_settings = {
    #     "server": "10.198.213.13",
    #     "database": "taihaotou",
    #     "server_uid": "ap00100690",
    #     "server_pwd": "Xavior0690",
    #     "local": False
    # }

    # sql_query = "SELECT * FROM T012_DomesticDelistInfo"
    # df = pd_read_mssql_data(sql_query, **db_settings)
    # print(df)

    db_settings = {
        "server": "10.198.213.13",
        "database": "ResearchRpt",
        "server_uid": "ap00100690",
        "server_pwd": "Xavior0690",
        "table": "test_table",
        "local": False
    }

    # sql_query = "SELECT * FROM T012_DomesticDelistInfo"
    df = pd.DataFrame()
    df['A'] = [3, 4, 5, 6, 7, 8]
    df['B'] = [31, 41, 51, 61, 71, 81]
    pd_write_mssql(df, **db_settings)
    