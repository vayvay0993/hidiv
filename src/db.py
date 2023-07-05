import pyodbc
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text as sql_text
from datetime import datetime

# import List from type for type hinting
from typing import List

import concurrent.futures
from typing import Tuple
import time


def commit(
    sql_query: str,
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
    local: bool = False,
):
    conn = pyodbc_connection(server, database, server_uid, server_pwd)

    con = pyodbc.connect(conn)
    cursor = conn.cursor()
    rowcount = cursor.execute(sql_query).rowcount
    conn.commit()
    conn.close()
    return rowcount


def pyodbc_connection(server, database, server_uid, server_pwd):
    conn_str = f"DRIVER={'SQL Server'};SERVER={server};DATABASE={database};UID={server_uid};PWD={server_pwd};"
    return pyodbc.connect(conn_str)


def get_one(sql_query, server, database, server_uid, server_pwd):
    con = pyodbc_connection(server, database, server_uid, server_pwd)
    cursor = conn.cursor()
    result = cursor.execute(sql_query).fetchone()
    conn.close()
    return result


def get_column_names(
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
    table: str,
    local: bool = False,
) -> List[str]:
    if local:
        conn_str = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"

    else:
        conn_str = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    result = cursor.execute(f"SELECT TOP 5 * FROM {table}")
    col_names = list(map(lambda x: x[0], result.description))
    conn.close()
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
#     conn.close()
#     return df


def pd_read_mssql_data(
    sql_query: str,
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
    local=False,
) -> pd.DataFrame:
    if local == True:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"
        con = pyodbc.connect(conn)
        df = pd.read_sql_query(sql_query, con)
        conn.close()
        return df

    if local == False:
        conn = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"
        quoted = quote_plus(conn)
        new_con = "mssql+pyodbc:///?odbc_connect={}".format(quoted)
        engine = create_engine(new_con)
        df = pd.read_sql(sql=sql_text(sql_query), con=engine.connect())

        return df


# def pd_write_mssql(
#     df: pd.DataFrame,
#     server: str,
#     database: str,
#     server_uid: str,
#     server_pwd: str,
#     table_name: str,
#     local: bool = False,
# ) -> None:
#     if local:
#         conn_str = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"

#     else:
#         conn_str = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"

#     create_table_if_not_exists(df, server, database, server_uid, server_pwd, table_name)

#     # establish connection to the server and database
#     # conn_str = f"DRIVER={'SQL Server'};SERVER={server};DATABASE={database};UID={server_uid};PWD={server_pwd};"
#     conn = pyodbc_connection(server, database, server_uid, server_pwd)

#     # create cursor object to execute SQL statements
#     cursor = conn.cursor()

#     col_names = get_column_names(
#         server, database, server_uid, server_pwd, table_name, local
#     )  # ["col_A", "col_B", "col_C"]
#     col_names_str = ",".join(
#         [f"[{col}]" for col in col_names]
#     )  # "[col_A], [col_B], [col_C]"

#     def row_to_value_str(row):
#         values_str = ",".join([f"'{row[col]}'" for col in col_names])
#         return values_str

#     for index, row in df.iterrows():
#         cursor.execute(
#             f"INSERT INTO dbo.{table_name}({col_names_str}) values ({row_to_value_str(row)})"
#         )
#         conn.commit()

#     conn.close()


def pd_write_mssql(
    df: pd.DataFrame,
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
    table_name: str,
    local: bool = False,
    max_workers: int = 6,
) -> None:
    if local:
        conn_str = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}"

    else:
        conn_str = f"DRIVER={'SQL Server'}; SERVER={server}; DATABASE={database}; UID={server_uid}; PWD={server_pwd}"

    create_table_if_not_exists(df, server, database, server_uid, server_pwd, table_name)

    col_names = get_column_names(
        server, database, server_uid, server_pwd, table_name, local
    )
    col_names_str = ",".join([f"[{col}]" for col in col_names])

    def row_to_value_str(row):
        values_str = ",".join([f"'{row[col]}'" for col in col_names])
        return values_str

    def insert_row(row_data: Tuple[int, pd.Series]) -> None:
        index, row = row_data
        conn = pyodbc_connection(server, database, server_uid, server_pwd)
        with conn.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO dbo.{table_name}({col_names_str}) values ({row_to_value_str(row)})"
            )
            conn.commit()
        conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(insert_row, df.iterrows())


def create_table_if_not_exists(
    df, server, database, server_uid, server_pwd, table_name
):
    # establish connection to the server and database
    conn_str = f"DRIVER={'SQL Server'};SERVER={server};DATABASE={database};UID={server_uid};PWD={server_pwd};"
    conn = pyodbc.connect(conn_str)

    # create cursor object to execute SQL statements
    cursor = conn.cursor()

    # check if table exists
    if not cursor.tables(table=table_name, tableType="TABLE").fetchone():
        # if table does not exist, create it
        column_names = list(df.columns)
        column_definitions = ", ".join(
            [f"{col_name} NVARCHAR(MAX)" for col_name in column_names]
        )
        create_table_query = f"CREATE TABLE {table_name} ({column_definitions})"
        cursor.execute(create_table_query)
        conn.commit()
        print(f"Table '{table_name}' created successfully.")
    else:
        print(f"Table '{table_name}' already exists.")

    # close the connection
    conn.close()


def drop_table_if_exists(server, database, server_uid, server_pwd, table_name):
    # establish connection to the server and database
    conn_str = f"DRIVER={'SQL Server'};SERVER={server};DATABASE={database};UID={server_uid};PWD={server_pwd};"
    conn = pyodbc.connect(conn_str)

    # create cursor object to execute SQL statements
    cursor = conn.cursor()

    # check if table exists
    if cursor.tables(table=table_name, tableType="TABLE").fetchone():
        # if table exists, drop it
        drop_table_query = f"DROP TABLE {table_name}"
        cursor.execute(drop_table_query)
        conn.commit()
        print(f"Table '{table_name}' dropped successfully.")
    else:
        print(f"Table '{table_name}' does not exist.")

    # close the connection
    conn.close()


def update_on_dup_key_insert(
    df,
    server,
    database,
    server_uid,
    server_pwd,
    table_name,
    max_workers: int = 5,
):
    print("-" * 100)
    # establish connection to the server and database
    conn_str = f"DRIVER={'SQL Server'};SERVER={server};DATABASE={database};UID={server_uid};PWD={server_pwd};"
    conn = pyodbc.connect(conn_str)

    # create cursor object to execute SQL statements
    cursor = conn.cursor()

    try:
        cursor.execute(f"SELECT count(*) FROM {table_name};")
        print(f"更新表 {table_name} 前資料ROW數 : {cursor.fetchone()[0]}")
    except:
        create_table_if_not_exists(
            df, server, database, server_uid, server_pwd, table_name=table_name
        )
        cursor.execute(f"SELECT count(*) FROM {table_name};")
        print(f"更新表 {table_name} 前資料ROW數 : {cursor.fetchone()[0]}")

    # insert data into temp table
    table_temp_name = table_name + "_TEMP"
    drop_table_if_exists(
        server,
        database,
        server_uid,
        server_pwd,
        table_name=table_temp_name,
    )

    create_table_if_not_exists(
        df, server, database, server_uid, server_pwd, table_name=table_temp_name
    )

    pd_write_mssql(df, server, database, server_uid, server_pwd, table_temp_name)

    # update on duplicate key and insert data
    cursor = conn.cursor()
    print(f"table name : {table_name} primary key :")
    pk_list = [x[8] for x in cursor.statistics(table_name) if x[4] == table_name]
    print(pk_list)
    print(f"table name : {table_name} key :")
    key_list = [x[3] for x in cursor.columns(table_name)]
    print(key_list)

    merge_query = (
        f"""
    MERGE {table_name} as t
    USING {table_temp_name} as c
    ON """
        + " AND ".join([f"(t.[{pk}] = c.[{pk}])" for pk in pk_list])
        + f"""
    WHEN MATCHED
    THEN UPDATE SET
    """
        + ",\n".join([f"[{k}]=c.[{k}]" for k in key_list])
        + """
    WHEN NOT MATCHED
    THEN INSERT ("""
        + ", ".join([f"[{k}]" for k in key_list])
        + """)
    VALUES("""
        + ", ".join([f"c.[{k}]" for k in key_list])
        + """);
    """
    )

    cursor.execute(merge_query)
    cursor.commit()

    # check number of rows after update
    cursor = conn.cursor()
    cursor.execute(f"SELECT count(*) FROM {table_name};")
    print(f"更新表 {table_name} 後資料ROW數 : {cursor.fetchone()[0]}")

    # drop temp table
    drop_table_if_exists(
        server,
        database,
        server_uid,
        server_pwd,
        table_name=table_temp_name,
    )

    print("-" * 100)
    conn.close()


if __name__ == "__main__":
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
        "database": "FinanceOther",
        "server_uid": "ap00100690",
        "server_pwd": "Xavior0690",
        "table_name": "test_table",
        # "local": False,
    }

    # sql_query = "SELECT * FROM T012_DomesticDelistInfo"
    df = pd.DataFrame()
    df["A"] = [3, 4, 5, 6, 7, 8]
    df["B"] = [31, 41, 51, 61, 71, 81]
    # time the execution
    start_time = time.time()
    # pd_write_mssql(df, **db_settings)
    pd_write_mssql_fast(df, **db_settings)
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")
    # update_on_dup_key_insert(df, **db_settings)

    # create_table_if_not_exists(df, **db_settings)
