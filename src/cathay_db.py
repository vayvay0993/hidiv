import datetime
import time

import numpy as np
import pandas as pd
import pyodbc
from sqlalchemy import create_engine, text as sql_text
from urllib.parse import quote_plus


def commit(
    sql_query: str,
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
):
    """Execute a SQL query on a specified database and commit the changes.

    This function connects to a database using provided credentials, executes a
    SQL query, commits changes if any, and closes the connection. The function
    uses pyodbc for establishing the connection.

    Args:
        sql_query (str): The SQL query to be executed on the database. ex: "SELECT * FROM test_table3;"
        server (str): The server name where the database is located. ex: "10.198.213.13"
        database (str): The name of the database on which the query is to be executed. ex: "FinanceOther"
        server_uid (str): The user ID to connect to the server. ex: "ap00100809"
        server_pwd (str): The password associated with the user ID to connect to the server. ex: "password"

    Returns:
        int: The number of rows affected by the query execution. If no rows are affected, it returns -1.
    """
    conn = pyodbc_connection(server, database, server_uid, server_pwd)
    cursor = conn.cursor()
    rowcount = cursor.execute(sql_query).rowcount
    conn.commit()
    conn.close()
    return rowcount


def pyodbc_connection(server, database, server_uid, server_pwd):
    conn_str = f"DRIVER={'SQL Server'};SERVER={server};DATABASE={database};UID={server_uid};PWD={server_pwd};"
    return pyodbc.connect(conn_str)


def get_column_names(
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
    table_name: str,
):
    conn = pyodbc_connection(server, database, server_uid, server_pwd)

    cursor = conn.cursor()
    result = cursor.execute(f"SELECT TOP 5 * FROM {table_name}")
    col_names = list(map(lambda x: x[0], result.description))
    conn.close()
    return col_names


def pd_read_mssql_data(
    sql_query: str,
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
):
    """
    Reads data from a MSSQL database using a SQL query and returns a pandas DataFrame.

    This function establishes a connection to a MSSQL database using provided credentials,
    executes a SQL query, fetches the data, and then returns the results as a pandas DataFrame.
    It uses pyodbc and SQLAlchemy to establish the connection and pandas to read the SQL query
    and create the DataFrame.

    Args:
        sql_query (str): The SQL query to be executed on the database. ex: "SELECT * FROM test_table3;"
        server (str): The server name where the database is located. ex: "10.198.213.13"
        database (str): The name of the database on which the query is to be executed. ex: "FinanceOther"
        server_uid (str): The user ID to connect to the server. ex: "ap00100809"
        server_pwd (str): The password associated with the user ID to connect to the server. ex: "password"

    Returns:
        pandas.DataFrame: A DataFrame containing the results of the SQL query.

    """
    conn = f"DRIVER={'SQL Server'};SERVER={server};DATABASE={database};UID={server_uid};PWD={server_pwd}"
    quoted = quote_plus(conn)
    new_con = f"mssql+pyodbc:///?odbc_connect={quoted}"
    engine = create_engine(new_con)
    df = pd.read_sql(sql=sql_text(sql_query), con=engine.connect())

    return df


def pd_write_mssql(
    df: pd.DataFrame,
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
    table_name: str,
):
    conn = f"DRIVER={'SQL Server'};SERVER={server};DATABASE={database};UID={server_uid};PWD={server_pwd}"
    quoted = quote_plus(conn)
    new_con = f"mssql+pyodbc:///?odbc_connect={quoted}"
    engine = create_engine(new_con, fast_executemany=True)

    df.to_sql(table_name, engine, if_exists="append", index=False)


def create_temp_table(
    server: str, database: str, server_uid: str, server_pwd: str, table_name: str
):
    conn = pyodbc_connection(server, database, server_uid, server_pwd)

    cursor = conn.cursor()
    sql_query = f"SELECT * INTO {table_name}_TEMP FROM {table_name} WHERE 1=2;"
    cursor.execute(sql_query)
    conn.commit()
    conn.close()
    return True


def drop_table_if_exists(
    server: str, database: str, server_uid: str, server_pwd: str, table_name: str
):
    # establish connection to the server and database
    conn = pyodbc_connection(server, database, server_uid, server_pwd)

    # create cursor object to execute SQL statements
    cursor = conn.cursor()

    # check if table exists
    if cursor.tables(table=table_name, tableType="TABLE").fetchone():
        # if table exists, drop it
        drop_table_query = f"DROP TABLE {table_name}"
        cursor.execute(drop_table_query)
        conn.commit()
        # close the connection
        conn.close()

        return True
        print(f"Table '{table_name}' dropped successfully.")
    else:
        conn.close()
        return False
        print(f"Table '{table_name}' does not exist.")


def update_on_dup_key_insert(
    df: pd.DataFrame,
    server: str,
    database: str,
    server_uid: str,
    server_pwd: str,
    table_name: str,
):
    """
    Updates the specified table on the database with the provided DataFrame, handling duplicate keys.

    This function connects to a database using provided credentials, creates a temporary table
    in the database, inserts the DataFrame into the temporary table, and then updates the
    specified table with data from the temporary table, handling duplicate keys. If a row with
    the same key exists in the table, it updates the row with data from the DataFrame. If a row
    with the key does not exist, it inserts the row from the DataFrame. Finally, it drops the
    temporary table and closes the connection.

    Args:
        df (pandas.DataFrame): The DataFrame to be inserted into the table.
        server (str): The server name where the database is located. ex: "10.198.213.13"
        database (str): The name of the database on which the operation is to be performed. ex: "FinanceOther"
        server_uid (str): The user ID to connect to the server. ex: "ap00100809"
        server_pwd (str): The password associated with the user ID to connect to the server. ex: "password"
        table_name (str): The name of the table to be updated. ex: "test_table3"

    Returns:
        bool: True if the operation is successful, otherwise an exception will be raised.
    """

    print("-" * 100)
    # establish connection to the server and database
    conn = pyodbc_connection(server, database, server_uid, server_pwd)

    # create cursor object to execute SQL statements
    cursor = conn.cursor()

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

    create_temp_table(server, database, server_uid, server_pwd, table_name)

    pd_write_mssql(df, server, database, server_uid, server_pwd, table_temp_name)

    # update on duplicate key and insert data
    cursor = conn.cursor()
    pk_list = [x[8] for x in cursor.statistics(table_name) if x[4] == table_name]
    key_list = [x[3] for x in cursor.columns(table_name)]
    merge_query = (
        f"""
    MERGE {table_name} as t
    USING {table_temp_name} as c
    ON """
        + " AND ".join([f"(t.[{pk}] = c.[{pk}])" for pk in pk_list])
        + f"""
    WHEN MATCHED THEN 
    UPDATE SET
    """
        + ",\n".join([f"[{k}]=c.[{k}]" for k in key_list])
        + """
    WHEN NOT MATCHED THEN 
    INSERT ("""
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

    return True


if __name__ == "__main__":
    db_settings = {
        "sql_query": "SELECT * FROM test_table4;",
        "server": "10.198.213.13",
        "database": "FinanceOther",
        "server_uid": "ap00100809",
        "server_pwd": "Cathay06",
        # "table_name": "test_table4",
        # "local": False,
    }

    df = pd.DataFrame()
    df["A"] = pd.date_range("2020-12-01", periods=1000, freq="D")
    # convert A column to string format YYYY-MM-DD
    df["A"] = df["A"].dt.strftime("%Y/%m/%d")
    df["B"] = np.arange(1000)
    df["C"] = np.random.randint(0, 1000, 1000)
    df["D"] = np.random.randint(0, 1000, 1000) + np.random.randint(0, 1000, 1000) / 100
    df["E"] = pd.date_range("2021-01-01", periods=1000, freq="D")

    # time the execution
    start_time = time.time()
    # update_on_dup_key_insert(df, **db_settings)
    print(pd_read_mssql_data(**db_settings))

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds")
