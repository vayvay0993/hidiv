import pandas as pd
from copy import deepcopy
import datetime


def count_quarter(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Assuming that quarters are defined by calendar quarters.
    # Start from 0, because if both dates are in the same quarter, the count should be 0.
    quarter_count = (
        (end_date.year - start_date.year) * 4 + end_date.quarter
    ) - start_date.quarter
    return quarter_count


def quarter_end_date(year_quarter: str):
    quarter_dict = {"1": "03", "2": "06", "3": "09", "4": "12"}
    return pd.Timestamp(
        year_quarter[:4] + "-" + quarter_dict[year_quarter[-1]] + "-01"
    ) + pd.offsets.QuarterEnd(0)


def last_announce_date(year_quarter: str, shift: int = 0):
    """Change the year_quarter to the last announce date of the financial statement

    Args:
        year_quarter (str): year_quarter in the format of YYYYQQ
        shift (int, optional): shift the date by the number of days. Defaults to 0.

    Returns:
        date: the last announce date of the financial statement
    """
    if year_quarter[-1] == "1":
        date = pd.Timestamp(year_quarter[:4] + "-05-15")
    elif year_quarter[-1] == "2":
        date = pd.Timestamp(year_quarter[:4] + "-08-14")
    elif year_quarter[-1] == "3":
        date = pd.Timestamp(year_quarter[:4] + "-11-14")
    elif year_quarter[-1] == "4":
        date = pd.Timestamp(str(int(year_quarter[:4]) + 1) + "-03-31")

    # if the date is a holiday, move it to the next business day
    if date.weekday() in [5, 6]:
        date = date + pd.offsets.BDay(1)

    # if shift is not 0, shift the date by the number of days
    if shift != 0:
        date = date + pd.offsets.BDay(shift)

    return date


def quarterly_to_daily(df_financial_statement, accounting_item, days_shift=1):
    """
    Args:
        df: financial statement data, should contain 股票代號, 年季, 建立日期, accounting item
        accounting_item: accounting item, ex: 資產總計(千)
    Returns:
        pd.DataFrame
    """

    # create a df to store the accounting_item
    df = deepcopy(df_financial_statement[["股票代號", "年季", "建立日期", accounting_item]])

    # drop na
    df.dropna(inplace=True)

    # drop the duplicate
    df.drop_duplicates(inplace=True)

    # change the accounting_item to float
    df[accounting_item] = df[accounting_item].astype(float)

    # get the last announce date of the financial statement
    df["last_announce_date"] = df["年季"].apply(
        lambda x: last_announce_date(x, days_shift)
    )

    # convert 建立日期 from YYYYMMDD to datetime format YYYY-MM-DD
    df["建立日期"] = pd.to_datetime(df["建立日期"], format="%Y%m%d")

    # calculate the data_valid_date, which is the earlier date of 建立日期 and last_announce_date
    df["data_valid_date"] = df[["建立日期", "last_announce_date"]].min(axis=1)

    # get 股票代號, data_valid_date, accounting_item
    df = deepcopy(df[["股票代號", "data_valid_date", accounting_item]])

    df.drop_duplicates(subset=["data_valid_date", "股票代號"], inplace=True)

    # sort value by 股票代號 and data_valid_date and reset index
    df = df.sort_values(by=["股票代號", "data_valid_date"])

    df = df.reset_index(drop=True)

    # pivot the table
    df = df.pivot(index="data_valid_date", columns="股票代號", values=accounting_item)

    # reindex the pivot table to daily data
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))

    # ffill the missing values
    df = df.fillna(method="ffill")

    # Melt the pivot table
    df = pd.melt(
        df.reset_index(), id_vars="index", var_name="股票代號", value_name=accounting_item
    )

    df.columns = ["date", "ticker", accounting_item]

    return df


def calc_periodic(df_financial_statement, column_name, period):
    df = deepcopy(df_financial_statement[["股票代號", "年季", "建立日期", f"{column_name}"]])

    df.dropna(inplace=True)

    df.drop_duplicates(inplace=True)

    df = df.sort_values(by=["股票代號", "年季"]).reset_index(drop=True)

    df[f"{column_name}"] = df[f"{column_name}"].astype(float)

    df["季底日期"] = df["年季"].apply(lambda x: quarter_end_date(x))

    # Set '建立日期' as the index of the dataframe
    df.set_index("季底日期", inplace=True)

    # Group by '股票代號' and resample, forward filling the missing values
    df_grouped = df.groupby("股票代號")[column_name].resample("Q").ffill().reset_index()

    df.reset_index(inplace=True)
    df = deepcopy(
        df_grouped.merge(df[["股票代號", "季底日期", "建立日期"]], how="left", on=["股票代號", "季底日期"])
    )
    # turn the '季底日期' to '年季'
    df["年季"] = df["季底日期"].apply(
        lambda x: x.strftime("%Y") + str(int(x.strftime("%m")) // 3).zfill(2)
    )

    # df["建立日期"] = df.groupby("股票代號")["建立日期"].bfill()

    # create a column to store the year_season_after: if year_season = 202001, then year_season_q_ago = 202002 or 202101
    if period == "qoq":
        df["year_season_after"] = df["年季"].apply(
            lambda x: str(int(x) + 1)
            if int(str(x)[-2:]) < 4
            else str(int(str(x)[:4]) + 1) + "01"
        )
    elif period == "yoy":
        df["year_season_after"] = df["年季"].apply(
            lambda x: str(int(x[:4]) + 1) + str(x[4:])
        )

    df = df.merge(
        df[["股票代號", "year_season_after", f"{column_name}"]],
        how="left",
        left_on=["股票代號", "年季"],
        right_on=["股票代號", "year_season_after"],
    )

    df[f"{column_name}_{period}"] = df[f"{column_name}_x"] / df[f"{column_name}_y"]

    df = df[["股票代號", "年季", "建立日期", f"{column_name}_{period}"]]

    return df


def flow_item_to_quarter_data(df_financial_statement, accounting_item: str, clean=True):
    df = df_financial_statement[["年季", "股票代號", "建立日期", accounting_item]].copy()

    # drop na
    df.dropna(inplace=True)

    # drop the duplicate
    df.drop_duplicates(inplace=True)

    df = df.sort_values(by=["股票代號", "年季"]).reset_index(drop=True)

    df[f"{accounting_item}"] = df[f"{accounting_item}"].astype(float)

    df["季底日期"] = df["年季"].apply(lambda x: quarter_end_date(x))

    df["季底日期_last"] = df.groupby("股票代號", as_index=False, group_keys=False)[
        "季底日期"
    ].shift(1)
    df["季底日期_next"] = df.groupby("股票代號", as_index=False, group_keys=False)[
        "季底日期"
    ].shift(-1)

    # calculate the date difference between '年季_last' and '年季_next' with '年季_dt'
    df["季底日期_next_dt"] = df["季底日期_next"] - df["季底日期"]
    df["季底日期_dt_last"] = df["季底日期"] - df["季底日期_last"]

    df["季底日期_max"] = df[["季底日期_dt_last", "季底日期_next_dt"]].max(axis=1)
    df["季底日期_max"] = df["季底日期_max"].apply(lambda x: x.days)

    # if 季底日期_max is smaller than 100, then it is a quarterly data, if greater than 200, then it is a yearly data, if between 100 and 200, then it is a half yearly data
    df["財報類別"] = df["季底日期_max"].apply(lambda x: 1 if x < 100 else 4 if x > 200 else 2)
    df.set_index("季底日期", inplace=True)

    df_grouped = df.groupby("股票代號")[accounting_item].resample("Q").mean().reset_index()
    df.reset_index(inplace=True)
    df = df_grouped.merge(
        df[["股票代號", "季底日期", "財報類別", "建立日期"]], how="left", on=["股票代號", "季底日期"]
    ).copy()

    # turn the '季底日期' to '年季'
    df["年季"] = df["季底日期"].apply(
        lambda x: x.strftime("%Y") + str(int(x.strftime("%m")) // 3).zfill(2)
    )

    # df["建立日期"] = df.groupby("股票代號")["建立日期"].bfill()
    df.fillna(0, inplace=True)

    df[f"{accounting_item}_4_row_sum"] = df.groupby(
        "股票代號", as_index=False, group_keys=False
    ).apply(lambda x: x[accounting_item].rolling(4).sum())

    df[f"number_of_quarter_sum"] = df.groupby(
        "股票代號", as_index=False, group_keys=False
    ).apply(lambda x: x["財報類別"].rolling(4).sum())

    df[f"{accounting_item}_4q_sum"] = (
        df[f"{accounting_item}_4_row_sum"] / df[f"number_of_quarter_sum"] * 4
    )

    df["last_announce_date"] = df["年季"].apply(lambda x: last_announce_date(x, 1))
    # calculate the data_valid_date, which is the earlier date of 建立日期 and last_announce_date
    # replace the 0 in 建立日期 with 2999-12-31
    df["建立日期"] = df["建立日期"].apply(
        lambda x: datetime.datetime.strptime("2999-12-31", "%Y-%m-%d") if x == 0 else x
    )
    df["data_valid_date"] = df[["建立日期", "last_announce_date"]].min(axis=1)
    # replace the 建立日期 with the data_valid_date
    df["建立日期"] = df["data_valid_date"]

    if clean == True:
        df = df[["股票代號", "年季", "建立日期", f"{accounting_item}_4q_sum"]]
    return df


def flow_item_to_single_quarter_data(
    df_financial_statement, accounting_item: str, clean=True
):
    df = df_financial_statement[["年季", "股票代號", "建立日期", accounting_item]].copy()

    # drop na
    df.dropna(inplace=True)

    # drop the duplicate
    df.drop_duplicates(inplace=True)

    df = df.sort_values(by=["股票代號", "年季"]).reset_index(drop=True)

    df[f"{accounting_item}"] = df[f"{accounting_item}"].astype(float)

    df["季底日期"] = df["年季"].apply(lambda x: quarter_end_date(x))

    df["季底日期_last"] = df.groupby("股票代號", as_index=False, group_keys=False)[
        "季底日期"
    ].shift(1)
    df["季底日期_next"] = df.groupby("股票代號", as_index=False, group_keys=False)[
        "季底日期"
    ].shift(-1)

    # calculate the date difference between '年季_last' and '年季_next' with '年季_dt'
    df["季底日期_next_dt"] = df["季底日期_next"] - df["季底日期"]
    df["季底日期_dt_last"] = df["季底日期"] - df["季底日期_last"]

    df["季底日期_max"] = df[["季底日期_dt_last", "季底日期_next_dt"]].max(axis=1)
    df["季底日期_max"] = df["季底日期_max"].apply(lambda x: x.days)

    # if 季底日期_max is smaller than 100, then it is a quarterly data, if greater than 200, then it is a yearly data, if between 100 and 200, then it is a half yearly data
    df["財報類別"] = df["季底日期_max"].apply(lambda x: 1 if x < 100 else 4 if x > 200 else 2)
    df.set_index("季底日期", inplace=True)

    df_grouped = df.groupby("股票代號")[accounting_item].resample("Q").mean().reset_index()
    df.reset_index(inplace=True)
    df = df_grouped.merge(
        df[["股票代號", "季底日期", "財報類別", "建立日期"]], how="left", on=["股票代號", "季底日期"]
    ).copy()

    # turn the '季底日期' to '年季'
    df["年季"] = df["季底日期"].apply(
        lambda x: x.strftime("%Y") + str(int(x.strftime("%m")) // 3).zfill(2)
    )

    df[f"{accounting_item}"] = df.groupby("股票代號")[f"{accounting_item}"].ffill()
    df["財報類別"] = df.groupby("股票代號")["財報類別"].ffill()
    df["建立日期"] = df.groupby("股票代號")["建立日期"].bfill()

    df["last_announce_date"] = df["年季"].apply(lambda x: last_announce_date(x, 1))
    df["data_valid_date"] = df[["建立日期", "last_announce_date"]].min(axis=1)
    df[f"{accounting_item}_single_q"] = df[f"{accounting_item}"] / df["財報類別"]

    df["建立日期"] = df["data_valid_date"]

    if clean == True:
        df = df[["股票代號", "年季", "建立日期", f"{accounting_item}_single_q"]].copy()

    return df
