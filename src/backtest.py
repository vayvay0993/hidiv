import re
import pandas as pd
import numpy as np
# from db import pd_read_mssql_data
from datetime import datetime
import os
# import any from Typing
from typing import Any, List, Union, Tuple, Dict, Optional, Callable
import copy as copy

class DataDescriptor:
    def __init__(self, name) -> None:
        self.name= name

    def __get__(self, instance, owner):
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{self.name.capitalize()} must be pd.DataFrame")
        instance.__dict__[self.name] = value


        
def get_changes(ar1: list, ar2: list, step: int, position = False):
    """ Get the changes between two lists of numbers in a given number of steps.

    This function calculates a linear interpolation between the two arrays `ar1` and `ar2` 
    over `step` number of steps. The interpolated arrays are then returned as a list.

    Args:
        ar1 (list): The first list of numbers. This can be any sequence of real numbers.
        ar2 (list): The second list of numbers. This should be the same length as `ar1` and can also be any sequence of real numbers.
        step (int): The number of steps over which to interpolate between `ar1` and `ar2`. Must be a positive integer.
        position (bool, optional): Not used in the current implementation of the function.

    Returns:
        list: A list of lists, where each sublist is one step in the interpolation from `ar1` to `ar2`. 
            The first sublist will be identical to `ar1`, and the last sublist will be identical to `ar2`. 
            There will be `step` number of sublists in total.
    """
    ar1, ar2 = np.asarray(ar1), np.asarray(ar2)
    weights = np.linspace(1, 0, step)
    ar1_weights = weights[:, None]
    ar2_weights = 1 - ar1_weights
    
    step_changes = (ar1_weights * ar1) + (ar2_weights * ar2)

    return step_changes.tolist()

class Backtest:
    _data_open = DataDescriptor('data_open')
    _data_close = DataDescriptor('data_close')
    _data_adj_close = DataDescriptor('data_adj_close')
    _data_high = DataDescriptor('data_high')
    _data_low = DataDescriptor('data_low')
    _data_volume = DataDescriptor('data_volume')
    _data_cash_dividends = DataDescriptor('data_cash_dividends')
    _data_market_capital = DataDescriptor('data_market_capital')
    _benchmark = DataDescriptor('data_market_capital')

    # initialize backtest object, wait user to use set_position and set_weighting to set position and weighting
    def __init__(self) -> None:
        self.__version__ = '0.0.19'

        self._start_date = None
        self._end_date = None
        # Set the initial capital to 1,000,000 by default
        self._initial_capital = 10000000
        self.tax_rate = 0.003

        self._position = None
        self._weighting = None
        self._weighting_cap = None

        # how many days to rebalance, default is 1, means rebalance on given date, if 5, means it takes 5 days to rebalance
        self._rebalance_days = 1

        self._record_trade_log = None
        # record position store the position of each day, value is the number of shares
        self._record_position = None 
        # record position value store the value of each day, value is the value of shares
        self._record_position_value = None
        self._record_weight = None
        self._record_asset_value = None
        self._record_cash_dividends = None

    @property
    def start_date(self) -> str | None:
        return self._start_date
    
    @property
    def end_date(self) -> str | None:
        return self._end_date

    @property
    def position(self) -> pd.DataFrame | None:
        return self._position
    
    @property
    def weighting(self):
        return self._weighting
    
    @property
    def weighting_cap(self):
        return self._weighting_cap
    
    @property
    def rebalance_days(self):
        return self._rebalance_days
    
    @property
    def initial_capital(self):
        return self._initial_capital
    
    @property
    def trade_log(self):
        return self._record_trade_log
    
    @property
    def record_position(self):
        return self._record_position
    
    @property
    def tax_rate(self) -> float:
        return self._tax_rate
    
    @property
    def record_trade_log(self):
        return self._record_trade_log
    
    @property
    def backtest_time(self) -> str:
        return self._backtest_time

    @property
    def initial_capital(self):
        return self._initial_capital
    
    @backtest_time.setter
    def backtest_time(self, backtest_time: str) -> None:
        # check if backtest_time is str
        if not isinstance(backtest_time, str):
            raise TypeError("Backtest_time must be str")
        self._backtest_time = backtest_time
    
    @tax_rate.setter
    def tax_rate(self, tax_rate: float) -> None:
        # check if tax_rate is float
        if not isinstance(tax_rate, float):
            raise TypeError("Tax_rate must be float")
        self._tax_rate = tax_rate

    
    @start_date.setter
    def start_date(self, start_date: str) -> None:
        # check if start_date is str
        if not isinstance(start_date, str):
            raise TypeError("Start_date must be str")
        self._start_date = start_date

    @end_date.setter
    def end_date(self, end_date: str) -> None:
        # check if end_date is str
        if not isinstance(end_date, str):
            raise TypeError("End_date must be str")
        self._end_date = end_date
    
    # set rebalance days
    @rebalance_days.setter
    def rebalance_days(self, rebalance_days: int) -> None:
        # check if rebalance_days is int
        if not isinstance(rebalance_days, int):
            raise TypeError("Rebalance_days must be int")
        self._rebalance_days = rebalance_days

    @initial_capital.setter
    def initial_capital(self, value) -> None:
        if value < 0:  # just an example condition
            raise ValueError("Initial capital must be a positive value.")
        self._initial_capital = value

    # set position, the input should be a pd.DataFrame with stock positions (0, 1) and index should be date, columns should be stock ticker
    @position.setter
    def position(self, position: pd.DataFrame) -> None:
        # check if position is pd.DataFrame
        if not isinstance(position, pd.DataFrame):
            raise TypeError("Position must be pd.DataFrame")
        
        # check if rebalance_days is 1
        if self._rebalance_days == 1:
            self._position = position

    # set weighting (weighting should sum to 1), input can be a pd.DataFrame or string(ex: ew for equal weighting, cw for cap weighting)
    @weighting.setter
    def weighting(self, weighting: pd.DataFrame or str) -> None:
        # check if weighting is pd.DataFrame
        if isinstance(weighting, pd.DataFrame):
            # check if weighting is valid
            if weighting.sum(axis=1).all() != 1:
                raise ValueError("Weighting must be valid")

            self._weighting = weighting

        # check if weighting is string
        elif isinstance(weighting, str):
            # check if weighting is valid
            if weighting not in ["ew", "cw"]:
                raise ValueError("Weighting must be valid, should be ew(equal weighting) or cw(cap weighting)")

            # equal weighting
            if weighting == "ew":
                # check if position is set
                if self.position is None:
                    raise ValueError("Position must be set")
                # create a copy of position and replace all 1 with 1/ (number of stocks that position is 1)
                self._weighting = self.position.copy()

                # Divide each row by the sum of the row
                self._weighting = self._weighting.apply(lambda x: x / x.sum(), axis=1)

            # cap weighting
            elif weighting == "cw":
                # check if position and data_market_capital is set
                if self.position is None:
                    raise ValueError("Position must be set")
                if self.data_market_capital is None:
                    raise ValueError("Data_market_capital must be set")

                position_market_cap = self.position * self.data_market_capital
                total_market_cap = position_market_cap.sum(axis=1)
                self._weighting = position_market_cap.div(total_market_cap, axis=0)

        # if _weighting_cap is set, cap the weighting
        if self._weighting_cap is not None:
            # while self._weight have value greater than _weighting_cap 
            while (self._weighting > self._weighting_cap).any().any():
                # calculate the difference between the value and _weighting_cap
                difference = self._weighting - self._weighting_cap
                # sum the difference if greater then 0
                difference[difference < 0] = 0
                sum_difference = difference.sum()

                # set the value greater than _weighting_cap to _weighting_cap
                self._weighting[self._weighting > self._weighting_cap] = self._weighting_cap
                # and add the difference to the value that is less than _weighting_cap by the ratio of it value by row


    # set weight cap
    @weighting_cap.setter
    def weighting_cap(self, weight_cap: float) -> None:
        # check if weight_cap is float
        if not isinstance(weight_cap, float):
            raise TypeError("Weight_cap must be float")
        self._weighting_cap = weight_cap

    # init the position
    def init_position(self) -> None:
        # check if data_close is set
        if self.data_close is None:
            raise ValueError("Data_close must be set")

        # set the position from copy of data_close
        self.position = copy.copy(self._data_close)
        # and set all values to 0
        self.position.iloc[:, :] = 0
    
    # # get data from db, input should be a dict with keys: server, database, server_uid, server_pwd, table_name
    # def get_data_from_db(self, db_setting, table_name) -> pd.DataFrame:
    #     # check if db_setting is dict
    #     if not isinstance(db_setting, dict):
    #         raise TypeError("db_setting must be dict")
    #     # check if db_setting has all keys
    #     if not all(
    #         key in db_setting for key in ["server", "database", "server_uid", "server_pwd"]
    #     ):
    #         raise ValueError("db_setting must be valid")
        
    #     server = db_setting["server"]
    #     database = db_setting["database"]
    #     server_uid = db_setting["server_uid"]
    #     server_pwd = db_setting["server_pwd"]

    #     # create sql query for data_open
    #     try:
    #         sql_query = f"select * from {database}.dbo.{table_name}"
    #         data = pd_read_mssql_data(sql_query, server, database, server_uid, server_pwd, local=False)
    #         return data
    #     except:
    #         raise ValueError("table do not exist")
    
    # create a data dict from db_setting
    def set_data_from_db(self, db_setting) -> None:
        # check if db_setting is dict
        if not isinstance(db_setting, dict):
            raise TypeError("db_setting must be dict")
        # check if db_setting has all keys
        if not all(
            key in db_setting for key in ["server", "database", "server_uid", "server_pwd"]
        ):
            raise ValueError("db_setting must be valid")
        
        # call get_data_from_db to get data
        self._data_open = self.get_data_from_db(db_setting, "backtest_open")
        self._data_close = self.get_data_from_db(db_setting, "backtest_close")
        self._data_adj_close = self.get_data_from_db(db_setting, "backtest_adj_close")
        self._data_high = self.get_data_from_db(db_setting, "backtest_high")
        self._data_low = self.get_data_from_db(db_setting, "backtest_low")
        self._data_volume = self.get_data_from_db(db_setting, "backtest_volume")
        self._data_cash_dividends = self.get_data_from_db(db_setting, "backtest_cash_dividends")
        self._data_market_capital = self.get_data_from_db(db_setting, "backtest_market_capital")

    # validate necessary columns in dataframes
    def validate_columns(self) -> None:
        # Check if data frames have the required columns and raise an alert with a message specifying which one is missing
        required_columns = set(self.position.columns)

        data_frames = {
            "data_open": self._data_open,
            "data_close": self._data_close,
            "data_adj_close": self._data_adj_close,
            "data_high": self._data_high,
            "data_low": self._data_low,
            "data_volume": self._data_volume,
            "data_market_capital": self._data_market_capital,
        }

        missing_required_columns = [
            df_name for df_name, df in data_frames.items() if not required_columns.issubset(df.columns)
        ]

        if any(missing_required_columns):
            missing_required_columns_str = ', '.join(missing_required_columns)
            raise ValueError(f"The following data frames do not have the required columns: {missing_required_columns_str}")

    
    # check position, weighting and data is set
    def check_set(self) -> None:
        # check if position are set
        if self.position is None:
            raise ValueError("Position must be set")
        # check if weighting are set
        if self._weighting is None:
            raise ValueError("Weighting must be set")

        # Check if data are set and raise an alert with a message specifying which property is None
        data_properties = {
            "_data_open": self._data_open,
            "_data_close": self._data_close,
            "_data_adj_close": self._data_adj_close,
            "_data_high": self._data_high,
            "_data_low": self._data_low,
            "_data_volume": self._data_volume,
            "_data_cash_dividends": self._data_cash_dividends,
            "_data_market_capital": self._data_market_capital,
        }

        missing_data_properties = [prop for prop, value in data_properties.items() if value is None]

        if any(missing_data_properties):
            missing_data_str = ', '.join(missing_data_properties)
            raise ValueError(f"Data must be set for the following properties: {missing_data_str}")

    # get the non full na rows index of position as rebalance days

    def get_rebalance_days(self) -> list:
        # check if position is set
        if self.position is None:
            raise ValueError("Position must be set")
        
        # get the non full na rows index of position as rebalance days
        return self.position.dropna(how="all").index.tolist()
    
    # get the index of data_close as trading days
    def get_trading_days(self) -> list:
        # check if data_close is set
        if self._data_close is None:
            raise ValueError("Data_close must be set")
        
        # get the index of data_close as trading days
        return self._data_close.index.tolist()
    

    # run backtest
    def run(self) -> None:
        
        # set backtest time, use datetime.now(), format: YYYY-MM-DD_HH:MM:SS
        self.backtest_time = datetime.now().strftime("%Y%m%d%H%M%S")
        # use check_set to check if position, weighting and data are set
        self.check_set()

        # use validate_columns to check if all necessary columns are present in dataframes
        self.validate_columns()

        # get rebalance days
        rebalance_days = self.get_rebalance_days()

        # get trading days
        trading_days = self.get_trading_days()

        # backtest logic:
        # 1. if rebalance day, rebalance
        # 2. if not rebalance day, check if stop loss or take profit, if yes, rebalance
        # 3. if not rebalance day, check if stop loss or take profit, if no, do nothing and record log
        # loop through each day by the trading days
        for day in trading_days:
            # check if day is a rebalance day
            if day in rebalance_days:
                # rebalance
                self.rebalance(day)
                self.log_trade(day)
            
            # TODOS: check if stop loss or take profit
            # else:
                # # check if stop loss or take profit
                # if self.stop_loss(day) or self.take_profit(day):
                #     # rebalance
                #     self.rebalance(day)
                # else:
                #     # do nothing and record log
                #     continue
            
            self.log_cash_dividends(day)
            self.log_nav(day)

    def log_nav(self, day) -> None:
        
        try:
            # get the next day of last date
            last_date = self._record_position.index[-1]
            # 1. calculate the period return of each stock by current adj close / adj close of next_day
            period_return = self._data_adj_close.loc[day] / self._data_adj_close.loc[last_date]

            # 2. multiply the period return of each stock by the quantity of each stock in last date
            current_position_value = pd.DataFrame(self._record_position.loc[last_date] * period_return * self._data_close.loc[last_date]).transpose()
            # 3. set the index of current_position_value to day
            current_position_value.index = [day]
            nav = current_position_value.sum(axis=1, skipna=True).values[0]

            self._record_position_value = pd.concat([self._record_position_value, current_position_value], axis=0)

        except:
            # do nothing
            pass

    def log_cash_dividends(self, day) -> None:
        # record cash dividends log:
        # rebalance logic:
        try:
            # get the index of last date in record_position
            current_quantity = self._record_position.loc[self._record_position.index[-1]]
            # time the current quantity with cash dividends to get cash dividends of each stock
            cash_dividends = current_quantity * self._data_cash_dividends.loc[day]
            cash_dividends = pd.DataFrame(cash_dividends).transpose()
            # set the index as day
            cash_dividends.index = [day] * len(cash_dividends)
            self._record_cash_dividends = pd.concat([self._record_cash_dividends, cash_dividends], axis=0)

        except:
            pass

    # rebalance
    def rebalance(self, day) -> None:
        
        # rebalance logic:
        try:
            # get the next day of last date
            last_date = self._record_position.index[-1]

            # 1. calculate the period return of each stock by current adj close / adj close of next_day
            period_return = self._data_adj_close.loc[day] / self._data_adj_close.loc[last_date]
            
            current_asset_value =  pd.DataFrame(self._record_position.loc[last_date] * period_return * self._data_close.loc[last_date]).transpose().sum(axis=1, min_count=1).values[0]


        except:
            # if _record_position is empty, set current_asset_value to initial_capital
            current_asset_value = self._initial_capital

        # 2. time the current asset value with weighting to get target stock value of each stock
        today_weighting  = self._weighting.loc[day]
        target_stock_value = current_asset_value *  today_weighting
        # 3. divide target value of each stock by current price to get target quantity of each stock
        today_close = self._data_close.loc[day]
        target_stock_quantity = target_stock_value / today_close
        # create a new dataframe, set the columns as the columns of position, and set the index as day, and set the value as target_stock_quantity
        target_stock_quantity = pd.DataFrame(target_stock_quantity).transpose()

        # record the rebalance log
        # 1. concat the target quantity to record_position, and set the index as day
        self._record_position = pd.concat([self._record_position, target_stock_quantity], axis=0)

    
    def log_trade(self, day) -> None:

        # record trade log:
        # 1. record the trade log of each day
        # 2. if today quantity of a stock is greater than last quantity, log as buy
        # 3. if today quantity of a stock is less than last quantity, log as sell
        # 4. if today quantity of a stock is equal to last quantity, do nothing
        today_quantity = self._record_position.loc[day]

        try:
            last_quantity = self._record_position.loc[self._record_position.index[-2]]
        except:
            # if _record_position has only one row, set last_quantity to 0
            last_quantity = 0

        # get the difference between today quantity and last quantity
        difference = today_quantity - last_quantity
        # get the columns of difference that are not zero
        difference = difference[difference != 0]
        # turn the difference into a dataframe
        difference = pd.DataFrame(difference)
        difference.columns = ["quantity"]



        # add a column of stock name
        difference["stock"] = difference.index
        # set the index as day
        difference.index = [day] * len(difference)
        # add a column of trade type
        difference["trade_type"] = np.where(difference["quantity"] > 0, "buy", "sell")
        # add a column of price
        difference["price"] = self._data_close.loc[day, difference["stock"]].values
        # add a column of tax, price * abs(quantity) * tax rate * , only for sell
        difference["tax"] = np.where(
            difference["trade_type"] == "sell",
            difference["price"] * difference["quantity"].abs() * self.tax_rate,
            0,
        )

        # filter out the stocks that quantity is 0 or nan
        difference = difference[(difference["quantity"] != 0) & (difference["quantity"].isna() == False)]

        # concat the difference to trade log
        self._record_trade_log = pd.concat([self._record_trade_log, difference], axis=0)

    # record log
    def record_log(self, day) -> None:
        self.log_trade(day)

    # calculate the daily return of the portfolio and calculate some statistics
    def calculate_nav(self) -> None:
        # calculate the daily return of the portfolio
        # 1. sum the value of each stock in record_position_value
        
        portfolio_value = pd.DataFrame(self._record_position_value.sum(axis=1))
        # set index as index and column as "nav"
        portfolio_value.columns = ["nav"]

        # 2. sum the amount of cash dividends in record_cash_dividends
        cash_dividends = pd.DataFrame(self._record_cash_dividends.sum(axis=1))
        # set index as index and column as "dividend"
        cash_dividends.columns = ["dividend"]

        # merge the portfolio_value and cash_dividends with column name "nav" and "dividend"
        df = pd.DataFrame(portfolio_value).merge(cash_dividends, left_index=True, right_index=True)

        self._backtest_portfolio = df
    
    def summary(self):
        backtest.calculate_nav()

        df = copy.copy(self._backtest_portfolio)

        # CAGR
        def calculate_cagr(start_value, end_value, num_years):
            return (end_value / start_value) ** (1 / num_years) - 1

        start_nav = df.iloc[0]['nav']
        end_nav = df.iloc[-1]['nav']
        num_years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = calculate_cagr(start_nav, end_nav, num_years)

        # max drawdown
        df['peak'] = df['nav'].cummax()
        df['drawdown'] = (df['nav'] - df['peak']) / df['peak']
        mdd = df['drawdown'].min()

        # sharpe ratio
        daily_returns = df['nav'].pct_change().dropna()
        annualized_returns = np.mean(daily_returns) * 252
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)
        risk_free_rate = 0.02  # Assume a constant risk-free rate of 2% for simplicity

        sharpe_ratio = (annualized_returns - risk_free_rate) / annualized_volatility


    def summary_mdd(self, period = 'yearly'):
        self.calculate_nav()

        df = copy.copy(self._backtest_portfolio)

        if period not in ['yearly', 'quarterly', 'monthly']:
            raise ValueError("period must be one of 'yearly', 'quarterly', or 'monthly'")

        df['period'] = df.index.to_period('Y' if period == 'yearly' else 'Q' if period == 'quarterly' else 'M')

        # create empty list to hold max drawdown for each period, mdd start date, and mdd end date
        mdd_list = []
        mdd_start_list = []
        mdd_end_list = []

        # loop through each period
        for period in df['period'].unique():
            df_period = copy.deepcopy(df[df['period'] == period])

            # calculate max drawdown for the period
            df_period['peak'] = df_period['nav'].cummax()
            df_period['drawdown'] = (df_period['nav'] - df_period['peak']) / df_period['peak']
            mdd = df_period['drawdown'].min()

            # append the max drawdown to the list
            mdd_list.append(mdd)

            # append the mdd start date and mdd end date to the list
            mdd_start_list.append(df_period[df_period['drawdown'] == 0].index[0])
            mdd_end_list.append(df_period[df_period['drawdown'] == mdd].index[0])

        # create a dataframe to hold the results
        df_mdd = pd.DataFrame({'mdd': mdd_list, 'mdd_start': mdd_start_list, 'mdd_end': mdd_end_list})

        return df_mdd


    def summary_statistics(self, period='yearly'):
        self.calculate_nav()

        df = copy.copy(self._backtest_portfolio)

        if period not in ['yearly', 'quarterly', 'monthly']:
            raise ValueError("period must be one of 'yearly', 'quarterly', or 'monthly'")

        df['period'] = df.index.to_period('Y' if period == 'yearly' else 'Q' if period == 'quarterly' else 'M')

        # Return
        df_grouped = df.groupby('period').agg({'nav': ['first', 'last']}).reset_index()
        period_return = (df_grouped[('nav', 'last')] / df_grouped[('nav', 'first')]) - 1

        # Yield
        df_grouped_y = df.groupby('period').agg({'nav': 'first', 'dividend': 'sum'}).reset_index()
        period_yield = df_grouped_y['dividend'] / df_grouped_y['nav']

        # MDD, Sharpe Ratio, Return Std
        period_mdd = []
        period_sharpe_ratio = []
        period_return_std = []

        for p in df['period'].unique():
            df_period = copy.deepcopy(df[df['period'] == p])
            daily_returns = df_period['nav'].pct_change().dropna()

            # MDD
            df_period['peak'] = df_period['nav'].cummax()
            df_period['drawdown'] = (df_period['nav'] - df_period['peak']) / df_period['peak']
            period_mdd.append(df_period['drawdown'].min())

            # Sharpe Ratio
            annualized_returns = np.mean(daily_returns) * 252
            annualized_volatility = np.std(daily_returns) * np.sqrt(252)
            risk_free_rate = 0.02
            period_sharpe_ratio.append((annualized_returns - risk_free_rate) / annualized_volatility)

            # Return std
            period_return_std.append(np.std(daily_returns))

        # Create a new DataFrame to store the results
        results = pd.DataFrame({
            f'{period}_period': df['period'].unique(),
            f'{period}_return': period_return,
            f'{period}_yield': period_yield,
            f'{period}_mdd': period_mdd,
            f'{period}_sharpe_ratio': period_sharpe_ratio,
            f'{period}_return_std': period_return_std
        })

        return results


    def data_to_excel(self) -> None:
        # create a folder to store the backtest result, folder name is "backtest_{current time}"
        folder_name = f'backtest_{self.backtest_time}'
        # create the folder if not exist
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        # save the backtest result to excel
        self._data_close.to_excel(f"{folder_name}/data_close.xlsx")
        self._data_adj_close.to_excel(f"{folder_name}/data_adj_close.xlsx")
        self._data_cash_dividends.to_excel(f"{folder_name}/data_cash_dividends.xlsx")
        self.weighting.to_excel(f"{folder_name}/data_weighting.xlsx")

    def data_to_feather(self) -> None:
        # create a folder to store the backtest result, folder name is "backtest_{current time}"
        folder_name = f'backtest_{self.backtest_time}'
        # create the folder if not exist
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        self._data_adj_close.to_feather(f"{folder_name}/data_adj_close.feather")
        self._data_close.to_feather(f"{folder_name}/data_close.feather")
        self._data_cash_dividends.to_feather(f"{folder_name}/data_cash_dividends.feather")
        self._data_high.to_feather(f"{folder_name}/data_high.feather")
        self._data_low.to_feather(f"{folder_name}/data_low.feather")
        self._data_open.to_feather(f"{folder_name}/data_open.feather")
        self._data_volume.to_feather(f"{folder_name}/data_volume.feather")

    def record_to_excel(self) -> None:
        # create a folder to store the backtest result, folder name is "backtest_{current time}"
        folder_name = f'backtest_{self.backtest_time}'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        # save the backtest result to excel
        self._record_position.to_excel(f"{folder_name}/record_position.xlsx")
        self._record_position_value.to_excel(f"{folder_name}/record_position_value.xlsx")
        self._record_trade_log.to_excel(f"{folder_name}/record_trade_log.xlsx")
        self._record_cash_dividends.to_excel(f"{folder_name}/record_cash_dividends.xlsx")

    @property
    def version(self):
        return self.__version__
    


if __name__ == "__main__":

    df_0002_日收盤表排行_open = pd.read_feather('data/df_0002_日收盤表排行_open.feather')
    df_0002_日收盤表排行_high = pd.read_feather('data/df_0002_日收盤表排行_high.feather')
    df_0002_日收盤表排行_low = pd.read_feather('data/df_0002_日收盤表排行_low.feather')
    df_0002_日收盤表排行_close = pd.read_feather('data/df_0002_日收盤表排行_close.feather')
    df_0002_日收盤表排行_volume = pd.read_feather('data/df_0002_日收盤表排行_volume.feather')
    df_0002_日收盤表排行_market_cap = pd.read_feather('data/df_0002_日收盤表排行_market_cap.feather')
    df_0154_日收盤還原表排行_還原分紅_close = pd.read_feather('data/df_0154_日收盤還原表排行_還原分紅_close.feather')

    # convert the '日期' column to timestamp
    df_0002_日收盤表排行_open['日期'] = pd.to_datetime(df_0002_日收盤表排行_open['日期'])
    df_0002_日收盤表排行_high['日期'] = pd.to_datetime(df_0002_日收盤表排行_high['日期'])
    df_0002_日收盤表排行_low['日期'] = pd.to_datetime(df_0002_日收盤表排行_low['日期'])
    df_0002_日收盤表排行_close['日期'] = pd.to_datetime(df_0002_日收盤表排行_close['日期'])
    df_0002_日收盤表排行_volume['日期'] = pd.to_datetime(df_0002_日收盤表排行_volume['日期'])
    df_0002_日收盤表排行_market_cap['日期'] = pd.to_datetime(df_0002_日收盤表排行_market_cap['日期'])
    df_0154_日收盤還原表排行_還原分紅_close['日期'] = pd.to_datetime(df_0154_日收盤還原表排行_還原分紅_close['日期'])

    # set the '日期' column as index
    df_0002_日收盤表排行_open = df_0002_日收盤表排行_open.set_index('日期')
    df_0002_日收盤表排行_high = df_0002_日收盤表排行_high.set_index('日期')
    df_0002_日收盤表排行_low = df_0002_日收盤表排行_low.set_index('日期')
    df_0002_日收盤表排行_close = df_0002_日收盤表排行_close.set_index('日期')
    df_0002_日收盤表排行_volume = df_0002_日收盤表排行_volume.set_index('日期')
    df_0002_日收盤表排行_market_cap = df_0002_日收盤表排行_market_cap.set_index('日期')
    df_0154_日收盤還原表排行_還原分紅_close = df_0154_日收盤還原表排行_還原分紅_close.set_index('日期')

    # set data to float
    df_0002_日收盤表排行_open = df_0002_日收盤表排行_open.astype(float)
    df_0002_日收盤表排行_high = df_0002_日收盤表排行_high.astype(float)
    df_0002_日收盤表排行_low = df_0002_日收盤表排行_low.astype(float)
    df_0002_日收盤表排行_close = df_0002_日收盤表排行_close.astype(float)
    df_0002_日收盤表排行_volume = df_0002_日收盤表排行_volume.astype(float)
    df_0002_日收盤表排行_market_cap = df_0002_日收盤表排行_market_cap.astype(float)
    df_0154_日收盤還原表排行_還原分紅_close = df_0154_日收盤還原表排行_還原分紅_close.astype(float)

    df_position = pd.DataFrame(data = np.where(df_0002_日收盤表排行_open > 0, 1, 0), columns= df_0002_日收盤表排行_open.columns, index= df_0002_日收盤表排行_open.index)

    df_position['year'] = df_position.index.year
    df_position['quarter'] = df_position.index.quarter
    df_position = df_position.groupby(['year','quarter']).head(1)

    del df_position['year']
    del df_position['quarter']

    backtest = Backtest()
    backtest.data_market_capital = df_0002_日收盤表排行_market_cap
    backtest.data_open = df_0002_日收盤表排行_open
    backtest.data_close = df_0002_日收盤表排行_close
    backtest.data_adj_close = df_0154_日收盤還原表排行_還原分紅_close
    backtest.data_high = df_0002_日收盤表排行_high
    backtest.data_low = df_0002_日收盤表排行_low
    backtest.data_volume = df_0002_日收盤表排行_volume
    backtest.data_cash_dividends = df_0002_日收盤表排行_volume


    backtest.init_position()
    backtest.position = df_position
    backtest.weighting = 'ew'

    backtest.run()
    # backtest.record_to_excel()
    # # backtest.data_to_excel()

    cw_portfolio = backtest._record_position_value.sum(axis=1)
    df_twa02 = pd.read_excel('TWA02.xlsx')
    df_compare = pd.DataFrame(cw_portfolio)
    df_compare.columns = ['backtest']
    df_compare['TWA02'] = [x[0] for x in df_twa02.values.tolist()][::-1]
    df_compare['TWA02'] = df_compare['TWA02'] / df_compare['TWA02'][0] * 10000000
    df_compare.tail()
    df_compare.plot()
    print(backtest.summary_statistics())
    print(backtest._record_position_value.sum(axis=1))
    # backtest.record_to_excel()
    # backtest.data_to_excel()