import pandas as pd
import yfinance
import requests
import numpy as np
from datetime import datetime
from models import ActionType, PositionType, Position, StrategySignal
from strategies import BaseStrategy, GetRichFastStrategy
from evaluation import evaluate_strategy         
import matplotlib.pyplot as plt
   
def make_api_call(base_url, endpoint="", method="GET", **kwargs):
    full_url = f'{base_url}{endpoint}'
    response = requests.request(method=method, url=full_url, **kwargs)
    if response.status_code == 200:
        return response
    else:
        raise Exception(f'API request failed with status code {response.status_code}: {response.text}')

def get_binance_historical_data(symbol, interval, start_date, end_date):

    # define basic parameters for call
    base_url = 'https://fapi.binance.com'
    endpoint = '/fapi/v1/klines'
    method = 'GET'
    
    # Set the start time parameter in the params dictionary
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1500,
        'startTime': start_date,
        'endTime': end_date
    }

    # Make initial API call to get candles
    response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)

    candles_data = []

    while len(response.json()) > 0:
        # Append the received candles to the list
        candles_data.extend(response.json())

        # Update the start time for the next API call
        params['startTime'] = candles_data[-1][0] + 1 # last candle open_time + 1ms
        
        if params['startTime'] > params['endTime']:
            break

        # Make the next API call
        response = make_api_call(base_url, endpoint=endpoint, method=method, params=params)

    
    # Wrap the candles data as a pandas DataFrame
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
               'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    dtype={
    'open_time': 'datetime64[ms, Asia/Jerusalem]',
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
    'close_time': 'datetime64[ms, Asia/Jerusalem]',
    'quote_asset_volume': 'float64',
    'number_of_trades': 'int64',
    'taker_buy_base_asset_volume': 'float64',
    'taker_buy_quote_asset_volume': 'float64',
    'ignore': 'float64'
    }
    
    df = pd.DataFrame(candles_data, columns=columns)
    df = df.astype(dtype)

    return df
 
def calc_realistic_price(row: pd.Series ,action_type: ActionType, slippage_factor=20):
    slippage_rate = ((row['close'] - row['open']) / row['open']) / slippage_factor
    slippage_price = row['open'] + row['open'] * slippage_rate
    
    if action_type == ActionType.BUY:
        return max(slippage_price, row['open'])
    else:
        return min(slippage_price, row['open'])   

def backtest(data: pd.DataFrame, strategy: BaseStrategy, starting_balance: int, slippage_factor: float=20.0, commission: float=0.0) -> pd.DataFrame:       
    
    def enter_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position_type: PositionType) -> Position:
        if position_type == PositionType.LONG:
            buy_price = calc_realistic_price(row, ActionType.BUY, slippage_factor=slippage_factor)
            qty_to_buy = strategy.calc_qty(buy_price, curr_balance, ActionType.BUY)
            position = Position(qty_to_buy, buy_price, position_type)
            data.loc[index, 'qty'] = curr_qty + qty_to_buy
            data.loc[index, 'balance'] = curr_balance - qty_to_buy * buy_price - commission
            long_tp = buy_price + 2 * abs(row["long_sl"] - buy_price)
            strategy.set_sl_tp(row["long_sl"], None, long_tp, None)
            
        
        elif position_type == PositionType.SHORT:
            sell_price = calc_realistic_price(row, ActionType.SELL, slippage_factor=slippage_factor)
            qty_to_sell = strategy.calc_qty(sell_price, curr_balance, ActionType.SELL)
            position = Position(qty_to_sell, sell_price, position_type)
            data.loc[index, 'qty'] = curr_qty - qty_to_sell
            data.loc[index, 'balance'] = curr_balance + qty_to_sell * sell_price - commission
            short_tp = sell_price - 2 * abs(row["short_sl"] - sell_price)
            strategy.set_sl_tp(None, row["short_sl"], None, short_tp)
        
        return position
    
    def close_position(data: pd.DataFrame, index: int, row: pd.Series, curr_qty: float, curr_balance: float, position: Position):
        if position.type == PositionType.LONG:
            sell_price = calc_realistic_price(row, ActionType.SELL, slippage_factor=slippage_factor)
            data.loc[index, 'qty'] = curr_qty - position.qty
            data.loc[index, 'balance'] = curr_balance + position.qty * sell_price - commission

        elif position.type == PositionType.SHORT:
            buy_price = calc_realistic_price(row, ActionType.BUY, slippage_factor=slippage_factor)
            data.loc[index, 'qty'] = curr_qty + position.qty
            data.loc[index, 'balance'] = curr_balance - position.qty * buy_price - commission
    
    data["strategy_signal"] = strategy.calc_signal(data)
    
    data['short_sl'] = data['high'].rolling(30).max().shift(1)
    data['long_sl'] = data['low'].rolling(30).min().shift(1)
    
    # initialize df 
    data['qty'] = 0.0
    data['balance'] = 0.0
    
    # Loop through the data to calculate portfolio value
    position: Position = None
    data.reset_index(inplace=True)
    max_candle = data.shape[0]
    
    for index, row in data.iterrows():
        curr_qty = data.loc[index - 1, 'qty'] if index > 0 else 0
        curr_balance = data.loc[index - 1, 'balance'] if index > 0 else starting_balance
        if position is not None:
            if curr_qty != 0:
                sl_tp_res = strategy.check_sl_tp(data.iloc[index - 1], position)
                if sl_tp_res is not None:
                    sl_tp_qty, sl_tp_price, sl_tp_action = sl_tp_res
                    if sl_tp_action == ActionType.BUY:
                        curr_balance = curr_balance - sl_tp_qty * sl_tp_price - commission
                        curr_qty = curr_qty + sl_tp_qty
                        position = None
                                                
                    elif sl_tp_action == ActionType.SELL:
                        curr_balance = curr_balance + sl_tp_qty * sl_tp_price - commission
                        curr_qty = curr_qty - sl_tp_qty
                        position = None 
                                  
        if row['strategy_signal'] == ActionType.BUY:
            if position is not None and position.type == PositionType.SHORT:
                row['strategy_signal'] = StrategySignal.CLOSE_SHORT
            elif curr_qty == 0:
                row['strategy_signal'] = StrategySignal.ENTER_LONG
        elif row['strategy_signal'] == ActionType.SELL:
            if position is not None and position.type == PositionType.LONG:
                row['strategy_signal'] = StrategySignal.CLOSE_LONG
            elif curr_qty == 0:
                row['strategy_signal'] = StrategySignal.ENTER_SHORT
                    
        # Close position at end of trade
        if index + 1 == max_candle and position is not None: 
            close_position(data, index, row, curr_qty, curr_balance, position)
            position = None
                
        # Handle enter long signal
        elif row['strategy_signal'] == StrategySignal.ENTER_LONG:
            position = enter_position(data, index, row, curr_qty, curr_balance, PositionType.LONG)
        
        # Handle enter short signal  
        elif row['strategy_signal'] == StrategySignal.ENTER_SHORT:
            position = enter_position(data, index, row, curr_qty, curr_balance, PositionType.SHORT)
        
        # Handle close long or short signal 
        elif row['strategy_signal'] in [StrategySignal.CLOSE_LONG, StrategySignal.CLOSE_SHORT] and position is not None:
            close_position(data, index, row, curr_qty, curr_balance, position)
            strategy.set_sl_tp(None, None, None, None)
            position = None
        
        else:
            data.loc[index, 'qty'] = curr_qty
            data.loc[index, 'balance'] = curr_balance
        
    # Calculate portfolio value
    data['portfolio_value'] = data['close'] * data['qty'] + data['balance']
    return data

def makeData():
    symbol = 'BTCUSDT'
    interval = '30m'
    start_date = int(datetime(year=2022, month=12, day=20).timestamp() * 1000)
    end_date = int(datetime(year=2024, month=1, day=1).timestamp() * 1000)
    return get_binance_historical_data(symbol, interval, start_date, end_date)

if __name__ == '__main__':
    btcusdt_df = makeData()
    strategy = GetRichFastStrategy()
    balance = 100000
    b_df = backtest(btcusdt_df.copy(deep=True), strategy, balance, 20)
    evaluate_strategy(b_df, 'Get Rich Fast Strategy')
    print("\nPortfolio value:", b_df["portfolio_value"].iloc[-1], "\n")
    
    strategy2 = GetRichFastStrategy(buy_atr_period=200)
    b_df2 = backtest(btcusdt_df.copy(deep=True), strategy2, balance, 20)
    evaluate_strategy(b_df2, 'Get Rich Fast Strategy with buy_atr_period = 200')
    print("\nPortfolio value:", b_df2["portfolio_value"].iloc[-1], "\n")

    b_df['Percentage_Change'] = ((b_df['portfolio_value'] - balance) / balance) * 100

    # Plotting
    plt.figure(figsize=(10, 8))

    # Plot the percentage changes over time
    plt.subplot(2, 1, 1)
    plt.plot(b_df['open_time'], b_df['Percentage_Change'], linestyle='-', color='b')
    plt.xlabel('open time')
    plt.ylabel('Percentage Change (%)')
    plt.title('Percentage Change in Portfolio Value Over Time')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Plot close prices with signals
    plt.subplot(2, 1, 2)
    plt.plot(b_df['open_time'], b_df['close'], color="gray", linestyle='-', label='Close Price', linewidth=0.75)
    plt.scatter(b_df['open_time'][b_df['strategy_signal'] == ActionType.BUY], b_df['close'][b_df['strategy_signal'] == ActionType.BUY], color='g', marker='^', label='Buy Signal')
    plt.scatter(b_df['open_time'][b_df['strategy_signal'] == ActionType.SELL], b_df['close'][b_df['strategy_signal'] == ActionType.SELL], color='r', marker='v', label='Sell Signal')
    plt.xlabel('open time')
    plt.ylabel('Close Price')
    plt.title('Close Prices with Signal')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    

