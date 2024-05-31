import pandas as pd
import numpy as np
import random
from models import ActionType, Position, PositionType, StrategySignal
from abc import ABC, abstractmethod
from typing import Tuple

START_DATE_INDEX = 576

class BaseStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.long_sl = None
        self.short_sl = None
        self.long_tp = None
        self.short_tp = None

    def set_sl_tp(self, long_sl, short_sl, long_tp, short_tp):
        self.long_sl = long_sl
        self.short_sl = short_sl
        self.long_tp = long_tp
        self.short_tp = short_tp
    
    @abstractmethod
    def calc_signal(self, data: pd.DataFrame):
        pass

    def calc_qty(self, real_price: float, balance: float, action: ActionType, **kwargs) -> float:
        if action == ActionType.BUY:
            qty = balance / real_price
        
        elif action == ActionType.SELL:
            qty =  balance / real_price
        
        return qty    
    
    def check_sl_tp(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        tp_res = self.is_take_profit(row, position)
        sl_res = self.is_stop_loss(row, position)
        if tp_res == None:
            return sl_res
        elif sl_res == None:
            return tp_res
        else:
            p = random.random()
            if p > 0.5:
                return sl_res
            else:
                return tp_res
    
    def is_stop_loss(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the stop-loss level.
        
        Returns:
            Tuple[float, float, ActionType] or None: If stop-loss is triggered, returns a tuple containing quantity and stop-loss price and action type, otherwise returns None.
        """
        if position.type == PositionType.LONG and self.long_sl != None:
            long_stop_loss_price = self.long_sl
            if row['low'] <= long_stop_loss_price:
                return position.qty, long_stop_loss_price, ActionType.SELL
        if position.type == PositionType.SHORT and self.short_sl != None:
            short_stop_loss_price = self.short_sl
            if row['high'] >= short_stop_loss_price:
                return position.qty, short_stop_loss_price, ActionType.BUY
    
    def is_take_profit(self, row: pd.Series, position: Position) -> Tuple[float, float, ActionType]:
        """
        Checks if the price has hit the take-profit level.

        Returns:
            Tuple[float, float, ActionType] or None: If take-profit is triggered, returns a tuple containing quantity and take-profit price and action type, otherwise returns None.
        """
        if position.type == PositionType.LONG and self.long_tp != None:
            long_take_profit_price = self.long_tp
            if row['low'] >= long_take_profit_price:
                return position.qty, long_take_profit_price, ActionType.SELL
        if position.type == PositionType.SHORT and self.short_tp != None:
            short_take_profit_price = self.short_tp
            if row['high'] <= short_take_profit_price:
                return position.qty, short_take_profit_price, ActionType.BUY

class GetRichFastStrategy(BaseStrategy):
    def __init__(self, buy_atr_period: int = 300) -> pd.Series:
        super().__init__()
        self.buy_atr_period = buy_atr_period
    
    def ATR(self, df: pd.DataFrame, period: int) -> pd.Series:
        tr = np.maximum(df['high'], df['close'].shift(1)) - np.minimum(df['low'], df['close'].shift(1))    
        atr = tr.rolling(period).mean()
        return atr

    def calc_UTBot(self, data, key_value, atr_period):
        atr = self.ATR(data, atr_period)
        loss_threshold = key_value * pd.Series(atr)
        trailing_stop = [0] * len(data)
        for i in range(1, len(data)):
            if data['close'].iloc[i] > trailing_stop[i-1] and data['close'].iloc[i-1] > trailing_stop[i-1]:
                trailing_stop[i] = max(trailing_stop[i-1], data['close'].iloc[i] - loss_threshold.iloc[i])
            elif data['close'].iloc[i] < trailing_stop[i-1] and data['close'].iloc[i-1] < trailing_stop[i-1]:
                trailing_stop[i] = min(trailing_stop[i-1], data['close'].iloc[i] + loss_threshold.iloc[i])
            elif data['close'].iloc[i] > trailing_stop[i-1]:
                trailing_stop[i] = data['close'].iloc[i] - loss_threshold.iloc[i]
            else:
                trailing_stop[i] = data['close'].iloc[i] + loss_threshold.iloc[i]
        
        above = (data['close'] > pd.Series(trailing_stop)) & (data['close'].shift(1) < pd.Series(trailing_stop).shift(1))
        below = (pd.Series(trailing_stop) > data['close']) & (pd.Series(trailing_stop).shift(1) < data['close'].shift(1))
        
        buy_signals = above.astype(int)
        sell_signals = below.astype(int)
        
        return buy_signals, sell_signals
    
    def MacdDiff(self, close, fast_length, slow_length):
        fast_ema = close.ewm(span=fast_length, min_periods=fast_length, adjust=False).mean()
        slow_ema = close.ewm(span=slow_length, min_periods=slow_length, adjust=False).mean()
        return fast_ema - slow_ema

    def SmoothSrs(self, srs, smoothing_f):
        smoothed_srs = pd.Series([0] * len(srs), index=srs.index)
        smoothed_srs.iloc[0] = srs.iloc[0]

        for i in range(1, len(srs)):
            if pd.isna(smoothed_srs.iloc[i-1]):
                smoothed_srs.iloc[i] = srs.iloc[i]
            else:
                smoothed_srs.iloc[i] = smoothed_srs.iloc[i-1] + smoothing_f * (srs.iloc[i] - smoothed_srs.iloc[i-1])

        return smoothed_srs

    def NormalizeSmoothSrs(self, series, window_length, smoothing_f):
        lowest = series.rolling(window_length).min()
        highest_range = series.rolling(window_length).max() - lowest
        normalized_series = series.copy()
        if (highest_range > 0).any():
            normalized_series = (series - lowest) / highest_range * 100
        else:
            normalized_series = pd.Series([pd.NA] * len(series), index=series.index)
        normalized_series = normalized_series.ffill()
        smoothed_series = self.SmoothSrs(normalized_series, smoothing_f)
        return smoothed_series
    
    def calc_STC(self, data: pd.DataFrame, stc_length = 80, fast_length = 27, slow_length = 50, AAA = 0.5) -> pd.Series:
        macd_diff = self.MacdDiff(data["close"], fast_length, slow_length)
        normalized_macd = self.NormalizeSmoothSrs(macd_diff, stc_length, AAA)
        final_stc = self.NormalizeSmoothSrs(normalized_macd, stc_length, AAA)
        return final_stc
    
    def calc_signal(self, data: pd.DataFrame) -> pd.Series:
        key_value = 2
        sell_atr_period = 1
        _, sell_signals = self.calc_UTBot(data, key_value, sell_atr_period)
        buy_signals, _ = self.calc_UTBot(data, key_value, self.buy_atr_period)
        sell_signals = sell_signals * -1
        data["UT_signals"] = sell_signals + buy_signals
        for i in range(1, len(data)):
            if i >= START_DATE_INDEX and data["UT_signals"].iloc[i] == 1:
                data.iloc[i, data.columns.get_loc('UT_signals')] = ActionType.BUY
            elif i >= START_DATE_INDEX and data["UT_signals"].iloc[i] == -1:
                data.iloc[i, data.columns.get_loc('UT_signals')] = ActionType.SELL
            else:
                data.iloc[i, data.columns.get_loc('UT_signals')] = StrategySignal.DO_NOTHING
        stc_length = 80
        fast_length = 27
        slow_length = 50
        AAA = 0.5
        data["STC"] = self.calc_STC(data, stc_length, fast_length, slow_length, AAA)
        data["strategy_signal"] = StrategySignal.DO_NOTHING
        stc_oversolded_mark = 25
        stc_overbought_mark = 75
        for i in range(1, len(data)):
            if data["UT_signals"].iloc[i] == ActionType.BUY:
                if data["STC"].iloc[i] > data["STC"].iloc[i - 1] and data["STC"].iloc[i] < stc_oversolded_mark:
                    data.iloc[i, data.columns.get_loc('strategy_signal')] = ActionType.BUY
            elif data["UT_signals"].iloc[i] == ActionType.SELL:
                if data["STC"].iloc[i] < data["STC"].iloc[i - 1] and data["STC"].iloc[i] > stc_overbought_mark:
                    data.iloc[i, data.columns.get_loc('strategy_signal')] = ActionType.SELL
        return data["strategy_signal"]

        
        
        
        