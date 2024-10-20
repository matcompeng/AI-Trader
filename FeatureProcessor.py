import os
import pandas as pd
import talib
from datetime import datetime, timedelta
from DataCollector import DataCollector
import json


class FeatureProcessor:
    def __init__(self, data_directory='data', intervals=None, trading_interval=None, dip_interval=None, dip_flag=None):
        self.data_directory = data_directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        self.intervals = intervals if intervals is not None else []
        self.dip_interval = dip_interval
        self.trading_interval = trading_interval
        self.dip_flag = dip_flag

    def process(self, data):
        try:
            all_features = {}

            for interval, market_data in data.items():
                # Extract the OHLCV DataFrame from the data dictionary
                df = market_data['ohlcv']

                # Convert necessary columns to numeric types (float) to ensure proper calculations
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Calculate price change
                df['price_change'] = ((df['close'] - df['open']) / df['open']) * 100

                # Calculate RSI
                df['RSI'] = talib.RSI(df['close'], timeperiod=14)

                # Calculate three SMA and three EMA with periods 7, 25, and 100
                df['SMA_7'] = talib.SMA(df['close'], timeperiod=7)
                df['SMA_25'] = talib.SMA(df['close'], timeperiod=25)
                df['SMA_100'] = talib.SMA(df['close'], timeperiod=100)
                df['EMA_7'] = talib.EMA(df['close'], timeperiod=7)
                df['EMA_25'] = talib.EMA(df['close'], timeperiod=25)
                df['EMA_100'] = talib.EMA(df['close'], timeperiod=100)

                df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
                df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)
                df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

                # Calculate Stochastic RSI using the existing RSI calculation
                df['stoch_rsi_k'], df['stoch_rsi_d'] = self.calculate_stoch_rsi(df['close'])

                # Calculate ATR
                df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

                # Calculate OBV (On-Balance Volume)
                df['OBV'] = talib.OBV(df['close'], df['volume'])

                # Calculate VWAP manually with a length period of 14
                df['VWAP'] = self.calculate_vwap(df, period=14)

                # Add this line after other TA-Lib calculations (e.g., RSI, SMA, etc.)
                df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)

                # Calculate Rate of Change (ROC)
                df['ROC'] = talib.ROC(df['close'], timeperiod=9)

                # Calculate support and resistance levels based on current time for the specified intervals
                # if interval in self.intervals:
                #     support_level, resistance_level = self.calculate_support_resistance(df, interval)
                #     df['support_level'] = support_level
                #     df['resistance_level'] = resistance_level
                # else:
                #     df['support_level'] = None
                #     df['resistance_level'] = None

                # Extract the latest row to use as the features
                features = df.iloc[-1].to_dict()

                # Add additional features from the original data (including order book)
                features['last_price'] = float(market_data['last_price'])
                features['bid'] = float(market_data['bid']) if market_data['bid'] else None
                features['ask'] = float(market_data['ask']) if market_data['ask'] else None
                features['high'] = float(market_data['high'])
                features['low'] = float(market_data['low'])
                features['volume'] = float(market_data['volume'])

                # Add order book features
                if 'order_book' in market_data:
                    order_book = market_data['order_book']
                    features['top_bid'] = float(order_book['bids'][0][0]) if order_book['bids'] else None
                    features['top_ask'] = float(order_book['asks'][0][0]) if order_book['asks'] else None
                    features['bid_ask_spread'] = features['top_ask'] - features['top_bid'] if features['top_bid'] and features['top_ask'] else None
                    features['bid_volume'] = sum(float(bid[1]) for bid in order_book['bids'])  # Sum of bid volumes
                    features['ask_volume'] = sum(float(ask[1]) for ask in order_book['asks'])  # Sum of ask volumes

                # Store the features for this interval
                all_features[interval] = features

            # Convert the features dictionary to a DataFrame and save it
            features_df = pd.DataFrame.from_dict(all_features, orient='index')
            self.save_to_csv(features_df)

            return all_features
        except Exception as e:
            print(f"Error processing features: {e}")
            raise Exception("Failed to Process Features")


    def calculate_vwap(self, df, period=14):
        # Calculate the typical price (TP) for each period
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Calculate the cumulative TPV and volume over the specified period
        df['cumulative_tpv'] = (df['typical_price'] * df['volume']).rolling(window=period).sum()
        df['cumulative_volume'] = df['volume'].rolling(window=period).sum()

        # Calculate VWAP for the period
        df['VWAP'] = df['cumulative_tpv'] / df['cumulative_volume']

        # Return the last calculated VWAP value
        return df['VWAP']

    def calculate_support_resistance(self, df, interval):
        now = datetime.utcnow()
        if interval == '1m':
            start_time = now.replace(minute=(now.minute // 1) * 1, second=0, microsecond=0)
            end_time = start_time + timedelta(minutes=1)
        elif interval == '5m':
            start_time = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
            end_time = start_time + timedelta(minutes=5)
        elif interval == '15m':
            start_time = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            end_time = start_time + timedelta(minutes=15)
        elif interval == '30m':
            start_time = now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0)
            end_time = start_time + timedelta(minutes=30)
        elif interval == '1h':
            start_time = now.replace(hour=(now.hour // 1) * 1, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=1)
        elif interval == '2h':
            start_time = now.replace(hour=(now.hour // 2) * 2, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=2)
        elif interval == '4h':
            start_time = now.replace(hour=(now.hour // 4) * 4, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=4)
        elif interval == '8h':
            start_time = now.replace(hour=(now.hour // 8) * 8, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=8)
        elif interval == '12h':
            start_time = now.replace(hour=(now.hour // 12) * 12, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(hours=12)
        elif interval == '1d':
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = start_time + timedelta(days=1)
        else:
            return None, None

        # Filter data for the current time interval
        df_interval = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]

        if not df_interval.empty:
            support_level = df_interval['low'].min()
            resistance_level = df_interval['high'].max()
            return support_level, resistance_level
        else:
            return None, None

    def calculate_stoch_rsi(self, close_prices, timeperiod=14, smooth_k=3, smooth_d=3):
        rsi = talib.RSI(close_prices, timeperiod)
        stoch_rsi_k = (rsi - rsi.rolling(window=timeperiod).min()) / (rsi.rolling(window=timeperiod).max() - rsi.rolling(window=timeperiod).min()) * 100
        stoch_rsi_k = stoch_rsi_k.rolling(window=smooth_k).mean()  # Smooth %K
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()  # Smooth %D
        return stoch_rsi_k, stoch_rsi_d

    def save_to_csv(self, df):
        try:
            file_path = os.path.join(self.data_directory, 'processed_features.csv')

            # Always overwrite the existing file
            df.to_csv(file_path, mode='w', header=True, index=True)

            print(f"Features saved to {file_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")


    def get_stable_historical_data(self):
        """
        Load historical context data from the JSON file.
        """
        historical_file = os.path.join(self.data_directory, f'{self.trading_interval}_stable_historical_context.json')

        if os.path.exists(historical_file):
            with open(historical_file, 'r') as file:
                historical_data = json.load(file)
            return historical_data
        return []

    def get_dip_historical_data(self):
        """
        Load historical context data from the JSON file.
        """
        historical_file = os.path.join(self.data_directory, f'{self.dip_interval}_dip_historical_context.json')

        if os.path.exists(historical_file):
            with open(historical_file, 'r') as file:
                historical_data = json.load(file)
            return historical_data
        return []


# Example usage:
if __name__ == "__main__":
    api_key = 'your_binance_api_key'
    api_secret = 'your_binance_api_secret'

    data_collector = DataCollector(api_key, api_secret, intervals=['5m', '15m', '1h','4h','8h','12h'])
    market_data = data_collector.collect_data()

    if market_data is not None:
        feature_processor = FeatureProcessor()
        all_features = feature_processor.process(market_data)

        if all_features:
            print("Processed Features:")
            print(all_features)
