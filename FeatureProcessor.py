import os

import numpy as np
import pandas as pd
import talib
from DataCollector import DataCollector

class FeatureProcessor:
    def __init__(self, data_directory='data'):
        self.data_directory = data_directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def process(self, data):
        try:
            # Extract the OHLCV DataFrame from the data dictionary
            df = data['ohlcv']

            # Convert necessary columns to numeric types (float) to ensure proper calculations
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)

            # Calculate price change
            df['price_change'] = ((df['close'] - df['open']) / df['open']) * 100

            # Calculate technical indicators using the full DataFrame
            df['RSI'] = talib.RSI(df['close'], timeperiod=14)
            df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
            df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
            df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
            df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
            df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)
            df['support_level'] = df['low'].min()
            df['resistance_level'] = df['high'].max()

            # Extract the latest row to use as the features
            features = df.iloc[-1].to_dict()

            # Add additional features from the original data (including order book)
            features['last_price'] = float(data['last_price'])
            features['bid'] = float(data['bid']) if data['bid'] else None
            features['ask'] = float(data['ask']) if data['ask'] else None
            features['high'] = float(data['high'])
            features['low'] = float(data['low'])
            features['volume'] = float(data['volume'])

            # Add order book features
            if 'order_book' in data:
                order_book = data['order_book']
                features['top_bid'] = float(order_book['bids'][0][0]) if order_book['bids'] else None
                features['top_ask'] = float(order_book['asks'][0][0]) if order_book['asks'] else None
                features['bid_ask_spread'] = features['top_ask'] - features['top_bid'] if features['top_bid'] and features['top_ask'] else None
                features['bid_volume'] = sum(float(bid[1]) for bid in order_book['bids'])  # Sum of bid volumes
                features['ask_volume'] = sum(float(ask[1]) for ask in order_book['asks'])  # Sum of ask volumes

            # Convert the features dictionary to a DataFrame
            features_df = pd.DataFrame([features])

            # Save the processed data to a CSV file, overwriting any existing file
            self.save_to_csv(features_df)

            return features
        except Exception as e:
            print(f"Error processing features: {e}")
            return None

    def calculate_price_change(self, current_price, reference_price):
        try:
            price_change = (current_price - reference_price) / reference_price * 100
            return price_change
        except Exception as e:
            print(f"Error calculating price change: {e}")
            return 0.0

    def calculate_volume_change(self, volume):
        try:
            reference_volume = 1000  # Static reference for demonstration
            volume_change = (volume - reference_volume) / reference_volume * 100
            return volume_change
        except Exception as e:
            print(f"Error calculating volume change: {e}")
            return 0.0

    def calculate_rsi(self, prices, period=14):
        try:
            rsi = talib.RSI(prices, timeperiod=period)
            return rsi.iloc[-1] if len(rsi) > 0 else None  # Get the most recent RSI value
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return None

    def calculate_sma(self, prices, window):
        try:
            sma = talib.SMA(prices, timeperiod=window)
            return sma.iloc[-1] if len(sma) > 0 else None  # Get the most recent SMA value
        except Exception as e:
            print(f"Error calculating SMA: {e}")
            return prices.mean()

    def calculate_ema(self, prices, window):
        try:
            ema = talib.EMA(prices, timeperiod=window)
            return ema.iloc[-1] if len(ema) > 0 else None  # Get the most recent EMA value
        except Exception as e:
            print(f"Error calculating EMA: {e}")
            return prices.mean()

    def calculate_macd(self, prices, fastperiod=12, slowperiod=26, signalperiod=9):
        try:
            macd, macd_signal, macd_hist = talib.MACD(prices, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
            return macd.iloc[-1], macd_signal.iloc[-1], macd_hist.iloc[-1]  # Get the most recent values
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return None, None, None

    def calculate_bollinger_bands(self, prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        try:
            upper_band, middle_band, lower_band = talib.BBANDS(prices, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
            return upper_band.iloc[-1], middle_band.iloc[-1], lower_band.iloc[-1]  # Get the most recent values
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            return None, None, None

    def calculate_support_resistance(self, prices):
        try:
            # Simplified support/resistance calculation (you may use more advanced methods)
            support_level = prices.min()
            resistance_level = prices.max()
            return support_level, resistance_level
        except Exception as e:
            print(f"Error calculating support/resistance levels: {e}")
            return None, None

    def save_to_csv(self, df):
        try:
            file_path = os.path.join(self.data_directory, 'processed_features.csv')

            # Always overwrite the existing file
            df.to_csv(file_path, mode='w', header=True, index=False)

            print(f"Features saved to {file_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

# Example usage:
# Example usage:
if __name__ == "__main__":
    api_key = 'your_binance_api_key'
    api_secret = 'your_binance_api_secret'

    data_collector = DataCollector(api_key, api_secret, timeframe='5m')
    market_data = data_collector.collect_data()

    if market_data is not None:
        feature_processor = FeatureProcessor()
        features = feature_processor.process(market_data)
        if features:
            print("Processed Features:")
            print(features)