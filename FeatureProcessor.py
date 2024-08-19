import os
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

                # Calculate ADX
                df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

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
            return None

    def save_to_csv(self, df):
        try:
            file_path = os.path.join(self.data_directory, 'processed_features.csv')

            # Always overwrite the existing file
            df.to_csv(file_path, mode='w', header=True, index=True)

            print(f"Features saved to {file_path}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

# Example usage:
if __name__ == "__main__":
    api_key = 'your_binance_api_key'
    api_secret = 'your_binance_api_secret'

    data_collector = DataCollector(api_key, api_secret, intervals=['1m', '5m', '15m', '1h'])
    market_data = data_collector.collect_data()

    if market_data is not None:
        feature_processor = FeatureProcessor()
        all_features = feature_processor.process(market_data)

        if all_features:
            print("Processed Features:")
            print(all_features)