import os
import pandas as pd
from binance.client import Client
import time

class DataCollector:
    def __init__(self, api_key, api_secret, symbol, intervals=None, limit=1000, data_directory='data', max_retries=3, retry_delay=5):
        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.intervals = intervals  # List of intervals to fetch data for
        self.limit = limit  # Limit for the number of data points to fetch
        self.data_directory = data_directory
        self.max_retries = max_retries  # Maximum number of retries
        self.retry_delay = retry_delay  # Delay in seconds between retries

        # Ensure the data directory exists
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def fetch_ohlcv(self, interval):
        for attempt in range(self.max_retries):
            try:
                # Fetch OHLCV data for the specified interval using the limit
                ohlcv = self.client.get_klines(symbol=self.symbol, interval=interval, limit=self.limit)
                return ohlcv
            except Exception as e:
                print(f"Error fetching OHLCV data for interval {interval}: {e}")
                if attempt + 1 < self.max_retries:
                    print(f"Retrying... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to fetch data for interval {interval} after {self.max_retries} attempts.")
                    return []

    def collect_data(self):
        try:
            data = {}

            for interval in self.intervals:
                ohlcv = self.fetch_ohlcv(interval)
                if not ohlcv:
                    continue

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                  'close_time', 'quote_asset_volume', 'number_of_trades',
                                                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                # Use the last row for the most recent data point
                ticker = df.iloc[-1]

                # Fetch order book data (bids, asks)
                order_book = self.client.get_order_book(symbol=self.symbol, limit=1000)

                # Collect all relevant data including the order book
                data[interval] = {
                    'timestamp': ticker['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'last_price': float(ticker['close']),
                    'bid': float(order_book['bids'][0][0]) if len(order_book['bids']) > 0 else None,
                    'ask': float(order_book['asks'][0][0]) if len(order_book['asks']) > 0 else None,
                    'high': float(ticker['high']),
                    'low': float(ticker['low']),
                    'volume': float(ticker['volume']),
                    'order_book': order_book,  # Include the full order book
                    'ohlcv': df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    # Include the full DataFrame for further processing
                }

            # Save the collected data to a CSV file
            self.save_to_csv(data)

            return data
        except Exception as e:
            print(f"Error collecting data: {e}")
            return None

    def save_to_csv(self, data):
        try:
            for interval, data_dict in data.items():
                # Convert the OHLCV DataFrame to a CSV file
                df = data_dict['ohlcv'].copy()

                # Save the order book as a separate CSV file for review (if needed)
                order_book_df = pd.DataFrame({
                    'bid_price': [bid[0] for bid in data_dict['order_book']['bids']],
                    'bid_volume': [bid[1] for bid in data_dict['order_book']['bids']],
                    'ask_price': [ask[0] for ask in data_dict['order_book']['asks']],
                    'ask_volume': [ask[1] for ask in data_dict['order_book']['asks']]
                })

                df['top_bid'] = data_dict['bid']
                df['top_ask'] = data_dict['ask']

                file_path = os.path.join(self.data_directory, f'collected_data_{interval}.csv')
                df.to_csv(file_path, mode='w', header=True, index=False)

                order_book_path = os.path.join(self.data_directory, f'order_book_data_{interval}.csv')
                order_book_df.to_csv(order_book_path, mode='w', header=True, index=False)

                # print(f"Collected data saved to {file_path}")
                # print(f"Order book data saved to {order_book_path}")
            print("Collected Data and Order Book Saved to CSV Files")
        except Exception as e:
            print(f"Error saving to CSV: {e}")

# Example usage:
if __name__ == "__main__":
    api_key = 'your_binance_api_key'
    api_secret = 'your_binance_api_secret'

    data_collector = DataCollector(api_key, api_secret, intervals=['5m', '15m', '1h', '2h',], symbol='BNBUSDT')
    market_data = data_collector.collect_data()

    if market_data is not None:
        print("Collected market data:")
        print(market_data)