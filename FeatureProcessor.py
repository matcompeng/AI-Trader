import os
import pandas as pd
import talib
from datetime import datetime, timedelta
from DataCollector import DataCollector
import numpy as np
import json


class FeatureProcessor:
    def __init__(self, data_directory='data', intervals=None, trading_interval=None, loss_interval=None, dip_interval=None, dip_flag=None, orderbook_threshold=None, volume_threshold=None):
        self.data_directory = data_directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        self.intervals = intervals if intervals is not None else []
        self.dip_interval = dip_interval
        self.trading_interval = trading_interval
        self.dip_flag = dip_flag
        self.orderbook_threshold = orderbook_threshold
        self.atr_value_1m = None
        self.price_value_1m = None
        self.volume_threshold = volume_threshold
        self.loss_interval  = loss_interval

    def process(self, data):
        try:
            all_features = {
                'latest': {},
                'history': {}
            }

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
                df['RSI_6'] = talib.RSI(df['close'], timeperiod=6)
                df['RSI_14'] = talib.RSI(df['close'], timeperiod=14)
                df['RSI_24'] = talib.RSI(df['close'], timeperiod=24)

                # Calculate three SMA and three EMA with periods 7, 25, and 100
                df['SMA_7'] = talib.SMA(df['close'], timeperiod=7)
                df['SMA_25'] = talib.SMA(df['close'], timeperiod=25)
                df['SMA_100'] = talib.SMA(df['close'], timeperiod=100)
                df['EMA_7'] = talib.EMA(df['close'], timeperiod=7)
                df['EMA_25'] = talib.EMA(df['close'], timeperiod=25)
                df['EMA_100'] = talib.EMA(df['close'], timeperiod=100)

                # Calculate MACD
                df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26,
                                                                            signalperiod=9)

                # Calculate Bollinger Bands
                df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20)

                # Calculate ADX
                df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

                # Calculate Stochastic RSI
                df['stoch_rsi_k'], df['stoch_rsi_d'] = self.calculate_stoch_rsi(df['close'])

                # Calculate ATR
                df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

                # Calculate OBV (On-Balance Volume)
                df['OBV'] = talib.OBV(df['close'], df['volume'])

                # Calculate VWAP
                df['VWAP'] = self.calculate_vwap(df, period=14)

                # Calculate SAR
                df['SAR'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)

                # Calculate Rate of Change (ROC)
                df['ROC'] = talib.ROC(df['close'], timeperiod=9)

                # Store ATR and price values for 1m interval for dynamic gap threshold calculation
                if interval == '1m':
                    self.atr_value_1m = df['ATR'].iloc[-1]
                    self.price_value_1m = df['close'].iloc[-1]

                # Calculate support and resistance levels based on the order book
                support_level, resistance_level, average_support, average_resistance = self.calculate_support_resistance_from_orderbook(
                    market_data['order_book'])
                df['support_level'] = support_level
                df['resistance_level'] = resistance_level
                df['average_support'] = average_support
                df['average_resistance'] = average_resistance

                # Store the historical data for the interval
                all_features['history'][interval] = df

                # Extract the latest row as the features for this interval
                latest_features = df.iloc[-1].to_dict()

                # Add additional features from the original data (including order book)
                latest_features['last_price'] = float(market_data['last_price'])
                latest_features['bid'] = float(market_data['bid']) if market_data['bid'] else None
                latest_features['ask'] = float(market_data['ask']) if market_data['ask'] else None
                latest_features['high'] = float(market_data['high'])
                latest_features['low'] = float(market_data['low'])
                latest_features['volume'] = float(market_data['volume'])

                # Add order book features
                if 'order_book' in market_data:
                    order_book = market_data['order_book']
                    latest_features['top_bid'] = float(order_book['bids'][0][0]) if order_book['bids'] else None
                    latest_features['top_ask'] = float(order_book['asks'][0][0]) if order_book['asks'] else None
                    latest_features['bid_ask_spread'] = latest_features['top_ask'] - latest_features['top_bid'] if \
                    latest_features['top_bid'] and latest_features['top_ask'] else None
                    latest_features['bid_volume'] = sum(
                        float(bid[1]) for bid in order_book['bids'])  # Sum of bid volumes
                    latest_features['ask_volume'] = sum(
                        float(ask[1]) for ask in order_book['asks'])  # Sum of ask volumes
                    if interval == self.loss_interval :
                        latest_features['stop_loss'] = self.suggest_stop_loss_based_on_order_book(order_book)
                    if interval == '1m' :
                        latest_features['stop_loss_scalping'] = self.stop_loss_based_on_order_book_for_scalping(order_book)

                # Store the latest features for this interval
                all_features['latest'][interval] = latest_features

            # Convert the latest features dictionary to a DataFrame and save it
            latest_features_df = pd.DataFrame.from_dict(all_features['latest'], orient='index')
            self.save_to_csv(latest_features_df)

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

    def calculate_support_resistance_from_orderbook(self, order_book):
        """
        Identify potential support and resistance levels from the order book.

        :param order_book: The order book containing current market bids and asks.
        :param threshold: The volume threshold to consider for identifying significant support or resistance
        :return: support_level, resistance_level (both as floats)
        """
        support_levels = []
        resistance_levels = []

        # Analyze bids for support levels
        for bid in order_book['bids']:
            price = float(bid[0])
            volume = float(bid[1])
            if volume >= self.orderbook_threshold:
                support_levels.append(price)

        # Analyze asks for resistance levels
        for ask in order_book['asks']:
            price = float(ask[0])
            volume = float(ask[1])
            if volume >= self.orderbook_threshold:
                resistance_levels.append(price)

        # For stop-loss: Use min for support and max for resistance
        support_level = min(support_levels) if support_levels else None
        resistance_level = max(resistance_levels) if resistance_levels else None

        # For take-profit: Use average of support and resistance levels
        average_support = sum(support_levels) / len(support_levels) if support_levels else None
        average_resistance = sum(resistance_levels) / len(resistance_levels) if resistance_levels else None

        # Return min/max for stop-loss and average for take-profit
        return support_level,resistance_level, average_support, average_resistance



    def calculate_support_resistance_from_interval(self, df, interval):
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

            # print(f"Features saved to {file_path}")
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

    def calculate_gap_threshold(self):
        """
        Calculates the gap threshold percentage based on the ATR of the '1m' interval.
        :return: Dynamic gap threshold percentage.
        """
        dynamic_gap_threshold = (self.atr_value_1m / self.price_value_1m) * 100
        # print(f"dynamic_gap_threshold: {dynamic_gap_threshold}")
        return dynamic_gap_threshold

    def suggest_stop_loss_based_on_order_book(self, order_book, max_iterations=100, increment_multiplier=1.2):
        """
        Suggests a stop-loss price based on gaps in the order book bids.

        :param order_book: Dictionary containing bids and asks. The bids should be a list of [price, volume].
        :param max_iterations: Maximum number of iterations to increment the volume threshold.
        :param increment_multiplier: Multiplier to increment the volume threshold in each iteration.
        :return: Suggested stop-loss price, or None if no significant gap is found.
        """
        # Extract the bid prices and volumes, and convert them to floats
        bids = [[float(price), float(volume)] for price, volume in order_book['bids']]

        if not bids:
            print("Order book bids are empty.")
            return None

        # Extract all volumes from the bids
        volumes = [bid[1] for bid in bids]

        # Calculate initial volume threshold using the 75th percentile
        volume_threshold = np.percentile(volumes, 75)

        iteration = 0
        significant_gaps = []

        # Iterate with incrementally increasing volume thresholds until we find significant gaps or reach max iterations
        while iteration < max_iterations:
            # Filter bids by the current volume threshold, and sort them in descending order of price
            filtered_bids = sorted([bid for bid in bids if bid[1] >= volume_threshold], key=lambda x: x[0],
                                   reverse=True)

            # Clear significant gaps list for the current iteration
            significant_gaps.clear()

            # Iterate through the filtered bids and look for significant gaps
            for i in range(len(filtered_bids) - 1):
                current_price = filtered_bids[i][0]
                next_price = filtered_bids[i + 1][0]

                # Calculate the percentage gap between consecutive bids
                gap_percentage = ((current_price - next_price) / current_price) * 100
                # print(f"gap_percentage: {gap_percentage:.2f}")

                # If the gap exceeds the calculated gap threshold, add it to the list
                if gap_percentage >= self.calculate_gap_threshold():
                    significant_gaps.append((current_price, next_price))

            # Check if we found any significant gaps
            if significant_gaps:
                break  # Stop if at least one significant gap is found

            # Increment the volume threshold for the next iteration
            volume_threshold *= increment_multiplier
            iteration += 1
            # print(f"Iteration {iteration}: Increased volume threshold to {volume_threshold:.2f}")

        # Determine the stop-loss value based on the significant gaps found
        if len(significant_gaps) >= 2:
            second_gap = significant_gaps[1]
            suggested_stop_loss = second_gap[1] - (2.0 * (second_gap[0] - second_gap[1]))
        elif len(significant_gaps) == 1:
            first_gap = significant_gaps[0]
            suggested_stop_loss = first_gap[1] - (2.0 * (first_gap[0] - first_gap[1]))
        else:
            # If no significant gaps are found after all iterations, use a default stop-loss strategy
            print("No significant gaps found after max iterations.")
            suggested_stop_loss = None

        if suggested_stop_loss is None:
            print("Stop loss value is not available.")
        else:
            print(f"|||Concurrent Stop Loss: {suggested_stop_loss:.4f}|||")

        return suggested_stop_loss

    def stop_loss_based_on_order_book_for_scalping(self, order_book, max_iterations=100, increment_multiplier=1.2):
        """
        Suggests a stop-loss price for scalping based on the first significant gap in the order book bids.

        :param order_book: Dictionary containing bids and asks. The bids should be a list of [price, volume].
        :param max_iterations: Maximum number of iterations to increment the volume threshold.
        :param increment_multiplier: Multiplier to increment the volume threshold in each iteration.
        :return: Suggested stop-loss price, or None if no significant gap is found.
        """
        # Extract the bid prices and volumes, and convert them to floats
        # Limiting to the first 100 bids
        bids = [[float(price), float(volume)] for price, volume in order_book['bids']]

        if not bids:
            print("Order book bids are empty.")
            return None

        # Extract all volumes from the bids
        volumes = [bid[1] for bid in bids]

        # Calculate initial volume threshold using the 75th percentile
        volume_threshold = np.percentile(volumes, 75)

        iteration = 0
        significant_gap = None

        # Iterate with incrementally increasing volume thresholds until we find the first significant gap or reach max iterations
        while iteration < max_iterations:
            # Filter bids by the current volume threshold, and sort them in descending order of price
            filtered_bids = sorted([bid for bid in bids if bid[1] >= volume_threshold], key=lambda x: x[0],
                                   reverse=True)

            # Iterate through the filtered bids and look for the first significant gap
            for i in range(len(filtered_bids) - 1):
                current_price = filtered_bids[i][0]
                next_price = filtered_bids[i + 1][0]

                # Calculate the percentage gap between consecutive bids
                gap_percentage = ((current_price - next_price) / current_price) * 100

                # If the gap exceeds the calculated gap threshold, set it as the first significant gap
                if gap_percentage >= self.calculate_gap_threshold():
                    significant_gap = (current_price, next_price)
                    break

            # Stop iteration if a significant gap is found
            if significant_gap:
                break

            # Increment the volume threshold for the next iteration
            volume_threshold *= increment_multiplier
            iteration += 1

        # Determine the stop-loss value based on the first significant gap found
        if significant_gap:
            suggested_stop_loss = significant_gap[0] - (1.0 * (significant_gap[0] - significant_gap[1]))
        else:
            # If no significant gap is found after all iterations, use a default stop-loss strategy
            print("No significant gaps found after max iterations.")
            suggested_stop_loss = None

        if suggested_stop_loss is None:
            print("Scalping Stop loss value is not available.")
        else:
            print(f"|||Scalping Stop Loss: {suggested_stop_loss:.4f}|||")

        return suggested_stop_loss


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
