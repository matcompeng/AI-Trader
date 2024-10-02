import json
import os
import time
import schedule
import logging
import traceback
import pandas as pd
from datetime import datetime, timedelta
from DataCollector import DataCollector
from FeatureProcessor import FeatureProcessor
from ChatGPTClient import ChatGPTClient
from Predictor import Predictor
from DecisionMaker import DecisionMaker
from Trader import Trader
from Notifier import Notifier
import csv
import threading

# TODO : make base dip amount to be used in dynamic function under decision manager
# TODO : Study ranging MIN_STABLE_INTERVALS with FearGreedIndex value
# TODO : study ranging RISK_TOLERANCE with FearGreedIndex value
# Bot Configurations ------------------------------------------------------------------------------
FEATURES_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '1d']
COIN = 'BNB'                    # Select Cryptocurrency.
TRADING_PAIR = 'BNBUSDT'        # Select Cryptocurrency Trading Pair
TRADING_INTERVAL = '15m'        # Select The Interval For Stable 'Buy' Trading And Gathering Historical Context.
PROFIT_INTERVAL = '1h'          # Select The Interval For Take Profit Calculations.
LOSS_INTERVAL = '1h'            # Select The Interval For Stop Loose Calculations.
SR_INTERVAL = '15m'             # Select The Interval That Trader Define Support and Resistance Levels.
DIP_INTERVAL = '1h'             # Select The Interval For Buying a Dip.
POSITION_CYCLE = [60, 300]      # Time periods in Seconds To Check Positions [Short,Long].
POSITION_TIMEOUT = 24           # Set The Timeout In Hours for Position.
PREDICTION_CYCLE = 15           # Time in Minutes to Run the Stable Prediction bot cycle.
DIP_CYCLE = 60                  # Time in Minutes to Run the Dip Historical Context Process.
INTERVAL_BANDWIDTH = '5m'       # Define The Interval To calculate Prediction Bandwidth.
PREDICT_BANDWIDTH = 0.45        # Define Minimum Bandwidth % to Activate Trading.
BASE_TAKE_PROFIT = 0.35         # Define Base Take Profit Percentage %.
BASE_STOP_LOSS = 0.35           # Define Base Stop Loose  Percentage %.
CAPITAL_AMOUNT = 30500          # Your Capital Investment.
RISK_TOLERANCE = 0.15           # The Portion Amount you want to take risk of capital for each Buying position.
AMOUNT_RSI_INTERVAL = '5m'      # Interval To get its RSI for Buying Amount Calculations Function.
AMOUNT_ATR_INTERVAL = '30m'     # Interval To get its ATR for Buying Amount Calculations Function.
USDT_DIP_AMOUNT = 2500          # Amount of Currency For Buying a Dip.
MIN_STABLE_INTERVALS = 5        # Set The Minimum Stable Intervals For Market Stable Condition.
TRAILING_POSITIONS_COUNT = 1    # Define The Minimum Count For Stable Positions To start Trailing Check.
# TRAILING_PERCENT = 0.25         # Set The Minimum % To Activate Trailing Stop Process.
# TRAILING_GAIN_REVERSE = 0.20    # Set the Sell Threshold % for Stable Portfolio Gain Reversal (Trailing Stop).
CHECK_POSITIONS_ON_BUY = True   # Set True If You Need Bot Manager Check The Positions During Buy Cycle.
# -------------------------------------------------------------------------------------------------

# Create the data directory if it doesn't exist
data_directory = 'data'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

# Configure logging to save in the data directory
log_file_path = os.path.join(data_directory, 'bot_manager.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PositionManager:
    def __init__(self, positions_file='positions.json'):
        self.positions_file = os.path.join(data_directory, positions_file)
        self.positions = self.load_positions()

    def load_positions(self):
        if os.path.exists(self.positions_file):
            with open(self.positions_file, 'r') as file:
                return json.load(file)
        return {}

    def save_positions(self):
        with open(self.positions_file, 'w') as file:
            json.dump(self.positions, file, indent=4)

    def add_position(self, position_id, entry_price, amount, dip_flag):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add timestamp
        self.positions[position_id] = {
            'entry_price': entry_price,
            'amount': amount,
            'dip': dip_flag,
            'timestamp': timestamp  # Save the timestamp
        }
        self.save_positions()

    def remove_position(self, position_id):
        if position_id in self.positions:
            del self.positions[position_id]
            self.save_positions()

    def get_positions(self):
        return self.positions


class BotManager:
    def __init__(self):
        # Initialize components
        api_key = 'your_binance_api_key'
        api_secret = 'your_binance_api_secret'

        self.data_collector = DataCollector(api_key, api_secret, intervals=FEATURES_INTERVALS, symbol=TRADING_PAIR)
        self.feature_processor = FeatureProcessor(intervals=FEATURES_INTERVALS, trading_interval=TRADING_INTERVAL, dip_interval=DIP_INTERVAL)
        self.chatgpt_client = ChatGPTClient()
        self.predictor = Predictor(self.chatgpt_client, coin=COIN, sr_interval=SR_INTERVAL, bot_manager=self)
        self.decision_maker = DecisionMaker(base_take_profit=BASE_TAKE_PROFIT, base_stop_loss=BASE_STOP_LOSS,
                                            profit_interval=PROFIT_INTERVAL, loose_interval=LOSS_INTERVAL,
                                            dip_interval=DIP_INTERVAL, risk_tolerance=RISK_TOLERANCE,
                                            amount_atr_interval=AMOUNT_ATR_INTERVAL,
                                            amount_rsi_interval=AMOUNT_RSI_INTERVAL,
                                            min_stable_intervals=MIN_STABLE_INTERVALS)
        self.trader = Trader(symbol=TRADING_PAIR)  # Initialize the Trader class
        self.notifier = Notifier()
        self.position_manager = PositionManager()
        self.initialize_position_period()

    def log_time(self, process_name, start_time):
        end_time = time.time()
        duration = end_time - start_time
        log_message = f"{process_name} took {duration:.2f} seconds."

        # Log to file
        logging.info(log_message)
        # Print to console
        print(log_message)

    def save_error_to_csv(self, error_message):
        try:
            # Capture the current time for the error log
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Capture the full traceback
            full_traceback = traceback.format_exc()

            # Prepare the error data
            error_data = {
                'timestamp': [timestamp],
                'error_message': [error_message],
                'traceback': [full_traceback]
            }

            # Convert to DataFrame
            df = pd.DataFrame(error_data)

            # File path for error log
            file_path = os.path.join(data_directory, 'error_logs.csv')

            # Append to the CSV file
            df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            print(f"Error details saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save error details to CSV: {e}")

    def convert_usdt_to_crypto(self, current_price, usdt_amount):
        # Convert USDT amount to cryptocurrency amount and round to 5 decimal places
        crypto_amount = round(usdt_amount / current_price, 5)
        return crypto_amount

    def calculate_prediction_bandwidth(self, all_features):
        """
        Calculate the percentage price change between the upper and lower Bollinger Bands of the PREDICTOR_INTERVAL interval.
        """
        upper_band = all_features[INTERVAL_BANDWIDTH].get('upper_band')
        lower_band = all_features[INTERVAL_BANDWIDTH].get('lower_band')

        if upper_band and lower_band:
            price_change = ((upper_band - lower_band) / lower_band) * 100
            return price_change

    def calculate_gain_loose(self,entry_price, current_price):
        gain_loose = ((current_price - entry_price) / entry_price) * 100
        return gain_loose

    def log_sold_position(self, position_id, trade_type, entry_price, sold_price, profit_usdt, gain_loose):
        """
        Log the details of a sold position to a CSV file.
        :param trade_type:
        :param profit_usdt:
        :param position_id: The ID of the position.
        :param entry_price: The entry price of the position.
        :param sold_price: The price at which the position was sold.
        :param gain_loose: The gain or loss percentage.
        """
        log_file_path = os.path.join(data_directory, 'trading_log.csv')

        # Prepare the data to be logged
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'position_id': position_id,
            'trade_type': trade_type,
            'entry_price': entry_price,
            'sold_price': sold_price,
            'profit_usdt': profit_usdt,
            'gain_loose': gain_loose
        }

        # Write to CSV file
        file_exists = os.path.isfile(log_file_path)
        with open(log_file_path, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'position_id', 'trade_type', 'entry_price', 'sold_price', 'profit_usdt', 'gain_loose']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()  # Write header only if the file does not exist

            writer.writerow(log_data)

        print(f"Position logged: {log_data}")
        logging.info(f"Position logged: {log_data}")

    def calculate_profit(self, trade_quantity, sold_price, entry_price):
        try:
            commission = trade_quantity * (entry_price + sold_price) * 0.00075
            profit_usdt = ((sold_price - entry_price) * trade_quantity) - commission

            return round(profit_usdt, 2)

        except (TypeError, ValueError):
            # Handle conversion failures or None values
            return 0

    def invested_budget(self):
        """
        Calculate the total invested amount based on the current positions recorded in positions.json.
        Returns three values:
        1. stable_invested: Invested amount where dip_flag != position['dip']
        2. dip_invested: Invested amount where dip_flag == position['dip']
        3. total_investment: Total invested amount for all positions
        :return: stable_invested, dip_invested, total_investment
        """
        try:
            # Get all positions
            positions = self.position_manager.get_positions()
            total_invested = 0.0
            stable_invested = 0.0
            dip_invested = 0.0

            # Iterate over each position and calculate the invested amount
            for position_id, position in positions.items():
                entry_price = float(position['entry_price'])
                amount = float(position['amount'])
                invested_amount = entry_price * amount
                total_invested += invested_amount

                # Check the dip flag and categorize the investment
                if position['dip'] == 1:
                    dip_invested += invested_amount
                else:
                    stable_invested += invested_amount

            return stable_invested, dip_invested, total_invested

        except Exception as e:
            logging.error(f"Error calculating invested budget: {e}")
            print(f"Error calculating invested budget: {e}")
            return 0.0, 0.0, 0.0

    def macd_positive(self, all_features, interval):
        """
        Checks if the MACD fast value is greater than the MACD signal for the given interval.

        :param all_features: A dictionary containing processed features for each interval.
        :param interval: The specific interval to check (e.g., '5m', '15m', '1h').
        :return: True if MACD fast > MACD signal, otherwise raises an error.
        """
        if interval in all_features:
            macd = all_features[interval].get('MACD')
            macd_signal = all_features[interval].get('MACD_signal')
            macd_hist = all_features[interval].get('MACD_hist')

            if macd is not None and macd_signal is not None:
                return macd >= macd_signal and macd_hist > 0
            else:
                raise ValueError(f"MACD values are not available for the interval {interval}.")
        else:
            raise ValueError(f"Interval {interval} not found in all_features.")

    def calculate_portfolio_take_profit(self, all_features):
        """
        Calculate the average take profit for the portfolio where 'dip' = 0.
        Uses the existing 'calculate_adjusted_take_profit' function from the DecisionMaker class.
        :return: The average take profit for stable positions in percentage.
        """
        try:
            positions = self.position_manager.get_positions()
            total_take_profit = 0.0
            count = 0

            # Iterate over each position where 'dip' = 0
            for position_id, position in positions.items():
                if position['dip'] == 0:
                    entry_price = float(position['entry_price'])

                    # Assuming you have the relevant bands (upper and lower) stored in all_features
                    upper_band_profit = all_features[PROFIT_INTERVAL].get('upper_band', None)
                    lower_band_profit = all_features[PROFIT_INTERVAL].get('lower_band', None)

                    # Calculate the adjusted take profit for this position
                    adjusted_take_profit = self.decision_maker.calculate_adjusted_take_profit(
                        entry_price, upper_band_profit, lower_band_profit)

                    total_take_profit += adjusted_take_profit
                    count += 1

            if count == 0:
                return 0.0

            # Calculate the average take profit across all relevant positions
            average_take_profit = total_take_profit / count
            return average_take_profit

        except Exception as e:
            logging.error(f"Error calculating portfolio take profit: {e}")
            print(f"Error calculating portfolio take profit: {e}")
            return 0.0

    def breaking_upper_bands(self, all_features, current_price):

        upper_band_15m = all_features['15m'].get('upper_band', None)
        upper_band_30m = all_features['30m'].get('upper_band', None)
        upper_band_1h = all_features['1h'].get('upper_band', None)

        if current_price > upper_band_15m and current_price > upper_band_30m:
            return True
        elif current_price > upper_band_30m and current_price > upper_band_1h:
            return True
        return False


    def position_expired(self, timestamp, timeout):
        """
        Check if the given timestamp has timed out based on the timeout parameter in hours.

        :param timestamp: The timestamp in the format '%Y-%m-%d %H:%M:%S' (as stored in position.json).
        :param timeout: The timeout duration in hours.
        :return: True if the timestamp has timed out, False otherwise.
        """
        try:
            # Convert the timestamp string to a datetime object
            position_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')

            # Calculate the expiration time by adding the timeout period to the position time
            expiration_time = position_time + timedelta(hours=timeout)

            # Get the current time
            current_time = datetime.now()

            # Check if the current time is greater than or equal to the expiration time
            return current_time >= expiration_time

        except Exception as e:
            # Raise a ValueError with a detailed error message
            raise ValueError(f"Error checking position timeout: {e}")


    def position_period(self, macd_positive):
        if macd_positive:
            return POSITION_CYCLE[1]
        return POSITION_CYCLE[0]

    def initialize_position_period(self):
        """
        Initialize the position_period.json file with a default value of 60 if the file doesn't exist.
        """
        try:
            position_period_file = os.path.join(data_directory, 'position_period.json')
            if not os.path.exists(position_period_file):
                with open(position_period_file, 'w') as file:
                    json.dump({"position_period": 60}, file)  # Default value set to 60
                logging.info("Position period file initialized with default value: 60")
            else:
                logging.info("Position period file already exists.")
        except Exception as e:
            logging.error(f"Failed to initialize position period file: {e}")

    def save_position_period(self, position_period):
        """
        Save the position_period value to a file in the data directory.
        :param position_period: Integer value representing the position period.
        """
        try:
            position_period_file = os.path.join(data_directory, 'position_period.json')
            with open(position_period_file, 'w') as file:
                json.dump({"position_period": position_period}, file)
            logging.info(f"Position period value saved: {position_period}")
        except Exception as e:
            logging.error(f"Failed to save position period value: {e}")

    def load_position_period(self):
        """
        Load the position_period value from the file in the data directory.
        :return: The position_period value if the file exists, otherwise None.
        """
        try:
            position_period_file = os.path.join(data_directory, 'position_period.json')
            if os.path.exists(position_period_file):
                with open(position_period_file, 'r') as file:
                    data = json.load(file)
                    return data.get("position_period", None)
            return None
        except Exception as e:
            logging.error(f"Failed to load position period value: {e}")
            return None


    def check_stable_positions(self):
        try:
            start_time = time.time()
            cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n----------Stable Position check cycle started at {cycle_start_time}.----------")
            logging.info(
                f"//---------------------Stable Position check cycle started at {cycle_start_time}--------------------//")

            position_period = self.load_position_period()
            print(f"\nrescheduled to run every {position_period} Seconds")
            logging.info(f"\nrescheduled to run every {position_period} Seconds")

            print("\nChecking if there is Stable Entries:")
            if self.stable_position():
                # Taking Bot Manager Class Instance
                bot_manager = BotManager()

                # Get features and make a decision on whether to sell
                market_data = self.data_collector.collect_data()
                all_features = self.feature_processor.process(market_data)
                stable_invested, dip_invested, total_invested = self.invested_budget()

                market_stable,stable_intervals = self.decision_maker.market_downtrend_stable(all_features)

                current_price = self.trader.get_current_price()
                if current_price is None:
                    print("Failed to get current price. Skipping position check.")
                    logging.info("Failed to get current price. Skipping position check.")
                    return

                print(f"Stable Intervals= {stable_intervals} ,{market_stable}")
                logging.info(f"Stable Intervals= {stable_intervals} ,{market_stable}")

                stable_positions_len = len([position for position_id, position in self.position_manager.get_positions().items() if position['dip'] == 0])
                print(f"Stable Positions Count: {stable_positions_len}")
                logging.info(f"Stable Positions Count: {stable_positions_len}")

                portfolio_gain = self.decision_maker.calculate_stable_portfolio_gain(bot_manager, current_price)
                macd_positive = self.macd_positive(all_features, TRADING_INTERVAL)
                position_period = self.position_period(macd_positive)
                self.save_position_period(position_period)
                portfolio_take_profit_avg = self.calculate_portfolio_take_profit(all_features)
                breaking_upper_bands = self.breaking_upper_bands(all_features, current_price)

                print(f"MACD Status: {macd_positive}")
                logging.info(f"MACD Status: {macd_positive}")

                print(f"Portfolio Trailing Percentage: {portfolio_take_profit_avg:.2f}%")
                logging.info(f"Portfolio Trailing Percentage: {portfolio_take_profit_avg:.2f}%")

                print(f"Portfolio Gain/Loss Percentage: {portfolio_gain:.2f}%")
                logging.info(f"Portfolio Gain/Loss Percentage: {portfolio_gain:.2f}%")

                if stable_positions_len >= TRAILING_POSITIONS_COUNT and macd_positive and (portfolio_gain >= portfolio_take_profit_avg or breaking_upper_bands):
                    print("Portfolio Now Processing Under Trailing Stop Level:\n")
                    logging.info("Portfolio Now Processing Under Trailing Stop Level:\n")
                    reversed_decision ,message = self.decision_maker.check_for_sell_due_to_reversal(bot_manager, current_price)

                    if reversed_decision == "Sell":
                        print(message)
                        logging.info(message)
                        positions_copy = list(self.position_manager.get_positions().items())
                        for position_id, position in positions_copy:
                            entry_price = position['entry_price']
                            amount = position['amount']
                            dip_flag = position['dip']

                            gain_loose = round(self.calculate_gain_loose(entry_price, current_price), 2)

                            if dip_flag == 0:
                                trade_type = 'Stable'
                                print(f"Selling position {position_id}")
                                logging.info(f"Selling position {position_id}")
                                trade_status, order_details = self.trader.execute_trade(reversed_decision, amount)
                                if trade_status == "Success":
                                    stable_invested, dip_invested, total_invested = self.invested_budget()
                                    profit_usdt = self.calculate_profit(trade_quantity=amount, sold_price=current_price,
                                                                        entry_price=entry_price)
                                    self.position_manager.remove_position(position_id)
                                    self.log_sold_position(position_id, trade_type, entry_price, current_price, profit_usdt,
                                                           gain_loose)
                                    print(f"Position {position_id} sold successfully")
                                    logging.info(f"Position {position_id} sold successfully")
                                    self.notifier.send_notification("Stable Trade Executed",
                                                                    f"Sold {amount} {COIN} at ${current_price}\n"
                                                                    f"Gain/Loose: {gain_loose}%\n"
                                                                    "Sell Mode: Trailing\n"
                                                                    f"Stable Invested: {round(stable_invested)} USDT\n"
                                                                    f"Dip Invested: {round(dip_invested)} USDT\n"
                                                                    f"Total Invested: {round(total_invested)} USDT")
                                else:
                                    error_message = f"Failed to execute Sell order: {order_details}"
                                    self.save_error_to_csv(error_message)
                                    self.notifier.send_notification("Trade Error", error_message)
                    else:
                        print(message)
                        logging.info(message)
                else:
                    print("positions Now Processing Under Fixed Profit-Loss:\n")
                    logging.info("positions Now Processing Under Fixed Profit-Loss:\n")
                    # Iterate over a copy of the positions to avoid runtime errors
                    positions_copy = list(self.position_manager.get_positions().items())
                    for position_id, position in positions_copy:
                        entry_price = position['entry_price']
                        amount = position['amount']
                        dip_flag = position['dip']
                        timestamp = position['timestamp']

                        if not all_features:
                            print("Failed to process features for position check.")
                            logging.info("Failed to process features for position check.")
                            return

                        final_decision, adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit = self.decision_maker.make_decision(
                            "Suspended", current_price, entry_price, all_features, self.position_expired(timestamp, POSITION_TIMEOUT), macd_positive)
                        gain_loose = round(self.calculate_gain_loose(entry_price, current_price), 2)

                        if final_decision == "Sell" and dip_flag == 0:
                            trade_type = 'Stable'
                            print(f"Selling position {position_id}")
                            logging.info(f"Selling position {position_id}")
                            trade_status, order_details = self.trader.execute_trade(final_decision, amount)
                            if trade_status == "Success":
                                stable_invested, dip_invested, total_invested = self.invested_budget()
                                profit_usdt = self.calculate_profit(trade_quantity=amount, sold_price=current_price,
                                                                    entry_price=entry_price)
                                self.position_manager.remove_position(position_id)
                                self.log_sold_position(position_id, trade_type, entry_price, current_price, profit_usdt, gain_loose)
                                print(f"Position {position_id} sold successfully")
                                logging.info(f"Position {position_id} sold successfully")
                                self.notifier.send_notification("Stable Trade Executed", f"Sold {amount} {COIN} at ${current_price}\n"
                                                                                  f"Gain/Loose: {gain_loose}%\n"
                                                                                  "Sell Mode: Fix-Loss/Profit\n"       
                                                                                  f"Stable Invested: {round(stable_invested)} USDT\n"
                                                                                  f"Dip Invested: {round(dip_invested)} USDT\n"
                                                                                  f"Total Invested: {round(total_invested)} USDT")
                            else:
                                error_message = f"Failed to execute Sell order: {order_details}"
                                self.save_error_to_csv(error_message)
                                self.notifier.send_notification("Trade Error", error_message)
                        else:
                            if dip_flag == 0:
                                print(f"Holding position: {position_id}, timestamp: {timestamp},Entry Price: {entry_price}, Current Price: {current_price}, ((Gain/Loose: {gain_loose}%))")
                                logging.info(f"Holding position: {position_id}, timestamp: {timestamp}, Entry Price: {entry_price}, Current Price: {current_price}, ((Gain/Loose: {gain_loose}%))")
                                print(f"dynamic_stop_loss_lower: {round(adjusted_stop_loss_lower, 2)}%, dynamic_stop_loss_middle: {round(adjusted_stop_loss_middle, 2)}%, dynamic_take_profit: {round(adjusted_take_profit, 2)}%\n")
                                logging.info(f"dynamic_stop_loss_lower: {round(adjusted_stop_loss_lower, 2)}%, dynamic_stop_loss_middle: {round(adjusted_stop_loss_middle, 2)}%, dynamic_take_profit: {round(adjusted_take_profit, 2)}\n%")

                    print(f"Stable Invested: {round(stable_invested)} USDT\n"
                                              f"Dip Invested: {round(dip_invested)} USDT\n"
                                              f"Total Invested: {round(total_invested)} USDT")
                    logging.info(f"Stable Invested: {round(stable_invested)} USDT\n"
                                              f"Dip Invested: {round(dip_invested)} USDT\n"
                                              f"Total Invested: {round(total_invested)} USDT")
                    self.log_time("Position check", start_time)
            else:
                print("No Stable Entry Founds")

            if self.dip_position():
                print("\nChecking if there is Dip Entries:")
                current_price = self.trader.get_current_price()
                positions_copy = [
                    (position_id, position) for position_id, position in self.position_manager.get_positions().items()
                    if position.get('dip') == 1
                ]

                # Calculate the average entry price using the new function
                avg_entry_price = self.calculate_average_entry_price(positions_copy)

                # Print and log the average entry price
                print(f"Average Entry Price for Dip Positions: {avg_entry_price:.2f}")
                logging.info(f"Average Entry Price for Dip Positions: {avg_entry_price:.2f}")

                for position_id, position in positions_copy:
                    entry_price = position['entry_price']
                    gain_loose = round(self.calculate_gain_loose(entry_price, current_price), 2)
                    amount = position['amount']
                    print(
                        f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")
                    logging.info(
                        f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")

                stable_invested, dip_invested, total_invested = self.invested_budget()
                print(f"\nStable Invested: {round(stable_invested)} USDT\n"
                      f"Dip Invested: {round(dip_invested)} USDT\n"
                      f"Total Invested: {round(total_invested)} USDT")
                logging.info(f"Stable Invested: {round(stable_invested)} USDT\n"
                             f"Dip Invested: {round(dip_invested)} USDT\n"
                             f"Total Invested: {round(total_invested)} USDT")

            else:
                print("No Dip Entry Founds")



        except Exception as e:
            logging.error(f"An error occurred during position check: {str(e)}")
            self.save_error_to_csv(str(e))

    def save_historical_context_for_stable(self):
        """
        Saves historical context of the processed features for a specific interval,
        while ensuring that only the latest 6 hours of data is stored. Data older than
        6 hours will be cleared.
        """
        try:
            historical_file = os.path.join(data_directory, f'{TRADING_INTERVAL}_stable_historical_context.json')

            # Collect market data and process features
            market_data = self.data_collector.collect_data()
            features = self.feature_processor.process(market_data)
            trading_feature = features[TRADING_INTERVAL]

            # Load existing data if the file already exists
            if os.path.exists(historical_file):
                with open(historical_file, 'r') as file:
                    historical_data = json.load(file)
            else:
                historical_data = []

            # Get the current timestamp and store it as a string
            current_time = datetime.now()
            timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
            trading_feature['timestamp'] = timestamp

            # Append the current feature data
            historical_data.append(trading_feature)

            # Filter the data to only keep entries from the last 6 hours
            one_day_ago = current_time - timedelta(days=1)
            historical_data = [entry for entry in historical_data
            if datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') > one_day_ago]

            # Save the updated historical data back to the JSON file
            with open(historical_file, 'w') as file:
                json.dump(historical_data, file, indent=4)

            print(f"Stable Historical context for interval '{TRADING_INTERVAL}' saved successfully.")
            logging.info(f"Stable Historical context for interval '{TRADING_INTERVAL}' saved successfully.")
        except Exception as e:
            print(f"Error saving Stable historical context for interval '{TRADING_INTERVAL}': {e}")
            logging.info(f"Error saving Stable historical context for interval '{TRADING_INTERVAL}': {e}")

    def dip_position(self):
        positions_copy = list(self.position_manager.get_positions().items())
        for position_id, position in positions_copy:
            dip_flag = position['dip']
            if dip_flag == 1:
                return True
        return False

    def stable_position(self):
        positions_copy = list(self.position_manager.get_positions().items())
        for position_id, position in positions_copy:
            if position['dip'] == 0:
                return True
        return False

    def save_historical_context_for_dip(self):
        """
        Saves historical context of the processed features for the specified interval.
        Only the latest 3 days of data will be stored. Data older than 3 days will be removed.
        """
        try:
            # Collect market data and process features
            market_data = self.data_collector.collect_data()
            features = self.feature_processor.process(market_data)
            dip_feature = features[DIP_INTERVAL]

            # Prepare the historical file path based on the interval
            historical_file = os.path.join(data_directory, f'{DIP_INTERVAL}_dip_historical_context.json')

            # Load existing data if the file already exists
            if os.path.exists(historical_file):
                with open(historical_file, 'r') as file:
                    historical_data = json.load(file)
            else:
                historical_data = []

            # Append the current feature data along with the timestamp
            current_time = datetime.now()
            timestamp = current_time.strftime('%Y-%m-%d %H:%M:%S')
            dip_feature['timestamp'] = timestamp
            historical_data.append(dip_feature)

            # Filter out data older than 3 days
            three_days_ago = current_time - timedelta(days=3)
            historical_data = [entry for entry in historical_data if
                               datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') > three_days_ago]

            # Save the updated historical data back to the JSON file
            with open(historical_file, 'w') as file:
                json.dump(historical_data, file, indent=4)

            print(f"Dip Historical context for interval '{DIP_INTERVAL}' saved successfully.")
            logging.info(f"Dip Historical context for interval '{DIP_INTERVAL}' saved successfully.")
        except Exception as e:
            print(f"Error saving Dip historical context for interval '{DIP_INTERVAL}': {e}")
            logging.info(f"Error saving Dip historical context for interval '{DIP_INTERVAL}': {e}")


    def run_prediction_cycle(self):

        # Taking Bot Manager Class Instance
        bot_manager = BotManager()

        attempt = 0
        while attempt < 3:
            try:
                start_time = time.time()
                cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(
                    f"\n\n&&&&&&&&&&Prediction cycle started at {cycle_start_time}, running every {TRADING_INTERVAL}.&&&&&&&&&&")
                logging.info(
                    f"//&&&&&&&&&&&&&&&&&&&Prediction cycle started at {cycle_start_time}&&&&&&&&&&&&&&&&&&&//")

                # Collect market data
                market_data_start = time.time()
                print("Collecting market data...")
                logging.info("Collecting market data...")
                market_data = self.data_collector.collect_data()
                self.log_time("Market data collection", market_data_start)

                if not market_data:
                    print("Failed to collect market data. Skipping this cycle.")
                    logging.info("Failed to collect market data. Skipping this cycle.")
                    return

                # Process features
                feature_processing_start = time.time()
                print("Processing features...")
                logging.info("Processing features...")
                all_features = self.feature_processor.process(market_data)
                historical_data = self.feature_processor.get_stable_historical_data()
                self.log_time("Feature processing", feature_processing_start)

                if not all_features:
                    print("Failed to process features. Skipping this cycle.")
                    logging.info("Failed to process features. Skipping this cycle.")
                    return

                # Get current price
                price_check_start = time.time()
                print("Getting current price before executing the prediction...")
                logging.info("Getting current price before executing the prediction...")
                current_price = self.trader.get_current_price()
                self.log_time("Current price check", price_check_start)
                if current_price is None:
                    print("Failed to get current price. Skipping this cycle.")
                    logging.info("Failed to get current price. Skipping this cycle.")
                    return

                # Check if the price change is greater than PREDICT_IN_BANDWIDTH% and check MACD status
                bandwidth_price_change = self.calculate_prediction_bandwidth(all_features)
                macd_positive = self.macd_positive(all_features, TRADING_INTERVAL)
                if bandwidth_price_change > PREDICT_BANDWIDTH and macd_positive:
                    prediction_start = time.time()
                    print("Generating prediction...")
                    logging.info("Generating prediction...")
                    prediction, explanation = self.predictor.get_prediction(all_features=all_features,
                                                                            current_price=current_price,
                                                                            historical_data=historical_data,
                                                                            prediction_type='Stable')

                    self.log_time("Prediction generation", prediction_start)
                    print(f"Predictor Recommends To  ///{prediction}///")
                    logging.info(f"Prediction: {prediction}. Explanation: {explanation}")
                else:
                    prediction = "Hold"
                    print(f"Bandwidth price change is less than {PREDICT_BANDWIDTH}%. Prediction: Hold")
                    logging.info(f"Bandwidth price change is less than {PREDICT_BANDWIDTH}%. Prediction: Hold")

                # Make a decision
                trade_decision_start = time.time()
                trading_cryptocurrency_amount = self.convert_usdt_to_crypto(current_price,
                                                                      self.decision_maker.calculate_buy_amount
                                                                      (all_features=all_features,
                                                                      amount_atr_interval=AMOUNT_ATR_INTERVAL,
                                                                      amount_rsi_interval=AMOUNT_RSI_INTERVAL,
                                                                      capital=CAPITAL_AMOUNT))

                dip_cryptocurrency_amount = self.convert_usdt_to_crypto(current_price, USDT_DIP_AMOUNT)

                final_decision, adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit = self.decision_maker.make_decision(
                    prediction, current_price, None, all_features, position_expired=None, macd_positive=None)
                self.log_time("Trade decision making", trade_decision_start)

                # Handle Buy and Sell decisions
                if final_decision == "Buy":
                    trade_execution_start = time.time()
                    print("Executing Buy trade...")
                    logging.info("Executing Buy trade...")
                    trade_status, order_details = self.trader.execute_trade(final_decision, trading_cryptocurrency_amount)
                    self.log_time("Trade execution (Buy)", trade_execution_start)

                    if trade_status == "Success":
                        position_id = str(int(time.time()))
                        self.position_manager.add_position(position_id, current_price, trading_cryptocurrency_amount, dip_flag=0)
                        stable_invested, dip_invested, total_invested = self.invested_budget()
                        print(
                            f"New position added: {position_id}, Entry Price: {current_price}, Amount: {trading_cryptocurrency_amount}")
                        logging.info(
                            f"New position added: {position_id}, Entry Price: {current_price}, Amount: {trading_cryptocurrency_amount}")
                        self.notifier.send_notification("Stable Trade Executed",
                                                        f"Bought {trading_cryptocurrency_amount} {COIN} at ${current_price}\n"
                                                                          f"Stable Invested: {round(stable_invested)} USDT\n"
                                                                          f"Dip Invested: {round(dip_invested)} USDT\n"
                                                                          f"Total Invested: {round(total_invested)} USDT")
                    else:
                        error_message = f"Failed to execute Buy order: {order_details}"
                        self.save_error_to_csv(error_message)
                        logging.error(f"Failed to execute Buy order: {order_details}")

                elif final_decision == "Buy_Dip":
                    trade_execution_start = time.time()
                    print("Executing Buying a Dip...")
                    logging.info("Executing Buying a Dip...")
                    trade_status, order_details = self.trader.execute_trade(final_decision, dip_cryptocurrency_amount)
                    self.log_time("Trade execution (Buy) For Dip", trade_execution_start)

                    if trade_status == "Success":
                        position_id = str(int(time.time()))
                        self.position_manager.add_position(position_id, current_price, dip_cryptocurrency_amount, dip_flag=1)
                        stable_invested, dip_invested, total_invested = self.invested_budget()
                        print(
                            f"New position added: {position_id}, Entry Price: {current_price}, Amount: {dip_cryptocurrency_amount}")
                        logging.info(
                            f"New position added: {position_id}, Entry Price: {current_price}, Amount: {dip_cryptocurrency_amount}")
                        self.notifier.send_notification("Dip Trade Executed",
                                                        f"Bought {dip_cryptocurrency_amount} {COIN} at ${current_price}\n"
                                                                 f"Stable Invested: {round(stable_invested)} USDT\n"
                                                                 f"Dip Invested: {round(dip_invested)} USDT\n"
                                                                 f"Total Invested: {round(total_invested)} USDT")
                    else:
                        error_message = f"Failed to execute Buy order: {order_details}"
                        self.save_error_to_csv(error_message)
                        logging.error(f"Failed to execute Buy order: {order_details}")

                elif prediction == "Buy" and final_decision == "Hold":
                    self.notifier.send_notification(title="Decision Maker", message=f"Decision Maker Hold The Buy Prediction Prediction at {current_price}")
                    print(f"Decision Maker Hold The Buy Prediction at {current_price}")
                    logging.info(f"Decision Maker Hold The Buy Prediction at {current_price}")

                else:
                    print("No Trade Executed")

                # Handle other conditions for "Hold" and "Sell"
                self.log_time("Prediction cycle", start_time)
                break

            except Exception as e:
                attempt += 1
                logging.error(f"An error occurred during the run (Attempt {attempt}): {str(e)}")
                self.save_error_to_csv(str(e))
                time.sleep(5)
                if attempt >= 3:
                    self.notifier.send_notification("Bot Stopped",
                                                    "The bot encountered repeated errors and is stopping.")
                    print("Bot has stopped due to repeated errors.")
                    raise

    def calculate_average_entry_price(self, positions_copy):
        """
        Calculate the average entry price for all positions with a 'dip' flag set to 1.
        :param positions_copy: List of positions to calculate the average for.
        :return: The average entry price.
        """
        total_entry_price = 0
        position_count = 0

        for position_id, position in positions_copy:
            total_entry_price += position['entry_price']
            position_count += 1

        # Avoid division by zero by returning 0 if no positions are found
        if position_count > 0:
            avg_entry_price = total_entry_price / position_count
        else:
            avg_entry_price = 0

        return avg_entry_price

    def check_dip_positions(self):
        try:
            start_time = time.time()
            cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n^^^^^^^^^^Dip Position check cycle started at {cycle_start_time}.^^^^^^^^^^")
            logging.info(
                f"//^^^^^^^^^^^^^^^^^^^^Dip Position check cycle started at {cycle_start_time}^^^^^^^^^^^^^^^^^^^^//")

            # Loading dip_positions
            if self.dip_position():
                positions_copy = [
                    (position_id, position) for position_id, position in self.position_manager.get_positions().items()
                    if position.get('dip') == 1
                ]

                # Calculate the average entry price
                avg_entry_price = self.calculate_average_entry_price(positions_copy)

                # Print and log the average entry price
                print(f"Average Entry Price for Dip Positions: {avg_entry_price:.2f}")
                logging.info(f"Average Entry Price for Dip Positions: {avg_entry_price:.2f}")

                # Loading market data
                market_data = self.data_collector.collect_data()
                all_features = self.feature_processor.process(market_data)

                # Loading Dip Historical context data
                historical_data = self.feature_processor.get_dip_historical_data()

                # Get current price
                price_check_start = time.time()
                print("Getting current price before executing the prediction...")
                logging.info("Getting current price before executing the prediction...")
                current_price = self.trader.get_current_price()
                self.log_time("Current price check", price_check_start)
                if current_price is None:
                    print("Failed to get current price. Skipping this cycle.")
                    logging.info("Failed to get current price. Skipping this cycle.")
                    return

                # Generate Dip Prediction
                prediction_start = time.time()
                print("Generating prediction...")
                logging.info("Generating prediction...")
                prediction, explanation = self.predictor.get_prediction(current_price=current_price,
                                                                        historical_data=historical_data,
                                                                        prediction_type='Dip',
                                                                        positions=positions_copy)
                self.log_time("Prediction generation", prediction_start)
                print(f"Predictor Recommends To  ///{prediction}///")
                logging.info(f"Prediction: {prediction}. Explanation: {explanation}")

                # Dip Trade Execution
                for position_id, position in positions_copy:
                    entry_price = position['entry_price']
                    amount = position['amount']

                    gain_loose = round(self.calculate_gain_loose(entry_price, current_price), 2)

                    if prediction == 'Sell':
                        trade_type = 'Dip'
                        print(f"Selling position {position_id}")
                        logging.info(f"Selling position {position_id}")
                        trade_status, order_details = self.trader.execute_trade(prediction, amount)
                        if trade_status == "Success":
                            profit_usdt = self.calculate_profit(trade_quantity=amount, sold_price=current_price,
                                                                entry_price=entry_price)
                            stable_invested, dip_invested, total_invested = self.invested_budget()
                            self.position_manager.remove_position(position_id)
                            self.log_sold_position(position_id, trade_type, entry_price, current_price, profit_usdt,
                                                   gain_loose)
                            print(f"Position {position_id} sold successfully")
                            logging.info(f"Position {position_id} sold successfully")
                            self.notifier.send_notification("Dip Trade Executed",
                                                            f"Sold {amount} {COIN} at ${current_price}\n"
                                                            f"Gain/Loose: {gain_loose}%\n"
                                                            "Sell Mode: Dip\n"
                                                            f"Stable Invested: {round(stable_invested)} USDT\n"
                                                            f"Dip Invested: {round(dip_invested)} USDT\n"
                                                            f"Total Invested: {round(total_invested)} USDT")
                        else:
                            error_message = f"Failed to execute Sell order: {order_details}"
                            self.save_error_to_csv(error_message)
                            self.notifier.send_notification("Trade Error", error_message)

                    else:
                        print("Prediction: ///Hold///")
                        logging.info("Prediction: ///Hold///")
                        print(
                            f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")
                        logging.info(
                            f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")

            else:
                print("\nNo Dip Entry Founds")
                logging.info("\nNo Dip Entry Founds")

        except Exception as e:
            logging.error(f"An error occurred during position check: {str(e)}")
            self.save_error_to_csv(str(e))

    def check_stable_prediction_timeframe(self):
        """
        Start the stable prediction cycle 45 seconds before the close of every '5m' interval.
        """
        try:
            while True:
                now = datetime.now()

                # Calculate the time until the next '5m' interval close time
                minutes = now.minute
                next_close_minute = (minutes // PREDICTION_CYCLE + 1) * PREDICTION_CYCLE
                next_close_time = now.replace(minute=next_close_minute % 60, second=0, microsecond=0)

                # If the next close time goes beyond the current hour
                if next_close_minute >= 60:
                    next_close_time = next_close_time + timedelta(hours=1)

                # Time to trigger the prediction cycle (30 seconds before close)
                run_time = next_close_time - timedelta(seconds=40)

                # Calculate how long to wait until 30 seconds before the next close
                time_to_wait = (run_time - now).total_seconds()

                # Wait for the calculated time
                if time_to_wait > 0:
                    print(f"Stable Prediction Waiting for {time_to_wait} seconds until {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    time.sleep(time_to_wait)

                # Now run the prediction cycle
                print(f"Running prediction cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.run_prediction_cycle()

                # Wait for 30 seconds to let the next '5m' close before scheduling the next prediction
                time.sleep(40)

        except Exception as e:
            print(f"Error in stable prediction: {e}")
            logging.error(f"Error in stable prediction: {e}")


    def check_stable_historical_timeframe(self):
        """
        Start the stable prediction cycle 45 seconds before the close of every '5m' interval.
        """
        try:
            while True:
                now = datetime.now()

                # Calculate the time until the next '5m' interval close time
                minutes = now.minute
                next_close_minute = (minutes // PREDICTION_CYCLE + 1) * PREDICTION_CYCLE
                next_close_time = now.replace(minute=next_close_minute % 60, second=0, microsecond=0)

                # If the next close time goes beyond the current hour
                if next_close_minute >= 60:
                    next_close_time = next_close_time + timedelta(hours=1)

                # Time to trigger the prediction cycle (30 seconds before close)
                run_time = next_close_time - timedelta(seconds=15)

                # Calculate how long to wait until 30 seconds before the next close
                time_to_wait = (run_time - now).total_seconds()

                # Wait for the calculated time
                if time_to_wait > 0:
                    print(f"Stable Historical Waiting for {time_to_wait} seconds until {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    time.sleep(time_to_wait)

                # Now run the prediction cycle
                print(f"Running prediction cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.save_historical_context_for_stable()

                # Wait for 30 seconds to let the next '5m' close before scheduling the next prediction
                time.sleep(15)

        except Exception as e:
            print(f"Error in stable prediction: {e}")
            logging.error(f"Error in stable prediction: {e}")

    def check_dip_historical_timeframe(self):
        """
        Start the dip prediction cycle 30 seconds before the close of every '15m' interval.
        """
        try:
            while True:
                now = datetime.now()

                # Calculate the time until the next '15m' interval close time
                minutes = now.minute
                next_close_minute = (minutes // DIP_CYCLE + 1) * DIP_CYCLE
                next_close_time = now.replace(minute=next_close_minute % 60, second=0, microsecond=0)

                # If the next close time goes beyond the current hour
                if next_close_minute >= 60:
                    next_close_time = next_close_time + timedelta(hours=1)

                # Time to trigger the dip historical context (30 seconds before close)
                run_time = next_close_time - timedelta(seconds=15)

                # Calculate how long to wait until 30 seconds before the next close
                time_to_wait = (run_time - now).total_seconds()

                # Wait for the calculated time
                if time_to_wait > 0:
                    print(
                        f"Dip Historical Waiting for {time_to_wait} seconds until {run_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    time.sleep(time_to_wait)

                # Now run the dip historical context saving
                print(f"Running dip historical context at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                self.save_historical_context_for_dip()

                # Wait for 30 seconds to let the next '15m' close before scheduling the next prediction
                time.sleep(15)

        except Exception as e:
            print(f"Error in dip historical prediction: {e}")
            logging.error(f"Error in dip historical prediction: {e}")

    def start(self):
        try:
            # Load the last saved position_period value
            last_position_period = self.load_position_period()
            if last_position_period is not None:
                print(f"Loaded last position period value: {last_position_period}")
                logging.info(f"Loaded last position period value: {last_position_period}")
            else:
                last_position_period = 60  # Default value
                print("No previous position period value found, using default: 60.")
                logging.info("No previous position period value found, using default: 60.")

            # Schedule the stable position check based on the last_position_period
            print(f"Scheduling position checks every {last_position_period} seconds.")
            logging.info(f"Scheduling position checks every {last_position_period} seconds.")

            # Cancel any previous scheduled jobs before rescheduling
            schedule.clear('position_check')  # Use a tag to identify the job

            # Reschedule the job with the new interval
            schedule.every(last_position_period).seconds.do(self.check_stable_positions).tag('position_check')

            # Start the historical context cycle in a separate thread
            prediction_thread = threading.Thread(target=self.check_stable_prediction_timeframe, daemon=True)
            prediction_thread.start()

            # Start the stable prediction cycle in a separate thread
            prediction_thread = threading.Thread(target=self.check_stable_historical_timeframe, daemon=True)
            prediction_thread.start()

            # Start the stable prediction cycle in a separate thread
            prediction_thread = threading.Thread(target=self.check_dip_historical_timeframe, daemon=True)
            prediction_thread.start()

            # Schedule the dip check to run every 12 hours
            schedule.every(1).hours.do(self.check_dip_positions)

            # Continuously monitor position_period and run the scheduled tasks
            while True:
                schedule.run_pending()

                # Continuously check for updates in position_period
                current_position_period = self.load_position_period()  # Load the latest value

                if current_position_period != last_position_period:
                    print(
                        f"Detected change in position period from {last_position_period} to {current_position_period}")
                    logging.info(
                        f"Detected change in position period from {last_position_period} to {current_position_period}")

                    # Update last_position_period
                    last_position_period = current_position_period

                    # Reschedule the job with the new interval
                    schedule.clear('position_check')  # Clear the previous job with the 'position_check' tag
                    schedule.every(current_position_period).seconds.do(self.check_stable_positions).tag(
                        'position_check')
                    print(f"Position check rescheduled to run every {current_position_period} seconds.")
                    logging.info(f"Position check rescheduled to run every {current_position_period} seconds.")

                time.sleep(1)  # Add a small delay to avoid tight looping

        except Exception as e:
            logging.error(f"Bot encountered a critical error and is stopping: {e}")
            self.save_error_to_csv(str(e))
            self.notifier.send_notification("Bot Stopped", f"The bot encountered a critical error and is stopping: {e}")
            print("Bot has stopped due to an error. Exiting program.")


# Example usage:
if __name__ == "__main__":
    bot_manager = BotManager()
    bot_manager.start()

