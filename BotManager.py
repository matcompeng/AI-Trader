import json
import os
import time
import schedule
import logging
import traceback
import pandas as pd
from datetime import datetime
from DataCollector import DataCollector
from FeatureProcessor import FeatureProcessor
from ChatGPTClient import ChatGPTClient
from Predictor import Predictor
from DecisionMaker import DecisionMaker
from Trader import Trader
from Notifier import Notifier
import csv

# Bot Configurations ------------------------------------------------------------------------------
TRADING_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '1d']
COIN = 'BNB'                    # Select Cryptocurrency
TRADING_PAIR = 'BNBUSDT'        # Select Cryptocurrency Trading Pair
PROFIT_INTERVAL = '1h'          # Select The Interval For Take Profit Calculations
LOOSE_INTERVAL = '1h'           # Select The Interval For Stop Loose Calculations
PREDICTOR_INTERVAL = '1h'       # Select The Interval That Activate/Deactivate Predictor through PREDICT_IN_BANDWIDTH
SR_INTERVAL = '1h'              # Select The Interval That Trader Define Support and Resistance Levels
CHECK_POSITIONS_ON_BUY = True   # Set True If You Need Bot Manager Check The Positions During Buy Cycle
POSITION_CYCLE = 15            # Time in seconds to check positions
PREDICTION_CYCLE = 3 * 60      # Time in seconds to run the full bot cycle              # Time in seconds between each run of the Bot Cycle in seconds
PREDICT_IN_BANDWIDTH = 2        # Define Minimum Bandwidth Percentage to Activate Trading
BASE_TAKE_PROFIT = 0.20         # Define Base Take Profit Percentage %
USDT_AMOUNT = 10                # Amount of Currency to trade for each Position
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

    def add_position(self, position_id, entry_price, amount):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Add timestamp
        self.positions[position_id] = {
            'entry_price': entry_price,
            'amount': amount,
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

        self.data_collector = DataCollector(api_key, api_secret, intervals=TRADING_INTERVALS, symbol=TRADING_PAIR)
        self.feature_processor = FeatureProcessor(intervals=TRADING_INTERVALS)
        self.chatgpt_client = ChatGPTClient()
        self.predictor = Predictor(self.chatgpt_client, coin=COIN, sr_interval=SR_INTERVAL)
        self.decision_maker = DecisionMaker(base_take_profit=BASE_TAKE_PROFIT, profit_interval=PROFIT_INTERVAL, loose_interval=LOOSE_INTERVAL)
        self.trader = Trader(symbol=TRADING_PAIR)  # Initialize the Trader class
        self.notifier = Notifier()
        self.position_manager = PositionManager()

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

    def calculate_band_price_change(self, all_features):
        """
        Calculate the percentage price change between the upper and lower Bollinger Bands of the PREDICTOR_INTERVAL interval.
        """
        upper_band_15m = all_features[PREDICTOR_INTERVAL].get('upper_band')
        lower_band_15m = all_features[PREDICTOR_INTERVAL].get('lower_band')

        if upper_band_15m and lower_band_15m:
            price_change_15m = ((upper_band_15m - lower_band_15m) / lower_band_15m) * 100
            return price_change_15m

    def price_is_over_band(self, all_features, current_price):

        lower_band_15m = all_features[PREDICTOR_INTERVAL].get('lower_band')
        if current_price >= lower_band_15m:
            return True

    def calculate_gain_loose(self,entry_price, current_price):
        gain_loose = ((current_price - entry_price) / entry_price) * 100
        return gain_loose

    def log_sold_position(self, position_id, entry_price, sold_price, profit_usdt, gain_loose):
        """
        Log the details of a sold position to a CSV file.
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
            'entry_price': entry_price,
            'sold_price': sold_price,
            'profit_usdt': profit_usdt,
            'gain_loose': gain_loose
        }

        # Write to CSV file
        file_exists = os.path.isfile(log_file_path)
        with open(log_file_path, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'position_id', 'entry_price', 'sold_price', 'profit_usdt', 'gain_loose']
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
        :return: Total invested amount in USDT
        """
        try:
            # Get all positions
            positions = self.position_manager.get_positions()
            total_invested = 0.0

            # Iterate over each position and calculate the invested amount
            for position_id, position in positions.items():
                entry_price = float(position['entry_price'])
                amount = float(position['amount'])
                invested_amount = entry_price * amount
                total_invested += invested_amount

            return total_invested

        except Exception as e:
            logging.error(f"Error calculating invested budget: {e}")
            print(f"Error calculating invested budget: {e}")
            return 0.0

    def check_positions(self):
        try:
            start_time = time.time()
            cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n*****Position check cycle started at {cycle_start_time}.*****")
            logging.info(
                f"//---------------------Position check cycle started at {cycle_start_time}--------------------//")

            # Get features and make a decision on whether to sell
            market_data = self.data_collector.collect_data()
            all_features = self.feature_processor.process(market_data)

            current_price = self.trader.get_current_price()
            if current_price is None:
                print("Failed to get current price. Skipping position check.")
                logging.info("Failed to get current price. Skipping position check.")
                return

            # Iterate over a copy of the positions to avoid runtime errors
            positions_copy = list(self.position_manager.get_positions().items())
            for position_id, position in positions_copy:
                entry_price = position['entry_price']
                amount = position['amount']

                if not all_features:
                    print("Failed to process features for position check.")
                    logging.info("Failed to process features for position check.")
                    return

                final_decision, adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit = self.decision_maker.make_decision(
                    "Hold", current_price, entry_price, all_features)
                gain_loose = round(self.calculate_gain_loose(entry_price, current_price), 2)

                if final_decision == "Sell":
                    print(f"Selling position {position_id}")
                    logging.info(f"Selling position {position_id}")
                    trade_status, order_details = self.trader.execute_trade(final_decision, amount)
                    if trade_status == "Success":
                        profit_usdt = self.calculate_profit(trade_quantity=amount, sold_price=current_price,
                                                            entry_price=entry_price)
                        self.position_manager.remove_position(position_id)
                        self.log_sold_position(position_id, entry_price, current_price, profit_usdt, gain_loose)
                        print(f"Position {position_id} sold successfully")
                        logging.info(f"Position {position_id} sold successfully")
                        self.notifier.send_notification("Trade Executed", f"Sold {amount} {COIN} at ${current_price}\n"
                                                                          f"Gain/Loose: {gain_loose}%\n"
                                                                          f"Total Invested: {round(self.invested_budget())} USDT")
                    else:
                        error_message = f"Failed to execute Sell order: {order_details}"
                        self.save_error_to_csv(error_message)
                        self.notifier.send_notification("Trade Error", error_message)
                else:
                    print(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")
                    logging.info(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")

            print(f"Total Invested So Far: {round(self.invested_budget())} USDT")
            logging.info(f"Total Invested So far: {round(self.invested_budget())} USDT")
            self.log_time("Position check", start_time)


        except Exception as e:
            logging.error(f"An error occurred during position check: {str(e)}")
            self.save_error_to_csv(str(e))

    def run_prediction_cycle(self):
        attempt = 0
        while attempt < 3:
            try:
                start_time = time.time()
                cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(
                    f"\n\n*****Prediction cycle started at {cycle_start_time}, running every {PREDICTION_CYCLE} seconds.*****")
                logging.info(
                    f"//---------------------Prediction cycle started at {cycle_start_time}--------------------//")

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

                # Check if the price change is greater than PREDICT_IN_BANDWIDTH%
                bandwidth_price_change = self.calculate_band_price_change(all_features)
                if bandwidth_price_change > PREDICT_IN_BANDWIDTH:
                    prediction_start = time.time()
                    print("Generating prediction...")
                    logging.info("Generating prediction...")
                    prediction, explanation = self.predictor.get_prediction(all_features, current_price)
                    self.log_time("Prediction generation", prediction_start)
                    print(f"Predictor Recommends To  ///{prediction}///")
                    logging.info(f"Prediction: {prediction}. Explanation: {explanation}")
                else:
                    prediction = "Hold"
                    print(f"Bandwidth price change is less than {PREDICT_IN_BANDWIDTH}%. Prediction: Hold")
                    logging.info(f"Bandwidth price change is less than {PREDICT_IN_BANDWIDTH}%. Prediction: Hold")

                # Make a decision
                trade_decision_start = time.time()
                crypto_amount = self.convert_usdt_to_crypto(current_price, USDT_AMOUNT)
                final_decision, adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit = self.decision_maker.make_decision(
                    prediction, current_price, None, all_features)
                self.log_time("Trade decision making", trade_decision_start)

                # Handle Buy and Sell decisions
                if final_decision == "Buy":
                    trade_execution_start = time.time()
                    print("Executing Buy trade...")
                    logging.info("Executing Buy trade...")
                    trade_status, order_details = self.trader.execute_trade(final_decision, crypto_amount)
                    self.log_time("Trade execution (Buy)", trade_execution_start)

                    if trade_status == "Success":
                        position_id = str(int(time.time()))
                        self.position_manager.add_position(position_id, current_price, crypto_amount)
                        print(
                            f"New position added: {position_id}, Entry Price: {current_price}, Amount: {crypto_amount}")
                        logging.info(
                            f"New position added: {position_id}, Entry Price: {current_price}, Amount: {crypto_amount}")
                        self.notifier.send_notification("Trade Executed",
                                                        f"Bought {crypto_amount} {COIN} at ${current_price}\n"
                                                        f"Total Invested: {round(self.invested_budget())} USDT")
                    else:
                        error_message = f"Failed to execute Buy order: {order_details}"
                        self.save_error_to_csv(error_message)
                        logging.error(f"Failed to execute Buy order: {order_details}")

                elif prediction == "Buy" and final_decision == "Hold":
                    self.notifier.send_notification(title="Decision Maker", message="Decision Maker Hold The Buy Prediction")
                    print("Decision Maker Hold The Buy Prediction")
                    logging.info("Decision Maker Hold The Buy Prediction")

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

    def start(self):
        try:
            # For testing purposes
            # self.run_prediction_cycle()

            # Schedule the position check every POSITION_CYCLE seconds
            schedule.every(POSITION_CYCLE).seconds.do(self.check_positions)

            # Schedule the prediction cycle every PREDICTION_CYCLE seconds
            schedule.every(PREDICTION_CYCLE).seconds.do(self.run_prediction_cycle)

            # Continuously run the scheduled tasks
            while True:
                schedule.run_pending()
                time.sleep(1)

        except Exception as e:
            logging.error(f"Bot encountered a critical error and is stopping: {e}")
            self.save_error_to_csv(str(e))
            self.notifier.send_notification("Bot Stopped", f"The bot encountered a critical error and is stopping: {e}")
            print("Bot has stopped due to an error. Exiting program.")


# Example usage:
if __name__ == "__main__":
    bot_manager = BotManager()
    bot_manager.start()



