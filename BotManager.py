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
TRADING_INTERVALS = ['1m', '5m', '15m', '1h', '1d']
COIN = 'BNB'                 # Select Cryptocurrency
TRADING_PAIR = 'BNBUSDT'     # Select Cryptocurrency Trading Pair
PROFIT_INTERVAL = '1h'       # Select The Interval For Take Profit Calculations
LOOSE_INTERVAL = '1h'        # Select The Interval For Stop Loose Calculations
PREDICTOR_INTERVAL = '1h'    # Select The Interval That Activate/Deactivate Predictor through PREDICT_IN_BANDWIDTH
SR_INTERVAL = '1h'           # Select The Interval That Trader Define Support and Resistance Levels
BOT_CYCLE = 2 * 60           # Time in seconds between each run of the Bot Cycle in seconds
PREDICT_IN_BANDWIDTH = 2     # Define Minimum Bandwidth Percentage to Activate Trading
BASE_TAKE_PROFIT = 0.25      # Define Base Take Profit Percentage %
USDT_AMOUNT = 10             # Amount of Currency to trade for each Position
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

    def log_sold_position(self, position_id, entry_price, sold_price, gain_loose):
        """
        Log the details of a sold position to a CSV file.
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




    def run(self):
        attempt = 0
        while attempt < 3:
            try:
                # Capture the current timestamp at the start of the cycle
                cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n\n*****Bot cycle started at {cycle_start_time}, running every {BOT_CYCLE} seconds.*****")
                logging.info(f"//---------------------Bot cycle started at {cycle_start_time}--------------------=//")

                start_time = time.time()
                print("Collecting market data...")
                market_data = self.data_collector.collect_data()
                self.log_time("Data collection", start_time)

                if not market_data:
                    print("Failed to collect market data. Skipping this cycle.")
                    return

                start_time = time.time()
                print("Processing features...")
                all_features = self.feature_processor.process(market_data)
                self.log_time("Feature processing", start_time)

                if not all_features:
                    print("Failed to process features. Skipping this cycle.")
                    return

                # Get the current price right before executing the prediction
                print("Getting current price before executing the prediction...")
                current_price = self.trader.get_current_price()  # Get the current price from the Trader
                print(f"Current price now is: {current_price}")
                if current_price is None:
                    print("Failed to get current price. Skipping this cycle.")
                    return

                # Check if the price change is greater than PREDICT_IN_BANDWIDTH%
                bandwidth_price_change = self.calculate_band_price_change(all_features)
                price_is_over_band = self.price_is_over_band(all_features, current_price)
                print(f"Current Bandwidth Price Change is: {round(bandwidth_price_change, 2)}% ")
                if  bandwidth_price_change > PREDICT_IN_BANDWIDTH:
                    print(f"Price change is greater than {PREDICT_IN_BANDWIDTH}%, processing prediction.")
                    logging.info(f"Price change is greater than {PREDICT_IN_BANDWIDTH}%, processing prediction.")

                    print("Generating prediction...")
                    prediction, explanation = self.predictor.get_prediction(all_features, current_price)
                    self.log_time("Prediction generation", start_time)

                    # Log the explanation from ChatGPT
                    print(f"Prediction: ///{prediction}///.")
                    logging.info(f"Prediction: {prediction}. Explanation: {explanation}")

                    # Proceed with the rest of the logic for making a trading decision
                else:
                    if bandwidth_price_change < PREDICT_IN_BANDWIDTH:
                        print(f"X/Price change is less than {PREDICT_IN_BANDWIDTH}%, skipping prediction and returning 'Hold'/X.")
                        logging.info("Price change is less than or equal to 95%, skipping prediction and returning 'Hold'.")
                        prediction = "Hold"
                    else:
                        print(f"X/current price is below lower band, skipping prediction and returning 'Hold'/X.")
                        logging.info("current price is below lower band, skipping prediction and returning 'Hold'.")
                        prediction = "Hold"


                # update the current price right before executing the trading
                print("Updating current price before executing the trade...")
                current_price = self.trader.get_current_price()  # Get the current price from the Trader
                print(f"Current price now is: {current_price}")
                if current_price is None:
                    print("Failed to get current price. Skipping this cycle.")
                    return

                # Convert USDT amount to cryptocurrency amount
                crypto_amount = self.convert_usdt_to_crypto(current_price, USDT_AMOUNT)
                print(f"Converted {USDT_AMOUNT} USDT to {crypto_amount} {COIN}")
                logging.info(f"Converted {USDT_AMOUNT} USDT to {crypto_amount} {COIN}")

                # Make the decision and get dynamic stop_loss and take_profit
                final_decision, adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit = self.decision_maker.make_decision(
                    prediction, current_price, None, all_features)

                # Log and print dynamic stop_loss and take_profit
                print(f"Dynamic Stop Loss Lower: {round(adjusted_stop_loss_lower, 1)}, Dynamic Stop Loss Middle: {round(adjusted_stop_loss_middle, 1)}, Dynamic Take Profit: {round(adjusted_take_profit, 5)}")
                logging.info(f"Dynamic Stop Loss Lower: {round(adjusted_stop_loss_lower, 1)}, Dynamic Stop Loss Middle: {round(adjusted_stop_loss_middle, 1)}, Dynamic Take Profit: {round(adjusted_take_profit, 5)}")

                if prediction == "Buy" and final_decision == "Hold":
                    self.notifier.send_notification("Decision Maker", "Decision Maker hold the Buy Prediction")

                if final_decision == "Buy":
                    # Execute buy trade and save position
                    start_time = time.time()
                    trade_status, order_details = self.trader.execute_trade(final_decision, crypto_amount)
                    self.log_time("Trade execution (Buy)", start_time)

                    if trade_status == "Success":
                        position_id = str(int(time.time()))
                        self.position_manager.add_position(position_id, current_price, crypto_amount)
                        print(f"New position added: {position_id}, Entry Price: {current_price}, Amount: {crypto_amount}")
                        self.notifier.send_notification("Trade Executed", f"Bought {crypto_amount} {COIN} at ${current_price}")
                    else:
                        error_message = f"Failed to execute Buy order: {order_details}"
                        self.save_error_to_csv(error_message)
                        self.notifier.send_notification("Trade Error", error_message)

                elif prediction in ["Hold", "Sell"]:
                    # Iterate over a copy of the positions to avoid the runtime error
                    positions_copy = list(self.position_manager.get_positions().items())
                    for position_id, position,amount in positions_copy:
                        entry_price = position['entry_price']
                        amount = position['amount']

                        start_time = time.time()
                        final_decision, adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit = self.decision_maker.make_decision(
                            prediction, current_price, entry_price, all_features)

                        if final_decision == "Sell":
                            self.log_time("Decision making (Sell)", start_time)
                            start_time = time.time()
                            print(f"Executing trade: {final_decision}")
                            trade_status, order_details = self.trader.execute_trade(final_decision, amount)
                            self.log_time("Trade execution (Sell)", start_time)

                            if trade_status == "Success":
                                gain_loose = self.calculate_gain_loose(entry_price, current_price)
                                profit_usdt = self.calculate_profit(trade_quantity=amount, sold_price=current_price,
                                                                    entry_price=entry_price)
                                self.position_manager.remove_position(position_id)
                                print(f"Position sold: {position_id}, Sell Price: {current_price}, Amount: {amount}")
                                self.notifier.send_notification("Trade Executed", f"Sold {amount} {COIN} at ${current_price}")
                                print(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")
                                logging.info(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")
                                # Log the sold position to the trading log CSV
                                self.log_sold_position(position_id, entry_price, current_price, profit_usdt, round(gain_loose, 2))
                            else:
                                error_message = f"Failed to execute Sell order: {order_details}"
                                self.save_error_to_csv(error_message)
                                self.notifier.send_notification("Trade Error", error_message)
                        else:
                            print("Decision Maker suggested to Hold. No trade action taken.")
                            print(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")
                            logging.info("Decision Maker suggested to Hold. No trade action taken.")
                            logging.info(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")

                else:  # This case is for "Hold"
                    print("Decision Maker suggested to Hold. No trade action taken.")
                    # Optional: Log or notify the hold prediction
                    # self.notifier.send_notification("Hold Decision", "No trade executed. The Predictor advised to hold.")

                break

            except Exception as e:
                attempt += 1
                error_message = str(e)
                logging.error(f"An error occurred during the run (Attempt {attempt}): {error_message}")
                self.save_error_to_csv(error_message)
                # Skip notification if the error is "TypeError: unsupported format string passed to NoneType.__format__"
                if "unsupported format string passed to NoneType.__format__" not in error_message:
                    self.notifier.send_notification("Bot Error",
                                                    f"An error occurred: {error_message}. Attempt {attempt} of 3.")

                print(f"An error occurred. Restarting cycle in 5 seconds... (Attempt {attempt} of 3)")
                time.sleep(5)

                if attempt >= 3:
                    self.notifier.send_notification("Bot Stopped", "The bot encountered repeated errors and is stopping.")
                    print("Bot has stopped due to repeated errors.")
                    raise


    def start(self):
        try:
            # Run immediately
            self.run()

            # Schedule the bot to run at the specified interval
            schedule.every(BOT_CYCLE).seconds.do(self.run)

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



