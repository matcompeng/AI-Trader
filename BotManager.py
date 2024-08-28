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

# Bot Configurations -----------------------------------------------------------
BOT_INTERVAL = 3 * 60       # Time in seconds between each run of the Bot Cycle
COIN = 'BNB'
TRADING_PAIR = 'BNBUSDT'
TRADING_INTERVALS = ['1m', '5m', '15m', '1h', '1d']  ## '8h', '12h', '1d']
USDT_AMOUNT = 10          # Amount of Currency to trade for each Position
# ------------------------------------------------------------------------------

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
        self.predictor = Predictor(self.chatgpt_client, coin=COIN)
        self.decision_maker = DecisionMaker()
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

    def run(self):
        attempt = 0
        while attempt < 3:
            try:
                # Capture the current timestamp at the start of the cycle
                cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n\n***Bot cycle started at {cycle_start_time}, running every {BOT_INTERVAL} seconds.***")
                logging.info(f"//-----------------Bot cycle started at {cycle_start_time}-----------------//")

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

                print("Generating prediction...")
                prediction, explanation = self.predictor.get_prediction(all_features)
                self.log_time("Prediction generation", start_time)

                # Log the explanation from ChatGPT
                print(f"Prediction: ///{prediction}///.")
                logging.info(f"Prediction: {prediction}. Explanation: {explanation}")

                # Get the current price right before executing the trade prediction
                print("Getting current price...")
                current_price = self.trader.get_current_price()  # Get the current price from the Trader
                print(f"Current price now is: {current_price}")
                if current_price is None:
                    print("Failed to get current price. Skipping this cycle.")
                    return

                # Convert USDT amount to cryptocurrency amount
                crypto_amount = self.convert_usdt_to_crypto(current_price, USDT_AMOUNT)
                print(f"Converted {USDT_AMOUNT} USDT to {crypto_amount} {COIN}")
                logging.info(f"Converted {USDT_AMOUNT} USDT to {crypto_amount} {COIN}")

                # Make the decision and get adjusted stop_loss and take_profit
                final_decision, adjusted_stop_loss, adjusted_take_profit = self.decision_maker.make_decision(
                    prediction, current_price, None, all_features)

                # Log and print adjusted stop_loss and take_profit
                print(f"Dynamic Stop Loss: {round(adjusted_stop_loss, 5)}, Dynamic Take Profit: {round(adjusted_take_profit, 5)}")
                logging.info(f"Dynamic Stop Loss: {round(adjusted_stop_loss, 5)}, Dynamic Take Profit: {round(adjusted_take_profit, 5)}")

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

                    if prediction == "Buy" and final_decision == "Hold":
                        self.notifier.send_notification("Decision Maker", "Decision Maker hold the Buy Prediction")

                elif prediction in ["Hold", "Sell"]:
                    # Iterate over a copy of the positions to avoid the runtime error
                    positions_copy = list(self.position_manager.get_positions().items())
                    for position_id, position in positions_copy:
                        entry_price = position['entry_price']
                        amount = position['amount']

                        start_time = time.time()
                        final_decision, adjusted_stop_loss, adjusted_take_profit = self.decision_maker.make_decision(
                            prediction, current_price, entry_price, all_features)

                        if final_decision == "Sell":
                            self.log_time("Decision making (Sell)", start_time)
                            start_time = time.time()
                            print(f"Executing trade: {final_decision}")
                            trade_status, order_details = self.trader.execute_trade(final_decision, amount)
                            self.log_time("Trade execution (Sell)", start_time)

                            if trade_status == "Success":
                                self.position_manager.remove_position(position_id)
                                print(f"Position sold: {position_id}, Sell Price: {current_price}, Amount: {amount}")
                                self.notifier.send_notification("Trade Executed", f"Sold {amount} {COIN} at ${current_price}")
                            else:
                                error_message = f"Failed to execute Sell order: {order_details}"
                                self.save_error_to_csv(error_message)
                                self.notifier.send_notification("Trade Error", error_message)
                        else:
                            print(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}")

                else:  # This case is for "Hold"
                    print("Predictor suggested to Hold. No trade action taken.")
                    # Optional: Log or notify the hold prediction
                    # self.notifier.send_notification("Hold Decision", "No trade executed. The Predictor advised to hold.")

                break


            except Exception as e:
                attempt += 1
                error_message = str(e)
                logging.error(f"An error occurred during the run (Attempt {attempt}): {error_message}")

                # Skip notification if the error is "TypeError: unsupported format string passed to NoneType.__format__"
                if "unsupported format string passed to NoneType.__format__" not in error_message:
                    self.save_error_to_csv(error_message)
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
            schedule.every(BOT_INTERVAL).seconds.do(self.run)

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



