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

# Global variables
INTERVAL = 5 * 60  # Time in seconds between each run of the bot
AMOUNT = 0.0002  # Amount of BTC to trade

# Configure logging
logging.basicConfig(filename='bot_manager.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PositionManager:
    def __init__(self, positions_file='positions.json'):
        self.positions_file = positions_file
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
        self.positions[position_id] = {
            'entry_price': entry_price,
            'amount': amount
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

        # Use multiple intervals
        intervals = ['1m', '5m', '15m', '1h', '1d']

        self.data_collector = DataCollector(api_key, api_secret, intervals=intervals)
        self.feature_processor = FeatureProcessor()
        self.chatgpt_client = ChatGPTClient()
        self.predictor = Predictor(self.chatgpt_client)
        self.decision_maker = DecisionMaker()
        self.trader = Trader()
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
            file_path = os.path.join('error_logs.csv')

            # Append to the CSV file
            df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            print(f"Error details saved to {file_path}")
        except Exception as e:
            logging.error(f"Failed to save error details to CSV: {e}")

    def run(self):
        attempt = 0
        while attempt < 3:
            try:
                start_time = time.time()
                print(f"\n\nBot started, running every {INTERVAL} seconds.")
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

                start_time = time.time()
                print("Generating prediction...")
                decision, explanation = self.predictor.get_prediction(all_features)
                self.log_time("Prediction generation", start_time)

                # Log the explanation from ChatGPT
                logging.info(f"Prediction: {decision}. Explanation: {explanation}")

                current_price = market_data['1m']['last_price']

                if decision == "Buy":
                    # Buy logic
                    start_time = time.time()
                    print(f"Executing trade: {decision}")
                    self.trader.execute_trade(decision, AMOUNT)
                    self.log_time("Trade execution (Buy)", start_time)

                    # Save the new position
                    position_id = str(int(time.time()))  # Use timestamp as a unique ID
                    self.position_manager.add_position(position_id, current_price, AMOUNT)
                    print(f"New position added: {position_id}, Entry Price: {current_price}, Amount: {AMOUNT}")
                    self.notifier.send_notification("Trade Executed", f"Bought {AMOUNT} BTC at ${current_price}")

                elif decision == "Sell":
                    # Sell logic
                    positions = self.position_manager.get_positions()
                    for position_id, position in positions.items():
                        entry_price = position['entry_price']
                        amount = position['amount']

                        start_time = time.time()
                        final_decision = self.decision_maker.make_decision(decision, current_price, entry_price)
                        self.log_time("Decision making (Sell)", start_time)

                        if final_decision == "Sell":
                            start_time = time.time()
                            print(f"Executing trade: {final_decision}")
                            self.trader.execute_trade(final_decision, amount)
                            self.log_time("Trade execution (Sell)", start_time)

                            self.position_manager.remove_position(position_id)
                            print(f"Position sold: {position_id}, Sell Price: {current_price}, Amount: {amount}")
                            self.notifier.send_notification("Trade Executed", f"Sold {amount} BTC at ${current_price}")
                        else:
                            print(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}")

                else:  # This case is for "Hold"
                    print("Predictor suggested to Hold. No trade action taken.")
                    # Optional: Log or notify the hold decision
                    # self.notifier.send_notification("Hold Decision", "No trade executed. The Predictor advised to hold.")

                # If the process completes without errors, break the loop
                break

            except Exception as e:
                attempt += 1
                logging.error(f"An error occurred during the run (Attempt {attempt}): {e}")
                self.save_error_to_csv(str(e))
                self.notifier.send_notification("Bot Error", f"An error occurred: {e}. Attempt {attempt} of 3.")
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
            schedule.every(INTERVAL).seconds.do(self.run)

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
