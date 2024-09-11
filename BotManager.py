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

# Bot Configurations ------------------------------------------------------------------------------
FEATURES_INTERVALS = ['1m', '5m', '15m', '30m', '1h', '1d']
COIN = 'BNB'                    # Select Cryptocurrency
TRADING_PAIR = 'BNBUSDT'        # Select Cryptocurrency Trading Pair
TRADING_INTERVAL = '1h'         # Select The Interval That Activate/Deactivate Predictor through PREDICT_IN_BANDWIDTH
PROFIT_INTERVAL = '1h'          # Select The Interval For Take Profit Calculations
LOOSE_INTERVAL = '1h'           # Select The Interval For Stop Loose Calculations
SR_INTERVAL = '1h'              # Select The Interval That Trader Define Support and Resistance Levels
DIP_INTERVAL = '1h'             # Select The Interval For Buying a Dip
POSITION_CYCLE = 15             # Time in seconds to check positions
PREDICTION_CYCLE = 5 * 60       # Time in seconds to run the Prediction bot cycle
INTERVAL_BANDWIDTH = '5m'       # Define The Interval To calculate Prediction Bandwidth
PREDICT_BANDWIDTH = 0.60        # Define Minimum Bandwidth % to Activate Trading
BASE_TAKE_PROFIT = 0.30         # Define Base Take Profit Percentage %
BASE_STOP_LOSS = 0.10           # Define Base Stop Loose  Percentage %
CAPITAL_AMOUNT = 500            # Your Capital Investment
RISK_TOLERANCE = 0.03           # The Portion Amount you want to take risk of capital for each Buying position
AMOUNT_RSI_INTERVAL = '5m'      # Interval To get its RSI for Buying Amount Calculations Function
AMOUNT_ATR_INTERVAL = '15m'     # Interval To get its ATR for Buying Amount Calculations Function
USDT_DIP_AMOUNT = 5             # Amount of Currency For Buying a Dip
MIN_STABLE_INTERVALS = 5        # Set The Minimum Stable Intervals For Market Stable Condition
GAIN_SELL_THRESHOLD = 0.25      # Set the Sell Threshold % for Stable Portfolio Gain Reversal
CHECK_POSITIONS_ON_BUY = True   # Set True If You Need Bot Manager Check The Positions During Buy Cycle
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
        self.predictor = Predictor(self.chatgpt_client, coin=COIN, sr_interval=SR_INTERVAL)
        self.decision_maker = DecisionMaker(base_take_profit=BASE_TAKE_PROFIT, base_stop_loss=BASE_STOP_LOSS,
                                            profit_interval=PROFIT_INTERVAL, loose_interval=LOOSE_INTERVAL,
                                            dip_interval=DIP_INTERVAL, risk_tolerance=RISK_TOLERANCE,
                                            amount_atr_interval=AMOUNT_ATR_INTERVAL,
                                            amount_rsi_interval=AMOUNT_RSI_INTERVAL,
                                            min_stable_intervals=MIN_STABLE_INTERVALS,
                                            gain_sell_threshold=GAIN_SELL_THRESHOLD)
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

    def calculate_prediction_bandwidth(self, all_features):
        """
        Calculate the percentage price change between the upper and lower Bollinger Bands of the PREDICTOR_INTERVAL interval.
        """
        upper_band = all_features[INTERVAL_BANDWIDTH].get('upper_band')
        lower_band = all_features[INTERVAL_BANDWIDTH].get('lower_band')

        if upper_band and lower_band:
            price_change = ((upper_band - lower_band) / lower_band) * 100
            return price_change

    def price_is_over_band(self, all_features, current_price):

        lower_band_15m = all_features[TRADING_INTERVAL].get('lower_band')
        if current_price >= lower_band_15m:
            return True

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
        1. stable_investment: Invested amount where dip_flag != position['dip']
        2. dip_investment: Invested amount where dip_flag == position['dip']
        3. total_investment: Total invested amount for all positions
        :return: stable_investment, dip_investment, total_investment
        """
        try:
            # Get all positions
            positions = self.position_manager.get_positions()
            total_invested = 0.0
            stable_investment = 0.0
            dip_investment = 0.0

            # Iterate over each position and calculate the invested amount
            for position_id, position in positions.items():
                entry_price = float(position['entry_price'])
                amount = float(position['amount'])
                invested_amount = entry_price * amount
                total_invested += invested_amount

                # Check the dip flag and categorize the investment
                if position['dip'] == 1:
                    dip_investment += invested_amount
                else:
                    stable_investment += invested_amount

            return stable_investment, dip_investment, total_invested

        except Exception as e:
            logging.error(f"Error calculating invested budget: {e}")
            print(f"Error calculating invested budget: {e}")
            return 0.0, 0.0, 0.0

    def check_stable_positions(self):
        try:
            start_time = time.time()
            cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n*****Stable Position check cycle started at {cycle_start_time}.*****")
            logging.info(
                f"//---------------------Stable Position check cycle started at {cycle_start_time}--------------------//")

            # Taking Bot Manager Class Instance
            bot_manager = BotManager()

            # Get features and make a decision on whether to sell
            market_data = self.data_collector.collect_data()
            all_features = self.feature_processor.process(market_data)
            total_invested, stable_invested, dip_invested = self.invested_budget()

            current_price = self.trader.get_current_price()
            if current_price is None:
                print("Failed to get current price. Skipping position check.")
                logging.info("Failed to get current price. Skipping position check.")
                return

            reversed_decision = self.decision_maker.check_for_sell_due_to_reversal(bot_manager, current_price)

            if reversed_decision == "Sell":
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
                            total_invested, stable_invested, dip_invested = self.invested_budget()
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
                                                            f"Stable Invested: {round(stable_invested)} USDT\n"
                                                            f"Dip Invested: {round(dip_invested)} USDT\n"
                                                            f"Total Invested: {round(total_invested)} USDT")
                        else:
                            error_message = f"Failed to execute Sell order: {order_details}"
                            self.save_error_to_csv(error_message)
                            self.notifier.send_notification("Trade Error", error_message)
            else:
                # Iterate over a copy of the positions to avoid runtime errors
                positions_copy = list(self.position_manager.get_positions().items())
                for position_id, position in positions_copy:
                    entry_price = position['entry_price']
                    amount = position['amount']
                    dip_flag = position['dip']

                    if not all_features:
                        print("Failed to process features for position check.")
                        logging.info("Failed to process features for position check.")
                        return

                    final_decision, adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit = self.decision_maker.make_decision(
                        "Hold", current_price, entry_price, all_features)
                    gain_loose = round(self.calculate_gain_loose(entry_price, current_price), 2)

                    if final_decision == "Sell" and dip_flag == 0:
                        trade_type = 'Stable'
                        print(f"Selling position {position_id}")
                        logging.info(f"Selling position {position_id}")
                        trade_status, order_details = self.trader.execute_trade(final_decision, amount)
                        if trade_status == "Success":
                            total_invested, stable_invested, dip_invested = self.invested_budget()
                            profit_usdt = self.calculate_profit(trade_quantity=amount, sold_price=current_price,
                                                                entry_price=entry_price)
                            self.position_manager.remove_position(position_id)
                            self.log_sold_position(position_id, trade_type, entry_price, current_price, profit_usdt, gain_loose)
                            print(f"Position {position_id} sold successfully")
                            logging.info(f"Position {position_id} sold successfully")
                            self.notifier.send_notification("Stable Trade Executed", f"Sold {amount} {COIN} at ${current_price}\n"
                                                                              f"Gain/Loose: {gain_loose}%\n"
                                                                              f"Stable Invested: {round(stable_invested)} USDT\n"
                                                                              f"Dip Invested: {round(dip_invested)} USDT\n"
                                                                              f"Total Invested: {round(total_invested)} USDT")
                        else:
                            error_message = f"Failed to execute Sell order: {order_details}"
                            self.save_error_to_csv(error_message)
                            self.notifier.send_notification("Trade Error", error_message)
                    else:
                        print(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, ((Gain/Loose: {gain_loose}%))")
                        logging.info(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, ((Gain/Loose: {gain_loose}%))")
                        print(f"dynamic_stop_loss_lower: {round(adjusted_stop_loss_lower, 2)}%, dynamic_stop_loss_middle: {round(adjusted_stop_loss_middle, 2)}%, dynamic_take_profit: {round(adjusted_take_profit, 2)}%\n")
                        logging.info(f"dynamic_stop_loss_lower: {round(adjusted_stop_loss_lower, 2)}%, dynamic_stop_loss_middle: {round(adjusted_stop_loss_middle, 2)}%, dynamic_take_profit: {round(adjusted_take_profit, 2)}\n%")


                print(f"Stable Invested: {round(stable_invested)} USDT\n"
                                          f"Dip Invested: {round(dip_invested)} USDT\n"
                                          f"Total Invested: {round(total_invested)} USDT")
                logging.info(f"Stable Invested: {round(stable_invested)} USDT\n"
                                          f"Dip Invested: {round(dip_invested)} USDT\n"
                                          f"Total Invested: {round(total_invested)} USDT")
                self.log_time("Position check", start_time)


        except Exception as e:
            logging.error(f"An error occurred during position check: {str(e)}")
            self.save_error_to_csv(str(e))

    def save_historical_context_for_stable(self):
        """
        Saves historical context of the processed features for a specific interval,
        while ensuring that only the latest 1 day of data is stored. Data older than
        1 day will be cleared.
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

            # Filter the data to only keep entries from the last 24 hours
            one_day_ago = current_time - timedelta(days=1)
            historical_data = [entry for entry in historical_data
                               if datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') > one_day_ago]

            # Save the updated historical data back to the JSON file
            with open(historical_file, 'w') as file:
                json.dump(historical_data, file, indent=4)

            print(f"Historical context for interval '{TRADING_INTERVAL}' saved successfully.")
            logging.info(f"Historical context for interval '{TRADING_INTERVAL}' saved successfully.")
        except Exception as e:
            print(f"Error saving historical context for interval '{TRADING_INTERVAL}': {e}")
            logging.info(f"Error saving historical context for interval '{TRADING_INTERVAL}': {e}")

    def check_dip_flag(self):
        positions_copy = list(self.position_manager.get_positions().items())
        for position_id, position in positions_copy:
            dip_flag = position['dip']
            if dip_flag == 1:
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

            print(f"Historical context for interval '{DIP_INTERVAL}' saved successfully.")
            logging.info(f"Historical context for interval '{DIP_INTERVAL}' saved successfully.")
        except Exception as e:
            print(f"Error saving historical context for interval '{DIP_INTERVAL}': {e}")
            logging.info(f"Error saving historical context for interval '{DIP_INTERVAL}': {e}")


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

                # Check if the price change is greater than PREDICT_IN_BANDWIDTH%
                bandwidth_price_change = self.calculate_prediction_bandwidth(all_features)
                if bandwidth_price_change > PREDICT_BANDWIDTH:
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
                    prediction, current_price, None, all_features)
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
                        total_invested, stable_invested, dip_invested = self.invested_budget()
                        self.position_manager.add_position(position_id, current_price, trading_cryptocurrency_amount, dip_flag=0)
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
                        total_invested, stable_invested, dip_invested = self.invested_budget()
                        self.position_manager.add_position(position_id, current_price, dip_cryptocurrency_amount, dip_flag=1)
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

    def check_dip_positions(self):
        try:
            start_time = time.time()
            cycle_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n*****Dip Position check cycle started at {cycle_start_time}.*****")
            logging.info(
                f"//---------------------Dip Position check cycle started at {cycle_start_time}--------------------//")

            # Loading dip_positions
            if self.check_dip_flag():
                positions_copy = [
                    position for position_key, position in self.position_manager.get_positions().items()
                    if position.get('dip_flag') == 1]

                #Loading Dip Historical context data
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
                                                                        prediction_type='Dip', positions=positions_copy)
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
                            total_invested, stable_invested, dip_invested = self.invested_budget()
                            self.position_manager.remove_position(position_id)
                            self.log_sold_position(position_id, trade_type, entry_price, current_price, profit_usdt, gain_loose)
                            print(f"Position {position_id} sold successfully")
                            logging.info(f"Position {position_id} sold successfully")
                            self.notifier.send_notification("Trade Executed", f"Sold {amount} {COIN} at ${current_price}\n"
                                                                              f"Gain/Loose: {gain_loose}%\n"
                                                                              f"Stable Invested: {round(stable_invested)} USDT\n"
                                                                              f"Dip Invested: {round(dip_invested)} USDT\n"
                                                                              f"Total Invested: {round(total_invested)} USDT")
                        else:
                            error_message = f"Failed to execute Sell order: {order_details}"
                            self.save_error_to_csv(error_message)
                            self.notifier.send_notification("Trade Error", error_message)

                    else:
                        print(
                            f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")
                        logging.info(
                            f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}, Gain/Loose: {gain_loose}%")

            else:
                print("No Dip Entry Founds")
                logging.info("No Dip Entry Founds")


        except Exception as e:
            logging.error(f"An error occurred during position check: {str(e)}")
            self.save_error_to_csv(str(e))


    def start(self):
        try:
            # For testing purposes
            # self.save_historical_context_for_trading()
            # self.run_prediction_cycle()

            # Schedule the position check every POSITION_CYCLE seconds
            schedule.every(POSITION_CYCLE).seconds.do(self.check_stable_positions)

            # Schedule the prediction cycle every PREDICTION_CYCLE seconds
            schedule.every(PREDICTION_CYCLE).seconds.do(self.run_prediction_cycle)

            schedule.every().hour.at(":59").do(self.save_historical_context_for_stable)

            schedule.every().hour.at(":59").do(self.save_historical_context_for_dip)

            schedule.every().hour.do(self.check_dip_positions)

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



