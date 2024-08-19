import json
import os
import time
import schedule
import logging
from DataCollector import DataCollector
from FeatureProcessor import FeatureProcessor
from ChatGPTClient import ChatGPTClient
from Predictor import Predictor
from DecisionMaker import DecisionMaker
from Trader import Trader
from Notifier import Notifier

# Global variables
INTERVAL = 1 * 5 # Time in seconds between each run of the bot
AMOUNT = 0.0001  # Amount of BTC to trade

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
        self.data_collector = DataCollector()
        self.feature_processor = FeatureProcessor()
        self.chatgpt_client = ChatGPTClient()
        self.predictor = Predictor(self.chatgpt_client)
        self.decision_maker = DecisionMaker()
        self.trader = Trader()
        self.notifier = Notifier()
        self.position_manager = PositionManager()

    def run(self):
        print("Collecting market data...")
        market_data = self.data_collector.collect_data()

        if not market_data:
            print("Failed to collect market data. Skipping this cycle.")
            return

        print("Processing features...")
        features = self.feature_processor.process(market_data)

        if not features:
            print("Failed to process features. Skipping this cycle.")
            return

        print("Generating prediction...")
        decision, explanation = self.predictor.get_prediction(features)

        # Log the explanation from ChatGPT
        logging.info(f"Prediction: {decision}. Explanation: {explanation}")

        current_price = market_data['last_price']

        if decision == "Buy":
            # Buy logic
            print(f"Executing trade: {decision}")
            self.trader.execute_trade(decision, AMOUNT)

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

                final_decision = self.decision_maker.make_decision(decision, current_price, entry_price)

                if final_decision == "Sell":
                    print(f"Executing trade: {final_decision}")
                    self.trader.execute_trade(final_decision, amount)
                    self.position_manager.remove_position(position_id)
                    print(f"Position sold: {position_id}, Sell Price: {current_price}, Amount: {amount}")
                    self.notifier.send_notification("Trade Executed", f"Sold {amount} BTC at ${current_price}")
                else:
                    print(f"Holding position: {position_id}, Entry Price: {entry_price}, Current Price: {current_price}")

        else:  # This case is for "Hold"
            print("Predictor suggested to Hold. No trade action taken.")
            # Optional: Log or notify the hold decision
            # self.notifier.send_notification("Hold Decision", "No trade executed. The Predictor advised to hold.")

    def start(self):
        # Schedule the bot to run at the specified interval
        schedule.every(INTERVAL).seconds.do(self.run)

        print(f"Bot started, running every {INTERVAL} seconds.")

        while True:
            schedule.run_pending()
            time.sleep(1)


# Example usage:
if __name__ == "__main__":
    bot_manager = BotManager()
    bot_manager.start()