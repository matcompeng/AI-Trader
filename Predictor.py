import os
import pandas as pd
from datetime import datetime
import logging
import time

from ChatGPTClient import ChatGPTClient
from DataCollector import DataCollector
from FeatureProcessor import FeatureProcessor

class Predictor:
    def __init__(self, chatgpt_client, data_directory='data', max_retries=3, retry_delay=5):
        self.chatgpt_client = chatgpt_client
        self.data_directory = data_directory
        self.max_retries = max_retries  # Maximum number of retries
        self.retry_delay = retry_delay  # Delay in seconds between retries

        # Ensure the data directory exists
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def format_prompt(self, all_features):
        prompt = "Here is the current market data across different intervals:\n\n"

        for interval, features in all_features.items():
            if features:
                interval_prompt = (
                    f"\nInterval: {interval}\n"
                    f"Price Change: {features.get('price_change', 'N/A'):.2f}%\n"
                    f"RSI: {features.get('RSI', 'N/A'):.2f}\n"
                    f"SMA (7): {features.get('SMA_7', 'N/A'):.2f}\n"
                    f"SMA (25): {features.get('SMA_25', 'N/A'):.2f}\n"
                    f"SMA (100): {features.get('SMA_100', 'N/A'):.2f}\n"
                    f"EMA (7): {features.get('EMA_7', 'N/A'):.2f}\n"
                    f"EMA (25): {features.get('EMA_25', 'N/A'):.2f}\n"
                    f"EMA (100): {features.get('EMA_100', 'N/A'):.2f}\n"
                    # f"MACD: {features.get('MACD', 'N/A'):.2f}\n"
                    # f"MACD Signal: {features.get('MACD_signal', 'N/A'):.2f}\n"
                    f"Bollinger Bands: {features.get('upper_band', 'N/A'):.2f}, {features.get('middle_band', 'N/A'):.2f}, {features.get('lower_band', 'N/A'):.2f}\n"
                    # f"ADX: {features.get('ADX', 'N/A'):.2f}\n"
                    f"Stochastic RSI %K: {features.get('stoch_rsi_k', 'N/A'):.2f}\n"
                    f"Stochastic RSI %D: {features.get('stoch_rsi_d', 'N/A'):.2f}\n"
                    f"ATR: {features.get('ATR', 'N/A'):.2f}\n"
                    f"VWAP: {features.get('VWAP', 'N/A'):.2f}\n"
                    # f"OBV: {features.get('OBV', 'N/A'):.2f}\n"
                    f"Support Level: {features.get('support_level', 'N/A'):.2f}\n"
                    f"Resistance Level: {features.get('resistance_level', 'N/A'):.2f}\n"
                )

                interval_prompt += (
                    f"Top Bid: {features.get('top_bid', 'N/A'):.2f}\n"
                    f"Top Ask: {features.get('top_ask', 'N/A'):.2f}\n"
                    f"Bid-Ask Spread: {features.get('bid_ask_spread', 'N/A'):.2f}\n"
                    f"Bid Volume: {features.get('bid_volume', 'N/A'):.2f}\n"
                    f"Ask Volume: {features.get('ask_volume', 'N/A'):.2f}\n\n"
                )
                prompt += interval_prompt

        prompt += (
            "I am looking to trade in the short and intermediate term within a day. "
            "Based on this data from multiple intervals, please provide a single, clear recommendation (Buy, Sell, or Hold) for BTC. "
            "Ensure that your explanation supports your recommendation. Avoid any conflicting suggestions."
        )
        return prompt

    def save_prompt(self, prompt):
        try:
            file_path = os.path.join(self.data_directory, 'latest_prompt.csv')
            df = pd.DataFrame([{"prompt": prompt}])
            df.to_csv(file_path, mode='w', header=True, index=False)
            print(f"Prompt saved to {file_path}")
        except Exception as e:
            print(f"Error saving prompt to CSV: {e}")

    def save_response(self, decision, explanation):
        try:
            file_path = os.path.join(self.data_directory, 'predictions.csv')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df = pd.DataFrame([{"timestamp": timestamp, "prediction": decision, "explanation": explanation}])
            df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            print(f"Prediction and explanation saved to {file_path}")
        except Exception as e:
            print(f"Error saving response to CSV: {e}")

    def get_prediction(self, all_features):
        prompt = self.format_prompt(all_features)
        self.save_prompt(prompt)
        for attempt in range(self.max_retries):
            try:
                response = self.chatgpt_client.get_prediction(prompt)
                decision, explanation = self.interpret_response(response)
                self.save_response(decision, explanation)
                return decision, explanation
            except Exception as e:
                if "Request timed out" in str(e):
                    logging.error("Error in ChatGPT API call: Request timed out.")
                    print("Error in ChatGPT API call: Request timed out. Retrying...")
                else:
                    logging.error(f"Error during communication with OpenAI: {e}")
                    print(f"Attempt {attempt + 1} failed. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

        # If all retries fail
        logging.error("All attempts to communicate with OpenAI failed. Skipping this cycle.")
        return "Hold", "Failed to get a response from OpenAI after multiple attempts."

    def interpret_response(self, response):
        if response is None:
            return "Hold", "No response from ChatGPT."

        response = response.lower()
        explanation = response  # Capture the entire response for logging

        # Keywords to identify Buy, Sell, and Hold
        buy_keywords = ["buy", "bullish"]
        sell_keywords = ["sell", "bearish"]
        hold_keywords = ["hold", "neutral"]

        # Count occurrences of keywords
        buy_count = sum([response.count(word) for word in buy_keywords])
        sell_count = sum([response.count(word) for word in sell_keywords])
        hold_count = sum([response.count(word) for word in hold_keywords])

        # Determine the final recommendation based on keyword counts
        if buy_count > sell_count and buy_count > hold_count:
            final_decision = "Buy"
        elif sell_count > buy_count and sell_count > hold_count:
            final_decision = "Sell"
        else:
            final_decision = "Hold"

        # Check for contradictions in the explanation
        if final_decision == "Buy" and "hold" in response:
            logging.warning("Contradictory information found: Recommendation is 'Buy' but explanation suggests 'Hold'.")
            final_decision = "Hold"  # Default to Hold in case of contradiction
        elif final_decision == "Sell" and "hold" in response:
            logging.warning("Contradictory information found: Recommendation is 'Sell' but explanation suggests 'Hold'.")
            final_decision = "Hold"

        return final_decision, explanation


# Example usage:
if __name__ == "__main__":
    # Initialize the ChatGPT client and the Predictor
    chatgpt_client = ChatGPTClient()
    predictor = Predictor(chatgpt_client)

    # Initialize the DataCollector and FeatureProcessor
    api_key = 'your_binance_api_key'
    api_secret = 'your_binance_api_secret'

    data_collector = DataCollector(api_key, api_secret, intervals=['1m', '5m', '15m', '1h', '1d'])
    market_data = data_collector.collect_data()

    if market_data is not None:
        # Process features from the collected market data
        feature_processor = FeatureProcessor()
        all_features = feature_processor.process(market_data)

        if all_features:
            # Get the prediction using the processed features
            decision, explanation = predictor.get_prediction(all_features)

            # Print the results
            print(f"Prediction: {decision}")
            print(f"Explanation: {explanation}")
        else:
            print("Failed to process features.")
    else:
        print("Failed to collect market data.")