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
                    f"Interval: {interval}\n"
                    f"Price Change: {features['price_change']:.2f}%\n"
                    f"RSI: {features['RSI']:.2f}\n"
                    f"SMA (30): {features['SMA_30']:.2f}\n"
                    f"SMA (100): {features['SMA_100']:.2f}\n"
                    f"EMA (30): {features['EMA_30']:.2f}\n"
                    f"EMA (100): {features['EMA_100']:.2f}\n"
                    f"MACD: {features['MACD']:.2f}\n"
                    f"MACD Signal: {features['MACD_signal']:.2f}\n"
                    f"Bollinger Bands: {features['upper_band']:.2f}, {features['middle_band']:.2f}, {features['lower_band']:.2f}\n"
                    f"ADX: {features['ADX']:.2f}\n"
                    f"Stochastic %K: {features['stoch_k']:.2f}\n"
                    f"Stochastic %D: {features['stoch_d']:.2f}\n"
                    f"ATR: {features['ATR']:.2f}\n" 
                    f"Support Level: {features['support_level']:.2f}\n"
                    f"Resistance Level: {features['resistance_level']:.2f}\n"
                    f"Top Bid: {features['top_bid']:.2f}\n"
                    f"Top Ask: {features['top_ask']:.2f}\n"
                    f"Bid-Ask Spread: {features['bid_ask_spread']:.2f}\n"
                    f"Bid Volume: {features['bid_volume']:.2f}\n"
                    f"Ask Volume: {features['ask_volume']:.2f}\n\n"
                )
                prompt += interval_prompt

        prompt += (
            "Based on this data from multiple intervals, "
            "please provide a single, clear recommendation (Buy, Sell, or Hold) for BTC. "
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

    data_collector = DataCollector(api_key, api_secret, intervals=['1m', '5m', '15m', '1h'])
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