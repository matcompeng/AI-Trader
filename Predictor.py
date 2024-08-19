import logging
import os
from datetime import datetime

import pandas as pd

from ChatGPTClient import ChatGPTClient
from DataCollector import DataCollector
from FeatureProcessor import FeatureProcessor


class Predictor:
    def __init__(self, chatgpt_client, data_directory='data'):
        self.chatgpt_client = chatgpt_client
        self.data_directory = data_directory

        # Ensure the data directory exists
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def format_prompt(self, features):
        prompt = (
            f"Here is the current market data:\n"
            f"Price Change: {features['price_change']:.2f}%\n"
            f"RSI: {features['RSI']:.2f}\n"
            f"SMA (50): {features['SMA_50']:.2f}\n"
            f"SMA (200): {features['SMA_200']:.2f}\n"
            f"EMA (50): {features['EMA_50']:.2f}\n"
            f"EMA (200): {features['EMA_200']:.2f}\n"
            f"MACD: {features['MACD']:.2f}\n"
            f"MACD Signal: {features['MACD_signal']:.2f}\n"
            f"Bollinger Bands: {features['upper_band']:.2f}, {features['middle_band']:.2f}, {features['lower_band']:.2f}\n"
            f"Support Level: {features['support_level']:.2f}\n"
            f"Resistance Level: {features['resistance_level']:.2f}\n"
            f"Top Bid: {features['top_bid']:.2f}\n"
            f"Top Ask: {features['top_ask']:.2f}\n"
            f"Bid-Ask Spread: {features['bid_ask_spread']:.2f}\n"
            f"Bid Volume: {features['bid_volume']:.2f}\n"
            f"Ask Volume: {features['ask_volume']:.2f}\n\n"
            "Based on this data, should I Buy, Sell, or Hold BTC? Please explain your decision and let me know if the data provided is sufficient for making this decision."
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

    def get_prediction(self, features):
        prompt = self.format_prompt(features)
        self.save_prompt(prompt)
        response = self.chatgpt_client.get_prediction(prompt)
        decision, explanation = self.interpret_response(response)
        self.save_response(decision, explanation)
        return decision, explanation

    def interpret_response(self, response):
        if response is None:
            return "Hold", "No response from ChatGPT."

        response = response.lower()
        explanation = response  # Capture the entire response for logging

        if "buy" in response:
            return "Buy", explanation
        elif "sell" in response:
            return "Sell", explanation
        elif "hold" in response:
            return "Hold", explanation
        else:
            return "Hold", explanation  # Default to Hold if the response is unclear


# Example usage:
if __name__ == "__main__":
    # Initialize the ChatGPT client and the Predictor
    chatgpt_client = ChatGPTClient()
    predictor = Predictor(chatgpt_client)

    # Initialize the DataCollector and FeatureProcessor
    api_key = 'your_binance_api_key'
    api_secret = 'your_binance_api_secret'

    data_collector = DataCollector(api_key, api_secret, timeframe='5m')
    market_data = data_collector.collect_data()

    if market_data is not None:
        # Process features from the collected market data
        feature_processor = FeatureProcessor()
        features = feature_processor.process(market_data)

        if features:
            # Get the prediction using the processed features
            decision, explanation = predictor.get_prediction(features)

            # Print the results
            print(f"Prediction: {decision}")
            print(f"Explanation: {explanation}")
        else:
            print("Failed to process features.")
    else:
        print("Failed to collect market data.")