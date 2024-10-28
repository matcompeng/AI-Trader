import os
import pandas as pd
from datetime import datetime
import logging
import time

from ChatGPTClient import ChatGPTClient
from DataCollector import DataCollector
from FeatureProcessor import FeatureProcessor
from Notifier import Notifier

class Predictor:
    def __init__(self, chatgpt_client, data_directory='data', max_retries=3, retry_delay=5, coin=None, bot_manager=None, trading_interval=None, dip_interval=None):
        self.chatgpt_client = chatgpt_client
        self.data_directory = data_directory
        self.max_retries = max_retries  # Maximum number of retries
        self.retry_delay = retry_delay  # Delay in seconds between retries
        self.coin = coin
        self.bot_manager = bot_manager  # Store the bot manager instance
        self.notifier = Notifier()
        self.trading_interval = trading_interval
        self.dip_interval = dip_interval

        # Ensure the data directory exists
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def format_stable_prompt(self, all_features, current_price, historical_data_1,historical_data_2, current_obv):

        #trading strategy section before historical context
        prompt = (
            "### Trading Strategy:\n"
            "1. Historical Context should be used to identify broader trends and key support/resistance levels, Current Market Data should be used to confirm short-term patterns and trigger decisions.\n"
            "2. Buy After Dip Reversal: Consider 'Buy' decisions when the price shows a strong reversal from a dip.\n"
            "   - Ensure upward momentum in key indicators such as:\n"
           f"     - RSI transitioning from oversold (RSI < 30) based on recent historical data of '{self.trading_interval}' or '{self.dip_interval}'.\n"
            "     - Increasing OBV as per Rule 5.\n"
           f"     - Positive MACD Histogram as per Rule 6.\n"
            "   - Use this strategy during high volatility or sudden market dips.\n"
            "3. Buy After Resistance Breakout: Consider 'Buy' decisions if the price closes above the latest significant resistance level and aligns with key technical indicators from Historical Context:\n"
            "   - Confirm breakout with:\n"
            "     - Positive MACD Histogram as per Rule 6.\n"
           f"     - Ensure RSI is below 60 in the current market data interval '{self.dip_interval}'.\n"
            "     - Ensure ADX > 20 in '5m' and '15m' intervals to confirm short-term momentum.\n"
           f"     - Ensure there is an uptrend momentum in the past 24 hours of the '{self.dip_interval}' historical context data.\n"
            "4. Strict RSI Condition:\n"
           f"   - Non-negotiable Rule: Must avoid 'Buy' if RSI exceeds 70 in interval '{self.dip_interval}' of current market data, regardless of other indicators. This rule takes precedence over all other rules, and if triggered, the decision should be 'Hold'.\n"
            "   - Important Note: Even if other indicators (such as MACD or OBV) are strongly bullish, a Buy decision is not allowed if RSI is above 70 in the specified interval.\n"
            "5. Decision Priority:\n"
            "   - **First Priority:** Prioritize **Point 2** (Buy After Dip Reversal) if the market is showing recovery from an oversold condition with upward momentum.\n"
            "   - **Second Priority:** If the conditions in **Point 2** are not present, prioritize **Point 3** (Buy After Resistance Breakout) if the market demonstrates steady momentum and a significant resistance breakout.\n"
            "6. Volume Confirmation with OBV:\n"
           f"   - Confirm price movement with OBV rising in both the recent '{self.trading_interval}' Historical Context data and current market data.\n"
            "   - Divergence between OBV and price suggests weakness, which should be a signal to avoid buying.\n"
            "7. Volume Confirmation with MACD Histogram:\n"
           f"   - The MACD Histogram must either show an increase in the number of consecutive bars above zero (positive) or a reduction in the magnitude of bars below zero (negative). This trend should be present in both the historical and current market data across the intervals ('{self.trading_interval}', '{self.dip_interval}').\n"
            "8. Strict Adherence to Strategy:\n"
            "   - Stick strictly to the rules outlined in this strategy. Do not infer or make any additional suggestions beyond what is explicitly stated."
            )

        # Include the historical context as one line per entry
        if historical_data_1:
            prompt += f"\n\n### Interval({self.trading_interval}) Historical Context:\n"
            for entry in historical_data_1[-24:]:
                historical_prompt = (
                    f"{entry['timestamp']}, "
                        f"Open: {entry['open']:.2f}%, "
                        f"High: {entry['high']:.2f}%, "
                        f"Low: {entry['low']:.2f}%, "
                        f"Close: {entry['close']:.2f}%, "
                        # f"Price Change: {entry['price_change']:.2f}%, "
                        f"RSI: {entry['RSI']:.2f}, "
                        # f"SMA (7): {entry['SMA_7']:.2f}, "
                        f"SMA (25): {entry['SMA_25']:.2f}, "
                        f"SMA (100): {entry['SMA_100']:.2f}, "
                        # f"EMA (7): {entry['EMA_7']:.2f}, "
                        f"EMA (25): {entry['EMA_25']:.2f}, "
                        f"EMA (100): {entry['EMA_100']:.2f}, "
                        # f"MACD: {entry['MACD']:.2f}, "
                        # f"MACD Signal: {entry['MACD_signal']:.2f}, "
                        f"MACD Hist: {entry['MACD_hist']:.2f}, "
                        # f"Bollinger Bands: {entry['upper_band']:.2f}, {entry['middle_band']:.2f}, {entry['lower_band']:.2f}, "
                        # f"StochRSI %K:: {entry['stoch_rsi_k']:.2f}, "
                        # f"StochRSI %D:: {entry['stoch_rsi_d']:.2f}, "
                        f"ADX: {entry['ADX']:.2f}, "
                        # f"ATR: {entry['ATR']:.2f}, "
                        # f"VWAP: {entry['VWAP']:.2f}, "
                        f"OBV: {entry['OBV']:.2f}\n"
                    # f"Support: {entry['support_level']}, "
                    # f"Resistance: {entry['resistance_level']}\n"
                )
                prompt += historical_prompt

            # Include the historical context as one line per entry
            if historical_data_2:
                prompt += f"\n\n### Interval({self.dip_interval}) Historical Context:\n"
                for entry in historical_data_2[-72:]:
                    historical_prompt = (
                        f"{entry['timestamp']}, "
                        f"Open: {entry['open']:.2f}%, "
                        f"High: {entry['high']:.2f}%, "
                        f"Low: {entry['low']:.2f}%, "
                        f"Close: {entry['close']:.2f}%, "
                        # f"Price Change: {entry['price_change']:.2f}%, "
                        f"RSI: {entry['RSI']:.2f}, "
                        # f"SMA (7): {entry['SMA_7']:.2f}, "
                        f"SMA (25): {entry['SMA_25']:.2f}, "
                        f"SMA (100): {entry['SMA_100']:.2f}, "
                        # f"EMA (7): {entry['EMA_7']:.2f}, "
                        f"EMA (25): {entry['EMA_25']:.2f}, "
                        f"EMA (100): {entry['EMA_100']:.2f}, "
                        # f"MACD: {entry['MACD']:.2f}, "
                        # f"MACD Signal: {entry['MACD_signal']:.2f}, "
                        f"MACD Hist: {entry['MACD_hist']:.2f}, "
                        # f"Bollinger Bands: {entry['upper_band']:.2f}, {entry['middle_band']:.2f}, {entry['lower_band']:.2f}, "
                        # f"StochRSI %K:: {entry['stoch_rsi_k']:.2f}, "
                        # f"StochRSI %D:: {entry['stoch_rsi_d']:.2f}, "
                        f"ADX: {entry['ADX']:.2f}, "
                        # f"ATR: {entry['ATR']:.2f}, "
                        # f"VWAP: {entry['VWAP']:.2f}, "
                        f"OBV: {entry['OBV']:.2f}\n"
                        # f"Support: {entry['support_level']}, "
                        # f"Resistance: {entry['resistance_level']}\n"
                    )
                    prompt += historical_prompt


        # Start with the market data header
        prompt += "\n\nHere is the current market data across different intervals:\n"

        # Include the current market data from different intervals
        for interval, features in all_features.items():
            if features:
                interval_prompt = (
                    f"Interval: {interval}\n"
                    f"Open: {features['open']:.2f}%, "
                    f"High: {features['high']:.2f}%, "
                    f"Low: {features['low']:.2f}%, "
                    f"Close: {features['close']:.2f}%, "
                    f"Price Change: {features['price_change']:.2f}%, "
                    f"RSI: {features['RSI']:.2f}, "
                    # f"SMA (7): {features['SMA_7']:.2f}, "
                    # f"SMA (25): {features['SMA_25']:.2f}, "
                    f"SMA (100): {features['SMA_100']:.2f}, "
                    # f"EMA (7): {features['EMA_7']:.2f}, "
                    # f"EMA (25): {features['EMA_25']:.2f}, "
                    f"EMA (100): {features['EMA_100']:.2f}, "
                    f"MACD: {features['MACD']:.2f}, "
                    f"MACD Signal: {features['MACD_signal']:.2f}, "
                    f"MACD Hist: {features['MACD_hist']:.2f}, "
                    f"Bollinger Bands: {features['upper_band']:.2f}, {features['middle_band']:.2f}, {features['lower_band']:.2f}, "  
                    f"StochRSI %K: {features['stoch_rsi_k']:.2f}, "  # Updated to reflect stochRSI
                    f"StochRSI %D: {features['stoch_rsi_d']:.2f}, "  # Updated to reflect stochRSI
                    f"ADX: {features['ADX']:.2f}, "
                    f"ATR: {features['ATR']:.2f}, "
                    f"VWAP: {features['VWAP']:.2f}, "  # Include VWAP in the prompt
                    f"OBV: {features['OBV']:.2f}, "
                    # f"Support Level: {features['support_level']:.2f}, "
                    # f"Resistance Level: {features['resistance_level']:.2f}, "
                )

                interval_prompt += (
                    f"Top Bid/Ask: {features['top_bid']:.2f}/{features['top_ask']:.2f}, "
                    f"Bid-Ask Spread: {features['bid_ask_spread']:.2f}, "
                    f"Bid Volume: {features['bid_volume']:.2f}, "
                    f"Ask Volume: {features['ask_volume']:.2f}\n"
                )
                prompt += interval_prompt

        # Append final instructions for ChatGPT at the end
        prompt += (
            f"\n\nI am looking to trade {self.coin} cryptocurrency in the short term within a day.\n"
            f"Knowing that the current price is: {current_price} for this cycle.\n"
            "Please provide a single, clear recommendation based on Current market data, Historical Context and Trading Strategy (must use format &Buy& or &Hold& for the final recommendation only and it should not include this format in your explanation)."
        )


        return prompt


    def format_dip_prompt(self, dip_positions, current_price, historical_data):

        prompt = "Below are the entry positions recorded during the market dip (each entry is presented on a single line):\n\n"
        # Iterate through the positions and format each entry
        for position_id, position_data in dip_positions:
            timestamp = position_data['timestamp']
            entry_price = position_data['entry_price']
            prompt += f"Timestamp: {timestamp}, Position_id: {position_id},Entry Price: {entry_price}\n"

        # Include historical data as one line per entry
        if historical_data:
            prompt += "\n\nHere is the historical context for the most recent 3 days (each entry is presented on a single line):\n\n"
            for entry in historical_data:
                historical_prompt = (
                    f"{entry['timestamp']}, "
                    f"{entry['price_change']:.2f}%, "
                    f"RSI: {entry['RSI']:.2f}, "
                    f"SMA (7): {entry['SMA_7']:.2f}, "
                    f"SMA (25): {entry['SMA_25']:.2f}, "
                    f"MACD: {entry['MACD']:.2f}, "
                    f"MACD Signal: {entry['MACD_signal']:.2f}, "
                    f"MACD Hist: {entry['MACD_hist']:.2f}, "
                    f"Bollinger Bands: {entry['upper_band']:.2f}, {entry['middle_band']:.2f}, {entry['lower_band']:.2f}, "
                    f"Stoch RSI %K: {entry['stoch_rsi_k']:.2f}, "
                    f"Stoch RSI %D: {entry['stoch_rsi_d']:.2f}, "
                    f"ATR: {entry['ATR']:.2f}, "
                    f"VWAP: {entry['VWAP']:.2f}, "
                    f"Support: {entry['support_level']}, "
                    f"Resistance: {entry['resistance_level']}, "
                    f"Last Price: {entry['last_price']:.2f}\n"
                )
                prompt += historical_prompt

        # Append final instructions for ChatGPT
        prompt += (
            f"\nI have provided the entry positions for my already bought trades, the current market price is {current_price}, "
             "and the historical market data from the past 3 days. Please use the following techniques to analyze the data:\n\n"
             "1. Volatility-Based Levels (using ATR) to determine suitable stop loss and take profit levels.\n"
             "2. Fibonacci Retracement Levels to refine and validate the stop loss and take profit levels.\n\n"
             "Your task is to evaluate the average entry price of all given positions based on the historical data, using the above techniques, "
             "to decide the appropriate course of action for the entire portfolio:\n\n"
             "1. Recommend selling for take profit if the market shows signs of reaching an optimal profit level based on the average entry price.\n"
             "2. Recommend selling for stop loss if the market shows signs of potential further decline below the average entry price.\n\n"
             "Please provide a clear recommendation for the entire portfolio (use &Sell_TP& for Take Profit, &Sell_SL& for Stop Loss, or &Hold&), "
             "including your reasoning based on the technical analysis of the data."
        )

        return prompt

    def save_stable_prompt(self, prompt):
        try:
            file_path = os.path.join(self.data_directory, 'latest_stable_prompt.csv')
            df = pd.DataFrame([{"prompt": prompt}])
            df.to_csv(file_path, mode='w', header=True, index=False)
            print(f"Prompt saved to {file_path}")
        except Exception as e:
            print(f"Error saving stable prompt to CSV: {e}")

    def save_dip_prompt(self, prompt):
        try:
            file_path = os.path.join(self.data_directory, 'latest_dip_prompt.csv')
            df = pd.DataFrame([{"prompt": prompt}])
            df.to_csv(file_path, mode='w', header=True, index=False)
            print(f"Prompt saved to {file_path}")
        except Exception as e:
            print(f"Error saving dip prompt to CSV: {e}")

    def save_stable_response(self, decision, explanation):
        try:
            file_path = os.path.join(self.data_directory, 'stable_predictions.csv')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df = pd.DataFrame([{"timestamp": timestamp, "prediction": decision, "explanation": explanation}])
            df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            print(f"Stable Prediction and explanation saved to {file_path}")
        except Exception as e:
            print(f"Error saving stable response to CSV: {e}")

    def save_dip_response(self, decision, explanation):
        try:
            file_path = os.path.join(self.data_directory, 'dip_predictions.csv')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df = pd.DataFrame([{"timestamp": timestamp, "prediction": decision, "explanation": explanation}])
            df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            print(f"Dip Prediction and explanation saved to {file_path}")
        except Exception as e:
            print(f"Error saving dip response to CSV: {e}")

    def get_prediction(self, all_features=None, current_price=None, historical_data_1=None, historical_data_2=None, prediction_type=None ,positions=None, trading_interval=None):
        prompt = None
        current_obv = None

        if trading_interval is not None:
            current_obv = all_features[trading_interval].get('OBV', None)

        if prediction_type == 'Stable':
            prompt = self.format_stable_prompt(all_features, current_price, historical_data_1, historical_data_2, current_obv)
            self.save_stable_prompt(prompt)
        elif prediction_type == 'Dip':
            prompt = self.format_dip_prompt(positions, current_price, historical_data_2)
            self.save_dip_prompt(prompt)

        for attempt in range(self.max_retries):
            try:
                response = self.chatgpt_client.get_prediction(prompt)

                # Extract the decision from the response
                decision = self.extract_decision_from_response(response)
                explanation = response  # Keep the entire response as the explanation

                if decision and prediction_type == 'Stable':
                    self.save_stable_response(decision, explanation)
                    return decision, explanation
                elif decision and prediction_type == 'Dip':
                    self.save_dip_response(decision, explanation)
                    return decision, explanation
                else:
                    self.notifier.send_notification("Predictor Error", message=f"{response}")
                    raise ValueError("No valid decision found in the response.")

            except Exception as e:
                if "Request timed out" or "Connection aborted" in str(e):
                    self.bot_manager.save_error_to_csv(str(e))
                    logging.error("Error in ChatGPT API call: Request timed out.")
                    print("Error in ChatGPT API call: Request timed out or connection aborted. Retrying...")
                else:
                    self.bot_manager.save_error_to_csv(str(e))
                    logging.error(f"Error during communication with OpenAI: {e}")
                    print(f"Attempt {attempt + 1} failed. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

        # If all retries fail
        logging.error("All attempts to communicate with OpenAI failed. Skipping this cycle.")
        return "Hold", "Failed to get a response from OpenAI after multiple attempts."

    def extract_decision_from_response(self, response):
        if response is None:
            return None

        # Look for a recommendation within double quotes (e.g., "Buy", "Sell", or "Hold")
        if "&hold&" in response.lower():
            return "Hold"
        # elif "&sell&" in response.lower():
        #     return "&Sell&"
        elif "&buy&" in response.lower():
            return "Buy"
        elif "&sell_tp&" in response.lower():
            return "Sell"
        elif "&sell_sl&" in response.lower():
            return "Sell"
        else:
            return None

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
        # elif sell_count > buy_count and sell_count > hold_count:
        #     final_decision = "Sell"
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

    data_collector = DataCollector(api_key, api_secret, intervals=['1m', '5m', '15m','30m', '1h', '1d'])
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