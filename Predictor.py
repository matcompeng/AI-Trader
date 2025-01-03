import os
from http.client import responses

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
        self.trading_strategy = (
            "### Knowing That I have Following Trading Strategy:\n"
            "1. **OBV Volume Confirmation Rule**:\n"
            f"   - Dose the current OBV in '{self.dip_interval}' market data interval > OBV in most recent record of '{self.dip_interval}' historical context, if yes make this rule true\n\n"

            "2. **MACD Histogram Confirmation Rule**:\n"
            "   - The MACD Histogram must either:\n"
            "       - Show an increase in the number of consecutive bars above zero (indicating positive momentum), or\n"
            "       - Show a reduction in the magnitude of consecutive bars below zero (indicating weakening negative momentum) ,In simpler terms If the histogram value becomes less negative (e.g., from -0.89 to -0.67), it indicates decreasing downward pressure and a potential shift to bullish momentum.\n"
            "   - Do not indicate reduction in consecutive bars above zero as positive momentum\n"
            f"   - This trend must be observed in both historical and current market data for intervals '{self.trading_interval}' and '{self.dip_interval}'.\n"
            f"   - Important: In order to do this calculations you must take only the most recent 2 candles of each '{self.trading_interval}' and '{self.dip_interval}' historical data context along with the 3rd candle that must taken from its relative current market intervals.\n\n"

            "3. **Breaking Significant Resistance Rule**:\n"
            f"   - A significant resistance level must defined by analyzing '{self.trading_interval}' historical context data as following:\n"
            "       1. Make a list of ranges which every range is determined by a group of candles where their 'High' and 'Close' prices has been overlapped. Then represent these ranges as 'nominated resistance ranges'\n"
            "       2. Take the provided current market price and check if it is crossed above **all** 'nominated resistance range', if yes then make this Rule True.\n"
            "   - Note: If there are no 'nominated resistance ranges' found, make this rule False\n\n"

            "4. **Uptrend Momentum Rule**:\n"
            "   1. Validate the **EMA (7)/EMA (25)** condition first:\n"
            f"       - Dose **EMA (7)** consistently above **EMA (25)** for **all** data available in **'{self.dip_interval}' historical context**.\n"
            "   2. Then validate EMA (25) raising condition:\n"
            f"       - Dose **EMA (25)** showing a pattern of rising consecutive timeframes using **'{self.dip_interval}' historical context**.\n"
            "   - If either fails, immediately flag the rule as false without proceeding further\n\n"

            "5. **StochRSI Rule**:\n"
            "   1. Validate the historical condition first:\n"
            "       - **Historical %k**: Dose ** historical %k value < 20 ** in all of 'StochRSI historical context data'.\n"
            "   2. then proceed to the current condition:\n"
            f"      - **Current %k**: Dose ** current %k value < 20 ** in '{self.trading_interval}' current market data interval.\n"
            "   - If either fails, immediately flag the rule as false without proceeding further\n\n"

            "6. **Strict RSI Condition Rule**:\n"
            f"   - *Mandatory Rule*: If RSI exceeds 67 only in current market data '{self.dip_interval}' interval, no 'Buy' decisions are allowed; the response must be 'Hold' regardless of other indicators.\n"
            "   - Important: This rule takes precedence over all other rules, even if MACD and OBV show strong bullish signals.\n\n"

            "7. **Buy After Dip Reversal**:\n"
            "   -  Follow this flowchart to confirm a reversal and to ensure accuracy in decision-making:\n"
            "       - *StochRSI Rule Check*: if StochRSI Rule (Rule 5 applies) is true, proceed to check 'OBV Confirmation'.\n"
            "       - *OBV Confirmation*: Is OBV increasing, suggesting higher buying volume (Rule 1 applies)? If yes, consider a 'Buy' signal.\n\n"

            "8. **Buy After Resistance Breakout**:\n"
            "   - Follow this flowchart to confirm a Resistance Breakout and to ensure accuracy in decision-making:\n"
            f"      - *Uptrend Momentum rule check*: If Uptrend Momentum rule (Rule 4 applies) is true, proceed to check 'Resistance Breakout'\n"
            "       - *Resistance Breakout check*: if Breaking Significant Resistance Rule (Rule 3 applies) is true, proceed to check 'MACD Histogram'.\n"
            "       - *MACD Histogram*: Is the MACD Histogram increasing (Rule 2 applies)? If yes, proceed to check 'ADX Confirmation'.\n"
            "       - *ADX Confirmation*: Is ADX above 20 on both '5m' and '15m' intervals, confirming short-term upward momentum? If yes, consider a 'Buy' signal.\n\n"

            "9. Decision Priority:\n"
            "    - **First Priority:** Prioritize **Point 2** (Buy After Dip Reversal) if the market is showing recovery from an oversold condition with upward momentum.\n"
            "    - **Second Priority:** If the conditions in **Point 2** are not present, prioritize **Point 3** (Buy After Resistance Breakout) if the market demonstrates steady momentum and a significant resistance breakout.\n\n"

            "10. **Strict Rule Adherence**:\n"
            "    - Follow each rule strictly as outlined in this strategy without inference or additional interpretation.\n"
            "    - *Reminder*: Decisions must adhere to each of the rules in sequence and respond with 'Buy' or 'Hold' strictly based on the conditions and flowchart met.\n"
            "    - Must Always mention in your explanation the specific values of indicators and price levels when analyzing or confirming trading conditions.\n"
        )


        # Ensure the data directory exists
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)


    def format_stable_prompt(self, all_features, current_price, current_obv):
        """
        Formats the prompt for the OpenAI API, including historical context and current market data.

        :param all_features: The dictionary containing processed feature data for all intervals.
        :param current_price: The current price of the cryptocurrency.
        :param current_obv: Current OBV (On Balance Volume) data.
        :return: Formatted prompt string.
        """
        # Trading strategy section before historical context
        prompt = self.trading_strategy

        # Include the historical context as one line per entry for the trading interval
        historical_data_short = all_features['history'].get(self.trading_interval, pd.DataFrame())
        if not historical_data_short.empty:
            prompt += f"\n\n### Interval({self.trading_interval}) Historical Context:\n"
            for _, row in historical_data_short.iloc[-48:].iterrows():
                historical_prompt = (
                    f"timestamp: {row['timestamp']}, "
                    f"Open: {row['open']:.5f}, "
                    f"High: {row['high']:.5f}, "
                    f"Low: {row['low']:.5f}, "
                    f"Close: {row['close']:.5f}, "
                    f"MACD Hist: {row['MACD_hist']:.5f}\n"
                )
                prompt += historical_prompt

        # Include the historical context for the dip interval
        historical_data_long = all_features['history'].get(self.dip_interval, pd.DataFrame())
        if not historical_data_long.empty:
            prompt += f"\n\n### Interval({self.dip_interval}) Historical Context:\n"
            for _, row in historical_data_long.iloc[-4:].iterrows():
                historical_prompt = (
                    f"timestamp: {row['timestamp']}, "
                    f"RSI (14): {row['RSI_14']:.5f}, "
                    f"EMA (7): {row['EMA_7']:.5f}, "
                    f"EMA (25): {row['EMA_25']:.5f}, "
                    f"MACD Hist: {row['MACD_hist']:.5f}, "
                    f"OBV: {row['OBV']:.5f}\n"
                )
                prompt += historical_prompt

        # Include the historical StochRSI context for the trading interval
        if not historical_data_short.empty:
            prompt += f"\n\n### StochRSI Historical Context:\n"
            for _, row in historical_data_short.iloc[-2:].iterrows():
                historical_prompt = (
                    f"timestamp: {row['timestamp']}, "
                    f"StochRSI %K: {row['stoch_rsi_k']:.5f}, "
                    f"StochRSI %D: {row['stoch_rsi_d']:.5f}\n"
                )
                prompt += historical_prompt

        # Start with the market data header
        prompt += "\n\n### Here is the current market data across different intervals:\n"

        # Include the current market data from different intervals
        for interval, latest_features in all_features['latest'].items():
            if latest_features:
                row = pd.Series(latest_features)
                interval_prompt = (
                    f"Interval '{interval}':\n"
                    f"RSI (14): {row['RSI_14']:.5f}, "
                    f"MACD Hist: {row['MACD_hist']:.5f}, "
                    f"StochRSI %K: {row['stoch_rsi_k']:.5f}, "
                    f"StochRSI %D: {row['stoch_rsi_d']:.5f}, "
                    f"ADX: {row['ADX']:.5f}, "
                    f"OBV: {row['OBV']:.5f}\n\n"
                )
                prompt += interval_prompt

        # Append final instructions for ChatGPT at the end
        prompt += (
            f"\n\nI am looking to trade {self.coin} cryptocurrency in the short term within a day.\n"
            f"Knowing that the current price is: {current_price} for this cycle.\n"
            "Please do the following Steps:\n"
            "STEP 1: Process all mentioned rules of points 1, 2, 3, 4, and 5 in trading strategy with taking into consideration point 10.\n"
            "STEP 2: Taking the evaluations from STEP 1, Generate the Buy or Hold Signal using points 6, 7, 8, and 9 with taking into consideration point 10. \n"
            "STEP 3: Before providing a final recommendation in your response, the system must perform a cross-check validation to ensure all conditions align with the trading strategy, this includes:\n"
            "          - Re-evaluating each rule mentioned in trading strategy again (1, 2, 3, 4, and 5) to confirm that all thresholds and conditions are correctly applied.\n"
            "          - Identifying any conflicting indicators or overlooked thresholds (e.g., ADX below 20 despite bullish conditions).\n"
            "          - Ensure no single indicator overrides the combined analysis unless specified by the strategy as a strict rule.\n"
            "          - Refer back to the strategy’s exact rule definitions when validating\n"
            "          - Avoid assuming logical shortcuts or skipping over details that require precise comparison.\n"
            "          - Always explicitly state which condition fails when flagging a rule as false.\n"
            "          - Revising the recommendation if any discrepancy is found during the cross-check phase.\n"
            "STEP 4: Must convert the 'Buy' and 'Hold' decision into format &Buy& and &Hold& for the final recommendation only in your response and it should not include this format in your explanation."
        )

        return prompt


    def format_strategy_response_prompt(self, response):

        prompt = self.trading_strategy

        prompt +=(
            "\n\n### Here's GPT API response:\n"
            f"((({response})))\n\n"
        )

        prompt +=(
            "### Here's Your Instructions:\n"
            "    - According to your roll here and submitted trading strategy, cross check then confirm or Revise the final recommendation of the provided GPT API response.\n"
            "    - Must post format &Buy& and &Hold& for the final recommendation only in your response and it should not include this format in your checking explanation."
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
                    f"{entry['price_change']:.5f}%, "
                    f"RSI (14): {entry['RSI_14']:.5f}, "
                    f"SMA (7): {entry['SMA_7']:.5f}, "
                    f"SMA (25): {entry['SMA_25']:.5f}, "
                    f"MACD: {entry['MACD']:.5f}, "
                    f"MACD Signal: {entry['MACD_signal']:.5f}, "
                    f"MACD Hist: {entry['MACD_hist']:.5f}, "
                    f"Bollinger Bands: {entry['upper_band']:.5f}, {entry['middle_band']:.5f}, {entry['lower_band']:.5f}, "
                    f"Stoch RSI %K: {entry['stoch_rsi_k']:.5f}, "
                    f"Stoch RSI %D: {entry['stoch_rsi_d']:.5f}, "
                    f"ATR: {entry['ATR']:.5f}, "
                    f"VWAP: {entry['VWAP']:.5f}, "
                    f"Support: {entry['support_level']}, "
                    f"Resistance: {entry['resistance_level']}, "
                    f"Last Price: {entry['last_price']:.5f}\n"
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

    def save_predictor_prompt(self, prompt):
        try:
            file_path = os.path.join(self.data_directory, 'latest_predictor_prompt.csv')
            df = pd.DataFrame([{"prompt": prompt}])
            df.to_csv(file_path, mode='w', header=True, index=False)
            print(f"Prompt saved to {file_path}")
        except Exception as e:
            print(f"Error saving stable prompt to CSV: {e}")

    def save_validator_prompt(self, prompt):
        try:
            file_path = os.path.join(self.data_directory, 'latest_validator_prompt.csv')
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

    def save_predictor_response(self, decision, explanation):
        try:
            file_path = os.path.join(self.data_directory, 'predictor_response.csv')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df = pd.DataFrame([{"timestamp": timestamp, "prediction": decision, "explanation": explanation}])
            df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            print(f"Stable Prediction and explanation saved to {file_path}")
        except Exception as e:
            print(f"Error saving stable response to CSV: {e}")

    def save_validator_response(self, decision, explanation):
        try:
            file_path = os.path.join(self.data_directory, 'validator_response.csv')
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df = pd.DataFrame([{"timestamp": timestamp, "validation": decision, "explanation": explanation}])
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

    def prompt_openai(self, all_features=None, current_price=None, historical_data_1=None, historical_data_2=None, prediction_type=None, positions=None, prompt_type=None ,predictor_response=None):
        prompt = None
        current_obv = None
        response = None

        if prediction_type == 'Stable' and prompt_type == 'predictor':
            prompt = self.format_stable_prompt(all_features, current_price, current_obv)
            self.save_predictor_prompt(prompt)
        elif prediction_type == 'Dip' and prompt_type == 'predictor':
            prompt = self.format_dip_prompt(positions, current_price, historical_data_2)
            self.save_dip_prompt(prompt)
        elif prediction_type == 'Stable' and prompt_type == 'validator':
            prompt = self.format_strategy_response_prompt(predictor_response)
            self.save_validator_prompt(prompt)


        for attempt in range(self.max_retries):
            try:
                if prompt_type == 'predictor':
                    response = self.chatgpt_client.get_prediction(prompt)
                elif prompt_type == 'validator':
                    response = self.chatgpt_client.get_validation(prompt)

                # Extract the decision from the response
                decision = self.extract_decision_from_response(response)
                explanation = response  # Keep the entire response as the explanation

                if decision and prediction_type == 'Stable' and prompt_type == 'predictor':
                    self.save_predictor_response(decision, explanation)
                    return decision, explanation
                elif decision and prediction_type == 'Dip' and prompt_type == 'predictor':
                    self.save_dip_response(decision, explanation)
                    return decision, explanation
                elif decision and prediction_type == 'Stable' and prompt_type == 'validator':
                    self.save_validator_response(decision, explanation)
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
            decision, explanation = predictor.prompt_openai(all_features)

            # Print the results
            print(f"Prediction: {decision}")
            print(f"Explanation: {explanation}")
        else:
            print("Failed to process features.")
    else:
        print("Failed to collect market data.")