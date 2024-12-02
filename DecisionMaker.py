import json
import os

import numpy as np
import pandas as pd
import logging


class DecisionMaker:
    def __init__(self, risk_tolerance=None, base_stop_loss=None, base_take_profit=None, trading_interval=None, profit_interval=None,
                 loose_interval=None, dip_interval=None, amount_rsi_interval=None, amount_atr_interval=None, min_stable_intervals=None, gain_sell_threshold=None,roc_down_speed=None, data_directory='data', scalping_intervals=None):
        self.risk_tolerance = risk_tolerance
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.profit_interval = profit_interval
        self.loose_interval = loose_interval
        self.dip_interval = dip_interval
        self.amount_rsi_interval = amount_rsi_interval
        self.amount_atr_interval = amount_atr_interval
        self.min_stable_intervals = min_stable_intervals
        self.data_directory = data_directory  # Set the data directory for file storage
        self.max_gain_file = os.path.join(self.data_directory, 'max_gain.json')
        self.max_gain = self.load_max_gain()  # Load max gain from the file
        self.sell_threshold = gain_sell_threshold  # 25% loss from max gain to trigger sell
        self.roc_down_speed = roc_down_speed
        self.trading_interval= trading_interval
        self.scalping_intervals = scalping_intervals
        self.oversold_reached = False  # Track if `current_k` has ever reached zero
        self.overbought_reached = False
        self.lowest_k_reached = None

        # Configure logging to save in the data directory
        log_file_path = os.path.join(self.data_directory, 'bot_manager.log')
        logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


    def save_max_gain(self):
        """
        Save the max_gain value to a JSON file.
        """
        try:
            with open(self.max_gain_file, 'w') as file:
                json.dump({'max_gain': self.max_gain}, file)
            print(f"Max gain {self.max_gain:.2f}% saved to {self.max_gain_file}")
        except Exception as e:
            print(f"Error saving max_gain to file: {e}")

    def load_max_gain(self):
        """
        Load the max_gain value from a JSON file.
        If the file does not exist, return 0.0.
        """
        try:
            if os.path.exists(self.max_gain_file):
                with open(self.max_gain_file, 'r') as file:
                    data = json.load(file)
                    return data.get('max_gain', 0.0)
            else:
                return 0.0  # Default value if the file doesn't exist
        except Exception as e:
            print(f"Error loading max_gain from file: {e}")
            return 0.0

    def set_max_gain(self, bot_manager, current_price):
        total_portfolio_gain = self.calculate_stable_portfolio_gain(bot_manager, current_price)
        self.max_gain = total_portfolio_gain  # Update the maximum gain
        self.save_max_gain()  # Save the updated max gain to the file

    def calculate_stable_portfolio_gain(self, bot_manager, current_price):
        """
        Calculate the total portfolio gain/loss based on the current positions and the current market price,
        only for positions where the scalping_flag is not equal to position['dip'].
        This method uses the calculate_gain_loose and invested_budget methods from BotManager class.
        :param bot_manager: Instance of BotManager to access existing methods.
        :param current_price: The current market price to compare with entry prices.
        :return: Average portfolio gain/loss in percentage for positions where scalping_flag != position['dip'].
        """
        total_gain_percent = 0.0
        valid_positions_count = 0

        # Iterate over each position and calculate percentage gain/loss only if scalping_flag != position['dip']
        for position_id, position in bot_manager.position_manager.get_positions().items():
            scalping_flag = position.get('scalping', None)

            # Check if the scalping_flag condition is met (only consider stable positions)
            if scalping_flag == 0:
                entry_price = float(position['entry_price'])
                gain_loss_percent = bot_manager.calculate_gain_loose(entry_price, current_price)

                # Accumulate the total percentage gain/loss
                total_gain_percent += gain_loss_percent
                valid_positions_count += 1

        # Calculate the average percentage gain/loss for all stable positions
        if valid_positions_count > 0:
            average_portfolio_gain_percent = total_gain_percent / valid_positions_count
        else:
            average_portfolio_gain_percent = 0.0

        return average_portfolio_gain_percent

    def check_for_sell_due_to_reversal(self, bot_manager, current_price, portfolio_stop_loss_avg):
        """
        Check if the portfolio gain has reached a maximum and lost 25% of that gain, triggering a sell decision.
        :param portfolio_stop_loss_avg:
        :param bot_manager: Instance of BotManager to access existing methods.
        :param current_price: The current market price.
        :return: "Sell" if a 25% reversal is detected, otherwise "Hold".
        """
        try:
            # Calculate the total portfolio gain using BotManager methods
            total_portfolio_gain = self.calculate_stable_portfolio_gain(bot_manager, current_price)

            self.max_gain = self.load_max_gain()  # Load max gain from the file

            # Check if the current gain is a new maximum
            if total_portfolio_gain > self.max_gain:
                self.max_gain = total_portfolio_gain  # Update the maximum gain
                self.save_max_gain()  # Save the updated max gain to the file
                print(f"New maximum gain reached: {self.max_gain:.2f}%")

            # If the current gain has decreased by 25% from the maximum, issue a sell signal
            if total_portfolio_gain < self.max_gain * (1 - portfolio_stop_loss_avg):
                message = f"Market has reversed. Current gain: {total_portfolio_gain:.2f}%, Max gain: {self.max_gain:.2f}%"
                self.max_gain = 0  # Reset The Maximum Gain
                self.save_max_gain()  # Save the reset max gain to the file
                return "Sell" ,message
            else:
                message = f"Current Portfolio Gain: {total_portfolio_gain:.2f}%, Max Gain Reached{self.max_gain:.2f}% ,No reversal detected (Hold)."
                return "Hold" ,message

        except Exception as e:
            message = f"Error in checking for sell due to reversal: {e}"
            return "Hold" ,message

    def calculate_buy_amount(self, all_features, amount_rsi_interval, amount_atr_interval, capital, current_price):
        """
        Calculate buy amount based on ATR (from 30m or 1h) and StochRSI (from 5m or 15m).

        :param capital: Available capital for trading.
        :param amount_atr_interval: Interval to use for ATR calculation.
        :param amount_rsi_interval: Interval to use for StochRSI calculation.
        :param all_features: A dictionary of dataframes for different intervals (e.g., '1m', '5m', '15m', '30m', '1h', '1d')
        :param current_price: The current price of the asset.
        :return: Recommended buy amount.
        """

        # Extract data for each interval
        current_atr = all_features['latest'][amount_atr_interval].get('ATR', None)
        current_stoch_rsi = all_features['latest'][amount_rsi_interval].get('stoch_rsi_k', None)

        # Ensure ATR and StochRSI values are available
        if current_atr is None or current_stoch_rsi is None:
            raise ValueError("ATR or StochRSI data is missing.")

        # Calculate volatility factor
        volatility_factor = current_atr / current_price

        # Smooth StochRSI using the last 4 values from the historical data DataFrame
        stoch_rsi_values = all_features['history'][amount_rsi_interval]['stoch_rsi_k'].iloc[-4:]
        smoothed_stoch_rsi = stoch_rsi_values.mean()

        # Calculate momentum factor with a sigmoid-like function for smoother scaling
        momentum_factor = 100 * np.exp(-0.05 * (smoothed_stoch_rsi - 1))

        # Adjust momentum factor based on trend strength (e.g., ADX)
        adx_value = all_features['latest'][amount_rsi_interval].get('ADX', None)
        if adx_value is not None and adx_value > 25:
            momentum_factor *= 1.2  # Boost by 20% if a strong trend is detected

        # Adjust momentum factor based on recent price change to reduce risk in volatile conditions
        recent_price_change = all_features['latest'][amount_rsi_interval].get('price_change', 0.0)
        if abs(recent_price_change) > 2:
            momentum_factor *= 0.8  # Reduce by 20% if recent price change is significant

        # Adjust the buy amount based on both volatility and momentum factors
        adjusted_risk = self.risk_tolerance * volatility_factor * momentum_factor
        buy_amount = capital * adjusted_risk

        print(
            f"ATR ({amount_atr_interval}): {current_atr:.2f}, Smoothed StochRSI ({amount_rsi_interval}): {smoothed_stoch_rsi:.2f}, Buy Amount: {buy_amount:.2f}")
        return buy_amount


    def calculate_adjusted_take_profit(self, entry_price, upper_band_profit, lower_band_profit):
        """
        Calculate adjusted take profit based on the price change within the Bollinger Bands.
        :param entry_price: The entry price of the position.
        :param upper_band_profit: The upper Bollinger Band for the 15m interval.
        :param lower_band_profit: The lower Bollinger Band for the 15m interval.
        :return: Adjusted take profit.
        """
        if upper_band_profit and lower_band_profit and entry_price:
            # Calculate the bandwidth
            band_width = upper_band_profit - lower_band_profit

            if band_width == 0:
                return self.base_take_profit  # Avoid division by zero

            # Calculate the price change ratio using entry price
            price_change_ratio = ((upper_band_profit - entry_price) / band_width)

            # Calculate the adjusted take profit
            adjusted_take_profit = self.base_take_profit * (1 + price_change_ratio)
            if adjusted_take_profit < self.base_take_profit:
                return self.base_take_profit

            return adjusted_take_profit

        return self.base_take_profit

    def calculate_adjusted_stop_middle(self, entry_price, upper_band_loss, middle_band_loss):
        """
        Calculate adjusted take profit based on the price change within the Bollinger Bands.
        :param entry_price: The entry price of the position.
        :param upper_band_loss: The upper Bollinger Band for the 15m interval.
        :param middle_band_loss: The lower Bollinger Band for the 15m interval.
        :return: Adjusted take profit.
        """
        if entry_price and upper_band_loss and middle_band_loss:
            if middle_band_loss < entry_price < upper_band_loss:
                # Calculate the bandwidth
                band_width = upper_band_loss - middle_band_loss

                if band_width == 0:
                    return -self.base_stop_loss  # Avoid division by zero

                # Calculate the price change ratio using entry price
                price_change_ratio = ((entry_price - middle_band_loss) / band_width)

                # Calculate the adjusted take profit
                adjusted_stop_loss_middle = self.base_stop_loss * (1 + price_change_ratio)
                if adjusted_stop_loss_middle < self.base_stop_loss:
                    return -self.base_stop_loss

                return -adjusted_stop_loss_middle

            return -self.base_stop_loss
        return 0

    def calculate_adjusted_stop_lower(self, entry_price, lower_band_loss, middle_band_loss):
        """
        Calculate adjusted take profit based on the price change within the Bollinger Bands.
        :param entry_price: The entry price of the position.
        :param lower_band_loss: The upper Bollinger Band for the 15m interval.
        :param middle_band_loss: The lower Bollinger Band for the 15m interval.
        :return: Adjusted take profit.
        """
        if lower_band_loss and entry_price and middle_band_loss:
            if lower_band_loss < entry_price < middle_band_loss:
                # Calculate the bandwidth
                band_width = middle_band_loss - lower_band_loss

                if band_width == 0:
                    return -self.base_stop_loss  # Avoid division by zero

                # Calculate the price change ratio using entry price
                price_change_ratio = ((entry_price - lower_band_loss) / band_width)

                # Calculate the adjusted take profit
                adjusted_stop_loss_lower = self.base_stop_loss * (1 + price_change_ratio)
                if adjusted_stop_loss_lower < self.base_stop_loss:
                    return -self.base_stop_loss

                return -adjusted_stop_loss_lower

            return -self.base_stop_loss
        return 0

    def market_stable(self, all_features):
        """
        Check if the market is stable based on volatility and other criteria across multiple intervals.
        :param all_features: Dictionary containing features for multiple intervals.
        :return: True if the market is stable, False otherwise.
        """
        stable_intervals = 0
        total_intervals = len(all_features)

        for interval, latest_features in all_features['latest'].items():

            # Check ATR (Average True Range) to measure volatility
            # atr = features.get('ATR', None)
            # close_price = features.get('close', None)
            # if atr and close_price:
            #     relative_atr = atr / close_price
            #     if relative_atr <= self.volatility_threshold:
            #         stable_intervals += 1

            # Additional checks for stability could include:
            rsi = latest_features.get('RSI_14', None)
            if rsi and 30 <= rsi <= 70:
                stable_intervals += 1

            close_price = latest_features.get('close', None)
            upper_band = latest_features.get('upper_band', None)
            lower_band = latest_features.get('lower_band', None)
            if upper_band and lower_band and close_price:
                if lower_band <= close_price <= upper_band:
                    stable_intervals += 1

        # Consider the market stable if a majority of intervals indicate stability
        # if stable_intervals >= (total_intervals * 2 * 0.75):  # e.g., 4 out of 5 intervals must be stable
        if total_intervals - (total_intervals - (stable_intervals / 2)) >= self.min_stable_intervals:  # e.g., 5 out of 6 intervals must be stable
            return True, stable_intervals

        return False

    def market_downtrend_stable(self, all_features):
        """
        Check if the market is stable based on volatility and other criteria across multiple intervals.
        :param all_features: Dictionary containing features for multiple intervals.
        :return: True if the market is stable, False otherwise.
        """
        stable_count = 0
        total_intervals = len(all_features)

        for interval, latest_features in all_features['latest'].items():

            roc = latest_features.get('ROC', None)
            if roc is not None:
                if roc > self.roc_down_speed:
                    stable_count += 1

            rsi = latest_features.get('RSI_14', None)
            if rsi >= 30:
                stable_count += 1

            close_price = latest_features.get('close', None)
            lower_band = latest_features.get('lower_band', None)
            if  lower_band and close_price:
                if close_price >= lower_band:
                    stable_count += 1


        stable_intervals = total_intervals - (total_intervals - (stable_count / 3))
        if stable_intervals >= self.min_stable_intervals:  # e.g., 5 out of 6 intervals must be stable
            return True, stable_intervals

        return False, stable_intervals

    def get_resistance_info(self, all_features):

        for interval, latest_features in all_features['latest'].items():

            if interval == self.profit_interval:
                return latest_features.get('resistance_level', None), latest_features.get('average_resistance', None)

    def get_support_info(self, all_features):

        for interval, latest_features in all_features['latest'].items():

            if interval == self.loose_interval:
                return latest_features.get('support_level', None), latest_features.get('average_support', None)

    def support_level_stable(self, all_features):
        support_level, average_support = self.get_support_info(all_features)
        if support_level and average_support:
            return True
        return False

    def resistance_level_stable(self, all_features):
        resistance_level, average_resistance = self.get_resistance_info(all_features)
        if resistance_level and average_resistance:
            return True
        return False

    def get_stop_loss(self, all_features):

        for interval, latest_features in all_features['latest'].items():

            if interval == self.loose_interval:
                return latest_features.get('stop_loss', None)

    def get_stop_loss_scalping(self, all_features):

        for interval, latest_features in all_features['latest'].items():

            if interval == self.scalping_intervals[0]:
                return latest_features.get('stop_loss_scalping', None)

    def loading_stop_loss(self):
        """
        Load the stop_loss value from the stop_loss.json file in the data directory.
        """
        try:
            stop_loss_file = os.path.join(self.data_directory, 'stop_loss.json')
            if os.path.exists(stop_loss_file):
                with open(stop_loss_file, 'r') as file:
                    data = json.load(file)
                    return data.get('stop_loss', None)
            else:
                raise FileNotFoundError(f"{stop_loss_file} does not exist.")
        except Exception as e:
            print(f"Error loading stop_loss from file: {e}")
            raise

    def loading_stop_loss_scalping(self):
        """
        Load the stop_loss value from the stop_loss.json file in the data directory.
        """
        try:
            stop_loss_scalping_file = os.path.join(self.data_directory, 'stop_loss_scalping.json')
            if os.path.exists(stop_loss_scalping_file):
                with open(stop_loss_scalping_file, 'r') as file:
                    data = json.load(file)
                    return data.get('stop_loss', None)
            else:
                raise FileNotFoundError(f"{stop_loss_scalping_file} does not exist.")
        except Exception as e:
            print(f"Error loading stop_loss_scalping from file: {e}")
            raise

    def is_there_dip(self, all_features):

        interval_lower_band = all_features['latest'][self.dip_interval].get('lower_band', None)
        interval_close_price = all_features['latest'][self.dip_interval].get('close', None)

        if interval_close_price < interval_lower_band:
            return True
        return False


    def should_sell(self, current_price, entry_price, adjusted_stop_loss_lower, adjusted_stop_loss_middle,
                    adjusted_take_profit, middle_band_loss, lower_band_loss, all_features, position_expired, macd_positive):
        # Calculate the percentage change from the entry price
        price_change = ((current_price - entry_price) / entry_price) * 100

        # market_stable, stable_intervals = self.market_downtrend_stable(all_features)
        support_level_stable = self.support_level_stable(all_features)
        resistance_level_stable = self.resistance_level_stable(all_features)
        stop_loss = self.loading_stop_loss()
        # # Check if the price has hit the take-profit threshold
        # if price_change >= adjusted_take_profit:
        #     return True

        #Check is the market has unstable downtrend condition for position settlement
        if not macd_positive and current_price < stop_loss:
            if entry_price > middle_band_loss:
                if price_change < adjusted_stop_loss_middle:
                    return True
            elif entry_price > lower_band_loss:
                if price_change < adjusted_stop_loss_lower:
                    return True

        #Check if the position expired
        elif position_expired:
            if entry_price > middle_band_loss:
                if price_change < adjusted_stop_loss_middle:
                    return True
            elif entry_price > lower_band_loss:
                if price_change < adjusted_stop_loss_lower:
                    return True

        # If none of the above conditions are met, do not sell
        return False



    def make_decision(self, prediction, current_price, entry_price, all_features, position_expired, macd_positive, bot_manager):
        """
        Make a final trading decision based on the prediction and risk management rules.
        :param bot_manager:
        :param macd_positive:
        :param position_expired:
        :param prediction: The initial prediction from the Predictor (Buy, Sell, Hold).
        :param current_price: The current market price of the asset.
        :param entry_price: The price at which the current position was entered (if any).
        :param all_features: Dictionary containing features for multiple intervals.
        :return: Final decision (Buy, Sell, Hold), adjusted_stop_loss, adjusted_take_profit.
        """

        def buy_price_too_close(instance=bot_manager, buy_price=current_price):
            """
            Checks if the buy price is too close to any existing bought positions based on the ATR value.
            :param instance: Instance of the bot manager
            :param buy_price: The price of the new buy decision
            :return: True if the price is too close to an existing position, otherwise False
            """

            if all_features is None or self.trading_interval is None:
                raise ValueError("All features and trading interval must be provided for comparison.")

            # Extract the ATR value for the specified trading interval
            atr_value = all_features['latest'][self.trading_interval].get('ATR', None)
            if atr_value is None:
                raise ValueError(f"ATR value not found for trading interval: {self.trading_interval}")

            try:
                # Loop through each position and check if the new buy price is too close
                for position_id, position in instance.position_manager.get_positions().items():
                    existing_price = position.get('entry_price')
                    if existing_price:
                        # Calculate the price difference
                        price_difference = abs(buy_price - existing_price)

                        # If the price difference is less than the ATR, decline the buy
                        if price_difference <= atr_value:
                            return True

            except Exception as e:
                raise Exception(f"Error checking positions: {e}")

            # If no close positions are found, return False
            return False

        # Get the necessary data
        lower_band_profit = all_features['latest'][self.profit_interval].get('lower_band', None)
        upper_band_profit = all_features['latest'][self.profit_interval].get('upper_band', None)
        lower_band_loss = all_features['latest'][self.loose_interval].get('lower_band', None)
        middle_band_loss = all_features['latest'][self.loose_interval].get('middle_band', None)
        upper_band_loss = all_features['latest'][self.loose_interval].get('upper_band', None)
        market_stable, stable_intervals = self.market_downtrend_stable(all_features)
        is_there_dip = self.is_there_dip(all_features)

        # Adjust stop_loss based
        adjusted_stop_loss_middle = self.calculate_adjusted_stop_middle(entry_price, upper_band_loss, middle_band_loss)
        adjusted_stop_loss_lower = self.calculate_adjusted_stop_lower(entry_price, lower_band_loss, middle_band_loss)

        # adjust take_profit base
        adjusted_take_profit = self.calculate_adjusted_take_profit(entry_price, upper_band_profit, lower_band_profit)

        if prediction == "Buy" and market_stable and not buy_price_too_close():
            return "Buy", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

        elif prediction == "Buy" and not market_stable and is_there_dip:
            return "Buy_Dip", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

        elif prediction == "Suspended" and entry_price:
            if self.should_sell(current_price, entry_price, adjusted_stop_loss_lower, adjusted_stop_loss_middle,
                                adjusted_take_profit, middle_band_loss, lower_band_loss, all_features, position_expired, macd_positive):
                return "Sell", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

        elif prediction == "Sell" and entry_price:
            return "Sell", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

        return "Hold", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit


    # ------------------------------------------------------------------------------------------------------------------------------------------

    def scalping_make_decision(self, all_features, scalping_positions, entry_gain_loss=None, current_price=None):
        """
        Make a scalping decision based on technical indicators.

        :param stop_loss_scalping_value:
        :param all_features: Dictionary containing all market data.
        :param scalping_positions: Boolean indicating if there are any active scalping positions.
        :param entry_gain_loss: Current gain or loss percentage of the active scalping position.
        :return: Decision to 'Buy_Sc', 'Sell_Sc', or 'Hold'.
        """

        def ema_status():
            interval_data = all_features['latest'].get(self.scalping_intervals[0], pd.DataFrame())

            ema_7 = interval_data.get('EMA_7', None)
            ema_25 = interval_data.get('EMA_25', None)

            if ema_7 is None or ema_25 is None:
                log_message = "EMA Status: ||Error|| ,Missing EMA values (EMA_7 or EMA_25 is None)"
                print(log_message)
                logging.info(log_message)
                return 'No Signal'

            if ema_7 > ema_25:
                log_message = "EMA Status: ||ema_positive||"
                print(log_message)
                logging.info(log_message)
                return 'ema_positive'
            elif ema_7 < ema_25:
                log_message = "EMA Status: ||ema_negative||"
                print(log_message)
                logging.info(log_message)
                return 'ema_negative'

            return 'No Signal'

        def stoch_rsi_signal():
            interval_data = all_features['latest'].get(self.scalping_intervals[0], pd.DataFrame())

            if len(interval_data) > 0:
                current_k = interval_data.get('stoch_rsi_k', None)
                current_d = interval_data.get('stoch_rsi_d', None)

                # Ensure that current_k and current_d are not None
                if current_k is None or current_d is None:
                    log_message = "StochRSI Cross Signal: ||Error|| ,Missing StochRSI values (current_k or current_d is None)"
                    print(log_message)
                    logging.info(log_message)
                    return 'No Signal'

                # Get the EMA status to determine the trigger threshold
                ema_signal = ema_status()

                # Set the dynamic threshold based on the EMA status
                if ema_signal == 'ema_positive':
                    trigger_threshold = 10
                elif ema_signal == 'ema_negative':
                    trigger_threshold = 1
                else:
                    # If EMA status is not positive or negative, no action required
                    log_message = "StochRSI Signal: ||No Action|| - EMA status not clear"
                    print(log_message)
                    logging.info(log_message)
                    return 'No Signal'

                # Only trigger the mechanism when current_k <= trigger_threshold
                if current_k <= trigger_threshold or self.lowest_k_reached is not None:
                    # Initialize or update the lowest value of %K
                    if self.lowest_k_reached is None or current_k < self.lowest_k_reached:
                        self.lowest_k_reached = current_k
                        log_message = f"Updated Lowest StochRSI K Value: ||{self.lowest_k_reached}||"
                        print(log_message)
                        logging.info(log_message)

                    # Check if %K has reversed significantly from the lowest value
                    if self.lowest_k_reached is not None and current_k > self.lowest_k_reached:
                        reversal_threshold = self.lowest_k_reached * 1.1  # Set a 10% increase as a significant reversal
                        if current_k > reversal_threshold:
                            log_message = f"StochRSI Signal: ||Oversold Reversal Detected|| (Lowest K: {self.lowest_k_reached}, Current K: {current_k})"
                            print(log_message)
                            logging.info(log_message)
                            # Reset the lowest value after detecting a reversal
                            self.lowest_k_reached = None
                            return 'oversold'

                # Check if %K is at an overbought level
                elif current_k > 90:
                    log_message = f"StochRSI Signal: ||overbought||"
                    print(log_message)
                    logging.info(log_message)
                    return 'overbought'

            else:
                log_message = "StochRSI Cross Signal: ||Error|| ,Insufficient data points)"
                print(log_message)
                logging.info(log_message)

            log_message = "StochRSI Signal: ||No Signal||"
            print(log_message)
            logging.info(log_message)
            return 'No Signal'

        def rsi_signal():
            """
            Generate a signal based on RSI values (6, 14, 24).
            :return: Signal 'RSI_Down', 'RSI_Up', or 'No Signal'.
            """
            interval_data = all_features['latest'].get(self.scalping_intervals[0], pd.DataFrame())

            # Get RSI values for the specified interval
            rsi_6 = interval_data.get('RSI_6', None)
            rsi_14 = interval_data.get('RSI_14', None)
            rsi_24 = interval_data.get('RSI_24', None)

            # Check if all RSI values are available
            if rsi_6 is None or rsi_14 is None or rsi_24 is None:
                log_message = f"RSI Signal: ||Error|| ,Missing RSI values for interval: {self.scalping_intervals[0]})"
                print(log_message)
                logging.info(log_message)
                return 'No Signal'

            # Determine the signal based on RSI conditions
            if rsi_6 < 30 and (rsi_6 < rsi_14 < rsi_24):
                log_message = f"RSI Signal: ||RSI_Down|| (RSI_6: {rsi_6}, RSI_14: {rsi_14}, RSI_24: {rsi_24})"
                print(log_message)
                logging.info(log_message)
                return 'RSI_Down'
            elif rsi_6 > 70 and (rsi_6 > rsi_14 > rsi_24):
                log_message = f"RSI Signal: ||RSI_Up|| (RSI_6: {rsi_6}, RSI_14: {rsi_14}, RSI_24: {rsi_24})"
                print(log_message)
                logging.info(log_message)
                return 'RSI_Up'

            log_message = "RSI Signal: ||No Signal||"
            print(log_message)
            logging.info(log_message)
            return 'No Signal'

        def rsi_fast_signal():
            """
            Generate a signal based on RSI values (6, 14).
            :return: Signal 'RSI_Fast_Up' or 'RSI_Fast_Down'.
            """
            interval_data = all_features['latest'].get(self.scalping_intervals[0], pd.DataFrame())

            rsi_6 = interval_data.get('RSI_6', None)
            rsi_14 = interval_data.get('RSI_14', None)

            if rsi_6 is None or rsi_14 is None:
                log_message = f"RSI Fast Signal: ||Error|| ,Missing RSI values for interval: {self.scalping_intervals[0]})"
                print(log_message)
                logging.info(log_message)
                return 'No Signal'

            if rsi_6 > rsi_14:
                log_message = "RSI Fast Signal: ||RSI_Fast_Up||"
                print(log_message)
                logging.info(log_message)
                return 'RSI_Fast_Up'
            elif rsi_6 < rsi_14:
                log_message = "RSI Fast Signal: ||RSI_Fast_Down||"
                print(log_message)
                logging.info(log_message)
                return 'RSI_Fast_Down'

            return 'No Signal'

        # Get signals from the defined functions
        stoch_signal = stoch_rsi_signal()
        ema_signal = ema_status()
        rsi_signal_value = rsi_signal()
        rsi_fast_signal_value = rsi_fast_signal()

        # Decision logic based on StochRSI, EMA, and RSI signals
        if scalping_positions:

            stop_loss_scalping_value = self.loading_stop_loss_scalping()

            if ema_signal == 'ema_positive' and (stoch_signal == 'overbought' or self.overbought_reached == True):
                self.overbought_reached = True
                log_message = "Scalping Decision: ||Overbought Reached with EMA Positive|| (StochRSI: overbought)"
                print(log_message)
                logging.info(log_message)
                # Wait for RSI to give RSI_Down signal
                if rsi_signal_value != 'RSI_Up':
                    log_message = "Scalping Decision: ||Sell_Sc|| (RSI: RSI_Down after overbought)"
                    print(log_message)
                    logging.info(log_message)
                    self.overbought_reached = False  # Reset the flag after sell
                    return 'Sell_Sc'

            elif ema_signal == 'ema_negative' and (stoch_signal == 'overbought' or self.overbought_reached == True):
                self.overbought_reached = True
                log_message = "Scalping Decision: ||Overbought Reached with EMA Negative|| (StochRSI: overbought)"
                print(log_message)
                logging.info(log_message)
                # Wait for RSI Fast Signal to give RSI_Fast_Down
                if rsi_fast_signal_value != 'RSI_Fast_Up':
                    log_message = "Scalping Decision: ||Sell_Sc|| (RSI Fast: RSI_Fast_Down after overbought)"
                    print(log_message)
                    logging.info(log_message)
                    self.overbought_reached = False  # Reset the flag after sell
                    return 'Sell_Sc'

            elif current_price < stop_loss_scalping_value:
                log_message = f"Scalping Decision: ||Sell_Sc|| For Stop Loss (Entry Gain/Loss: {entry_gain_loss})"
                print(log_message)
                logging.info(log_message)
                self.overbought_reached = False  # Reset the flag after sell
                return 'Sell_Sc'

        else:

            if stoch_signal == 'oversold' and rsi_signal_value == 'RSI_Down':
                log_message = "Scalping Decision: ||Buy_Sc|| (StochRSI: oversold, RSI: RSI_Down)"
                print(log_message)
                logging.info(log_message)
                return 'Buy_Sc'

        # If no conditions are met, return 'Hold'
        log_message = "Scalping Decision: ||Hold|| (No definitive conditions met for Buy or Sell)"
        print(log_message)
        logging.info(log_message)
        return 'Hold'








