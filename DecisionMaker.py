import json
import os

import numpy as np
import pandas as pd
import logging
from Notifier import Notifier


class DecisionMaker:
    def __init__(self, risk_tolerance=None, base_stop_loss=None, base_take_profit=None, trading_interval=None, profit_interval=None,
                 loose_interval=None, dip_interval=None, amount_rsi_interval=None, amount_atr_interval=None, min_stable_intervals=None, gain_sell_threshold=None, roc_speed=None, data_directory='data', scalping_intervals=None):
        self.risk_tolerance = risk_tolerance
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.profit_interval = profit_interval
        self.loose_interval = loose_interval
        self.dip_interval = dip_interval
        self.notifier = Notifier()
        self.amount_rsi_interval = amount_rsi_interval
        self.amount_atr_interval = amount_atr_interval
        self.min_stable_intervals = min_stable_intervals
        self.data_directory = data_directory  # Set the data directory for file storage
        self.max_gain_file = os.path.join(self.data_directory, 'max_gain.json')
        self.max_gain = self.load_max_gain()  # Load max gain from the file
        self.sell_threshold = gain_sell_threshold  # 25% loss from max gain to trigger sell
        self.roc_speed = roc_speed
        self.trading_interval= trading_interval
        self.scalping_intervals = scalping_intervals
        self.candidate_resistance_cache = {}

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

    def calculate_buy_amount(self, all_features, interval, amount_atr_interval, capital, current_price):
        """
        Calculate buy amount based on ATR (from 30m or 1h) and RSI_12 derived from the timeframe of the latest downtrend in MACD_hist.

        :param interval: The interval to use for RSI_12 and MACD_hist calculations.
        :param capital: Available capital for trading.
        :param amount_atr_interval: Interval to use for ATR calculation.
        :param all_features: A dictionary of dataframes for different intervals (e.g., '1m', '5m', '15m', '30m', '1h', '1d').
        :param current_price: The current price of the asset.
        :return: Recommended buy amount.
        """

        # Extract ATR data
        current_atr = all_features['latest'][amount_atr_interval].get('ATR', None)

        # Ensure ATR value is available
        if current_atr is None:
            raise ValueError("ATR data is missing.")

        # Extract historical data for the interval
        interval_data = all_features['history'].get(interval, None)
        if interval_data is None or interval_data.empty:
            raise ValueError(f"Historical data for interval {interval} is missing.")

        # Extract MACD_hist and RSI_12 columns
        macd_hist = interval_data['MACD_hist']
        rsi_12 = interval_data['RSI_12']

        # Ensure MACD_hist has at least one negative value
        if not any(macd_hist < 0):
            raise ValueError("No negative MACD_hist values found for the interval.")

        # Identify the latest downtrend in MACD_hist
        latest_downtrend_indices = macd_hist[macd_hist < 0].index

        # Group consecutive indices to find the latest downtrend
        latest_downtrend = [latest_downtrend_indices[0]]
        for idx in range(1, len(latest_downtrend_indices)):
            if latest_downtrend_indices[idx] - latest_downtrend_indices[idx - 1] == 1:
                latest_downtrend.append(latest_downtrend_indices[idx])
            else:
                # Stop at the first gap, as we only care about the most recent downtrend
                break

        # Find the minimum MACD_hist value in the latest downtrend
        min_macd_hist_index = macd_hist[latest_downtrend].idxmin()
        corresponding_rsi_12 = rsi_12[min_macd_hist_index]

        # Ensure RSI_12 value is available
        if pd.isna(corresponding_rsi_12):
            raise ValueError(f"No RSI_12 value available for the minimum MACD_hist at index {min_macd_hist_index}.")

        # Calculate momentum factor using a sigmoid-like function
        if corresponding_rsi_12 <= 20:
            momentum_factor = 100
        elif corresponding_rsi_12 >= 80:
            momentum_factor = 0
        else:
            momentum_factor = 100 * (1 - np.exp(-0.05 * (corresponding_rsi_12 - 20)))

        # Adjust momentum factor based on trend strength (e.g., ADX)
        adx_value = all_features['latest'][interval].get('ADX', None)
        if adx_value is not None and adx_value > 25:
            momentum_factor *= 1.2  # Boost by 20% if a strong trend is detected

        # Adjust momentum factor based on recent price change to reduce risk in volatile conditions
        recent_price_change = all_features['latest'][interval].get('price_change', 0.0)
        if abs(recent_price_change) > 2:
            momentum_factor *= 0.8  # Reduce by 20% if recent price change is significant

        # Calculate volatility factor
        volatility_factor = current_atr / current_price

        # Adjust the buy amount based on both volatility and momentum factors
        adjusted_risk = self.risk_tolerance * volatility_factor * momentum_factor
        buy_amount = capital * adjusted_risk

        # Logging and debugging information
        print(
            f"ATR ({amount_atr_interval}): {current_atr:.2f}, RSI_12 ({interval}): {corresponding_rsi_12:.2f}, "
            f"Momentum Factor: {momentum_factor:.2f}, Buy Amount: {buy_amount:.2f}"
        )

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

    def calculate_adjusted_stop_middle(self, entry_price, upper_band_loss, middle_band_loss, trading_type=None):
        """
        Calculate adjusted stop loss based on the price change within the Bollinger Bands.
        :param entry_price: The entry price of the position.
        :param upper_band_loss: The upper Bollinger Band for the interval.
        :param middle_band_loss: The middle Bollinger Band for the interval.
        :param trading_type: The type of trading ('scalping' or other).
        :return: Adjusted stop loss or price change ratio based on trading type.
        """
        if entry_price and upper_band_loss and middle_band_loss:
            if middle_band_loss < entry_price < upper_band_loss:
                # Calculate the bandwidth
                band_width = upper_band_loss - middle_band_loss

                if band_width == 0:
                    return 0  # Avoid division by zero

                # Calculate the price change ratio using entry price
                price_change_ratio = ((entry_price - middle_band_loss) / band_width)

                # Return the price change ratio for scalping
                if trading_type == 'scalping':
                    return -price_change_ratio

                # Calculate the adjusted stop loss for other trading types
                adjusted_stop_loss_middle = self.base_stop_loss * (1 + price_change_ratio)
                if adjusted_stop_loss_middle < self.base_stop_loss:
                    return -self.base_stop_loss

                return -adjusted_stop_loss_middle

            return -self.base_stop_loss if trading_type != 'scalping' else 0
        return 0

    def calculate_adjusted_stop_lower(self, entry_price, lower_band_loss, middle_band_loss, trading_type=None):
        """
        Calculate adjusted stop loss based on the price change within the Bollinger Bands.
        :param entry_price: The entry price of the position.
        :param lower_band_loss: The lower Bollinger Band for the interval.
        :param middle_band_loss: The middle Bollinger Band for the interval.
        :param trading_type: The type of trading ('scalping' or other).
        :return: Adjusted stop loss or price change ratio based on trading type.
        """
        if lower_band_loss and entry_price and middle_band_loss:
            if lower_band_loss < entry_price < middle_band_loss:
                # Calculate the bandwidth
                band_width = middle_band_loss - lower_band_loss

                if band_width == 0:
                    return 0  # Avoid division by zero

                # Calculate the price change ratio using entry price
                price_change_ratio = ((entry_price - lower_band_loss) / band_width)

                # Return the price change ratio for scalping
                if trading_type == 'scalping':
                    return -price_change_ratio

                # Calculate the adjusted stop loss for other trading types
                adjusted_stop_loss_lower = self.base_stop_loss * (1 + price_change_ratio)
                if adjusted_stop_loss_lower < self.base_stop_loss:
                    return -self.base_stop_loss

                return -adjusted_stop_loss_lower

            return -self.base_stop_loss if trading_type != 'scalping' else 0
        return 0

    def market_stable(self, all_features):
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
                if self.roc_speed[0] < roc < self.roc_speed[1]:
                    stable_count += 1

            # Additional checks for stability could include:
            rsi = latest_features.get('RSI_12', None)
            if rsi and 30 <= rsi <= 70:
                stable_count += 1

            close_price = latest_features.get('close', None)
            upper_band = latest_features.get('upper_band', None)
            lower_band = latest_features.get('lower_band', None)
            if upper_band and lower_band and close_price:
                if lower_band <= close_price <= upper_band:
                    stable_count += 1

        stable_intervals = total_intervals - (total_intervals - (stable_count / 3))
        if stable_intervals >= self.min_stable_intervals:  # e.g., 5 out of 6 intervals must be stable
            return True, stable_intervals

        return False, stable_intervals

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
                if roc > self.roc_speed[0]:
                    stable_count += 1

            rsi = latest_features.get('RSI_12', None)
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

    def detect_latest_resistance(
        self,
        all_features,
        interval,
        current_price,
        lookback=30,
        pivot_size=2,
        cluster_tolerance=0.002,
        min_touches=2
    ):
        """
        Detects the most recent 'strong' resistance level from the given OHLCV DataFrame
        by looking at pivot highs within the last `lookback` candles, requiring `min_touches` for
        confirmation. Also handles single-candle new highs by keeping a 'candidate' resistance
        if the price breaks an old level but doesn't yet have 2 touches.

        :param all_features: Dictionary containing 'history' DataFrames for different intervals.
        :param interval: Which interval to use (e.g., '1m', '5m', '15m').
        :param current_price: The current live price of the asset.
        :param lookback: Number of recent candles to look at (e.g., 30).
        :param pivot_size: How many candles on each side to confirm a local pivot high.
        :param cluster_tolerance: Fraction (e.g., 0.002 = 0.2%) to cluster nearby pivot highs.
        :param min_touches: Minimum pivot-high count in a cluster to call it “confirmed resistance.”
        :return: Float price of the latest confirmed resistance OR a candidate resistance.
        """
        # -------------------------------------------------------------------------------------------
        # 1) Retrieve the DataFrame and slice the last `lookback` candles
        # -------------------------------------------------------------------------------------------
        df = all_features['history'].get(interval, pd.DataFrame())
        if df.empty:
            return None  # Not enough data

        recent_df = df.iloc[-lookback:].copy()
        highs = recent_df['high'].values
        if len(highs) < pivot_size * 2:
            return None  # Not enough candles to identify pivots

        # -------------------------------------------------------------------------------------------
        # 2) Identify pivot highs (local maxima with pivot_size candles on each side)
        # -------------------------------------------------------------------------------------------
        pivot_indices = []
        for i in range(pivot_size, len(highs) - pivot_size):
            current_high = highs[i]
            left_side = highs[i - pivot_size : i]
            right_side = highs[i + 1 : i + 1 + pivot_size]

            if all(current_high > h for h in left_side) and all(current_high >= h for h in right_side):
                pivot_indices.append(i)

        # If no pivots, return whatever candidate might exist
        if not pivot_indices:
            # Check if we have a candidate from before
            candidate = self.candidate_resistance_cache.get(interval)
            return candidate if candidate else None

        # Gather pivot-high prices as (DataFrame index, high_value)
        pivot_prices = [(recent_df.iloc[idx].name, highs[idx]) for idx in pivot_indices]

        # -------------------------------------------------------------------------------------------
        # 3) Cluster pivot highs (because price rarely rejects at exactly the same number repeatedly)
        # -------------------------------------------------------------------------------------------
        pivot_prices.sort(key=lambda x: x[1])  # sort by pivot-high ascending
        clusters = []
        current_cluster = [pivot_prices[0]]

        for i in range(1, len(pivot_prices)):
            _, curr_price_val = pivot_prices[i]
            _, prev_price_val = pivot_prices[i - 1]

            # If within cluster_tolerance * prev_price_val, they belong to the same cluster
            if abs(curr_price_val - prev_price_val) <= cluster_tolerance * prev_price_val:
                current_cluster.append(pivot_prices[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [pivot_prices[i]]
        # Add the last cluster
        if current_cluster:
            clusters.append(current_cluster)

        # -------------------------------------------------------------------------------------------
        # 4) Find clusters with at least `min_touches` pivot highs and pick the highest one
        # -------------------------------------------------------------------------------------------
        qualified_clusters = []
        for cluster in clusters:
            if len(cluster) >= min_touches:
                # You can choose to define the cluster price as the max or average
                cluster_highs = [c[1] for c in cluster]
                cluster_max_price = max(cluster_highs)
                qualified_clusters.append(cluster_max_price)

        if qualified_clusters:
            confirmed_resistance = max(qualified_clusters)
        else:
            # If no cluster meets the min_touches threshold, we have no "confirmed" resistance
            confirmed_resistance = None

        # -------------------------------------------------------------------------------------------
        # 5) Handle "Candidate" Resistance if there's a breakout above confirmed_resistance
        # -------------------------------------------------------------------------------------------
        # We keep a separate dictionary entry for a candidate new high if
        # the price has broken above the old confirmed resistance but we don’t yet have 2 touches.

        old_candidate = self.candidate_resistance_cache.get(interval, None)

        # (A) If we do have a confirmed resistance, check if current_price is above it
        if confirmed_resistance and current_price > confirmed_resistance:
            # This means we've broken the old confirmed level, so let's see if
            # the market formed a new higher high that doesn't yet have 2 touches.
            highest_since_break = recent_df[recent_df['high'] > confirmed_resistance]['high'].max()
            if pd.isna(highest_since_break):
                highest_since_break = current_price  # fallback if nothing found

            # We store it as a candidate
            self.candidate_resistance_cache[interval] = highest_since_break

            # For *output*, if we want to reflect that the "latest resistance" is the old confirmed level,
            # we might still return the old confirmed. Or if you prefer, return the candidate.
            # That choice depends on your strategy. We'll return the candidate here if higher than the old confirmed:
            if highest_since_break > confirmed_resistance:
                return highest_since_break
            else:
                return confirmed_resistance

        # (B) If we have no confirmed_resistance yet, see if there's a candidate in place
        elif not confirmed_resistance:
            # If we do have an old candidate, check if it’s been "re-touched" at least once
            # among the newly detected pivot highs
            if old_candidate:
                # check how many pivot highs are within cluster_tolerance of old_candidate
                candidate_touches = [
                    price_val for (idx, price_val) in pivot_prices
                    if abs(price_val - old_candidate) <= cluster_tolerance * old_candidate
                ]
                if len(candidate_touches) >= min_touches:
                    # This candidate is now confirmed
                    confirmed_resistance = max(candidate_touches)
                    self.candidate_resistance_cache[interval] = None
                    return confirmed_resistance
                else:
                    # Candidate remains unconfirmed
                    return old_candidate
            else:
                # No confirmed, no candidate => nothing
                return None

        # (C) If we do have a confirmed_resistance but price is still below it
        else:
            # Possibly we want to check if the old candidate has now been retested enough times
            # to turn it into a confirmed. This scenario might happen if the price spiked,
            # then came back below the old confirmed level, so let's see if we re-touched
            # the candidate from below.
            if old_candidate and old_candidate > confirmed_resistance:
                # See how many pivot highs are near the candidate
                candidate_touches = [
                    price_val for (idx, price_val) in pivot_prices
                    if abs(price_val - old_candidate) <= cluster_tolerance * old_candidate
                ]
                if len(candidate_touches) >= min_touches:
                    # Confirm the candidate and discard it
                    confirmed_resistance = max(candidate_touches)
                    self.candidate_resistance_cache[interval] = None
                    return confirmed_resistance
                else:
                    # Keep the old candidate if it's still relevant
                    # (maybe price is dancing around in that region)
                    return confirmed_resistance
            else:
                # No candidate or candidate <= confirmed => Just return the confirmed
                return confirmed_resistance



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

    def scalping_make_decision(self, all_features, scalping_positions, entry_gain_loss=None, current_price=None, scalping_interval=None, market_stable=None, entry_price=None):
        """
        Make a scalping decision based on technical indicators.

        :param entry_price:
        :param market_stable:
        :param scalping_interval:
        :param current_price:
        :param all_features: Dictionary containing all market data.
        :param scalping_positions: Boolean indicating if there are any active scalping positions.
        :param entry_gain_loss: Current gain or loss percentage of the active scalping position.
        :return: Decision to 'Buy_Sc', 'Sell_Sc', or 'Hold'.
        """

        # Retrieve band levels from all_features
        lower_band = all_features['latest'][scalping_interval].get('lower_band', None)
        middle_band = all_features['latest'][scalping_interval].get('middle_band', None)
        upper_band = all_features['latest'][scalping_interval].get('upper_band', None)
        fake_upper_band = upper_band - (upper_band * 0.05 / 100)

        def save_flag_to_file(interval, flag_name, value):
            """
            Save a flag value to a file specific to the interval and flag.

            :param interval: The interval the flag belongs to (e.g., '1m', '5m').
            :param flag_name: The name of the flag (e.g., 'oversold_reached').
            :param value: The value of the flag to save.
            """
            try:
                # Construct the file path
                flag_file = os.path.join(self.data_directory, f"{interval}_{flag_name}.json")

                # Save the flag value to the file
                with open(flag_file, 'w') as file:
                    json.dump({'value': value}, file, indent=4)

                # print(f"Flag '{flag_name}' for interval '{interval}' saved with value: {value}")
            except Exception as e:
                print(f"Error saving flag '{flag_name}' for interval '{interval}': {e}")

        def load_flag_from_file(interval, flag_name):
            """
            Load a flag value from a file specific to the interval and flag.

            :param interval: The interval the flag belongs to (e.g., '1m', '5m').
            :param flag_name: The name of the flag (e.g., 'oversold_reached').
            :return: The value of the flag if the file exists, otherwise None.
            """
            try:
                # Construct the file path
                flag_file = os.path.join(self.data_directory, f"{interval}_{flag_name}.json")

                # Check if the file exists
                if os.path.exists(flag_file):
                    with open(flag_file, 'r') as file:
                        data = json.load(file)
                        return data.get('value', None)
                return None
            except Exception as e:
                print(f"Error loading flag '{flag_name}' for interval '{interval}': {e}")
                return None

        def reset_flag(interval, flag_name):
            save_flag_to_file(interval, flag_name, None)

        # Load the flags from the file system for the specific interval
        oversold_reached = load_flag_from_file(scalping_interval, 'oversold_reached') or False
        overbought_reached = load_flag_from_file(scalping_interval, 'overbought_reached') or False
        lowest_k_reached = load_flag_from_file(scalping_interval, 'lowest_k_reached')
        max_gain_reached = load_flag_from_file(scalping_interval, 'max_gain_reached')
        uptrend_counter = load_flag_from_file(scalping_interval, 'uptrend_counter') or 0
        macd_counter = load_flag_from_file(scalping_interval, 'macd_counter') or 0
        break_counter = load_flag_from_file(scalping_interval, 'break_counter') or 0

        def uptrend_momentum(interval, price):
            """
            Validate uptrend momentum for a given historical context.

            **Uptrend Momentum Rule**:
               1. Validate if the angle of the MACD Histogram (MACD_hist) is positively above a threshold of 10 degrees.
               2. During this check:
                  - Ensure that the current MACD Histogram value is greater than the previous value.
                  - Additionally, ensure that the previous MACD Histogram value is 95% of the current value or less.
               3. Calculate the angle of EMA_100 using percentage changes. If the angle is negative (< 0), return False regardless of other calculations.
            :return: Boolean indicating if the uptrend momentum is valid.
            """
            try:
                angle_threshold = None
                macd_diff_threshold = None

                # Extract data for the specified interval from the historical context
                interval_data = all_features['history'].get(interval, None)

                if interval == '5m':
                    angle_threshold = 20
                    macd_diff_threshold = 0.98
                elif interval == '15m':
                    angle_threshold = 20
                    macd_diff_threshold = 0.90

                if interval_data is None or interval_data.empty:
                    log_message = f"Uptrend Momentum: ||Error|| - (Historical data for interval {interval} is not available or empty)"
                    print(log_message)
                    logging.info(log_message)
                    return False

                # Use the last 20 records for EMA_100 calculation
                ema_200 = interval_data['EMA_200'].tail(10).values

                if len(ema_200) < 2:
                    log_message = "Uptrend Momentum: ||Error|| - (Not enough data for EMA_100 angle calculation)"
                    print(log_message)
                    logging.info(log_message)
                    return False

                # Calculate percentage change in EMA_100
                ema_percentage_change = (ema_200 - ema_200[0]) / ema_200[0] * 100

                # Compute the slope using percentage changes
                x_ema = np.arange(len(ema_percentage_change))
                ema_slope, _ = np.polyfit(x_ema, ema_percentage_change, 1)

                # Convert slope to angle in degrees
                ema_angle = np.degrees(np.arctan(ema_slope))

                # If EMA_100 angle is negative, return False
                if ema_angle < 3:
                    log_message = f"Uptrend Momentum: ||Failed|| - (EMA_200 angle |{ema_angle:.2f}°| is negative)"
                    print(log_message)
                    logging.info(log_message)
                    return False

                # Use only the last 4 records for MACD Histogram calculation
                macd_hist = interval_data['MACD_hist_fast'].tail(4).values

                if len(macd_hist) < 4:
                    log_message = f"Uptrend Momentum: ||Error|| - (Not enough data for MACD Histogram angle calculation)"
                    print(log_message)
                    logging.info(log_message)
                    return False

                # Calculate the angle of the MACD Histogram
                normalized_macd_hist = macd_hist / np.max(np.abs(macd_hist))  # Normalize to [0, 1]
                x_macd_hist = np.arange(len(normalized_macd_hist))
                macd_hist_slope, _ = np.polyfit(x_macd_hist, normalized_macd_hist, 1)

                # Convert slope to angle in degrees
                macd_hist_angle = np.degrees(np.arctan(macd_hist_slope))

                # Check if the MACD Histogram angle exceeds the specified threshold
                angle_flag = macd_hist_angle >= angle_threshold

                # Validate MACD and its signal
                macd = interval_data['MACD_fast'].values
                macd_signal = interval_data['MACD_signal_fast'].values

                macd_signal_flag = macd[-1] > macd_signal[-1]

                if angle_flag and macd_signal_flag:
                    log_message = f"Uptrend Momentum: ||Conditions Met|| - (MACD Histogram angle |{macd_hist_angle:.2f}°| >= threshold |{angle_threshold}°|, EMA_100 angle |{ema_angle:.2f}°|, MACD/Signal |{macd_signal_flag}|)"
                    print(log_message)
                    logging.info(log_message)
                    return True

                else:
                    log_message = f"Uptrend Momentum: ||Conditions Failed|| - (MACD Histogram angle |{macd_hist_angle:.2f}°| >= threshold |{angle_threshold}°|, EMA_100 angle |{ema_angle:.2f}°|, MACD/Signal |{macd_signal_flag}|)"
                    print(log_message)
                    logging.info(log_message)
                    return False

            except Exception as e:
                log_message = f"Uptrend Momentum: ||Error|| - (An error occurred while validating uptrend momentum: {e})"
                print(log_message)
                logging.info(log_message)
                return False

        def stoch_rsi_signal(lowest_k_reached):
            interval_data = all_features['latest'].get(scalping_interval, pd.DataFrame())

            if len(interval_data) > 0:
                current_k = interval_data.get('stoch_rsi_k', None)
                current_d = interval_data.get('stoch_rsi_d', None)

                # Ensure that current_k and current_d are not None
                if current_k is None or current_d is None:
                    log_message = "StochRSI Cross Signal: ||Error|| - (Missing StochRSI values (current_k or current_d is None)"
                    print(log_message)
                    logging.info(log_message)
                    return 'No Signal'


                # Set the dynamic threshold based on the EMA status
                if  scalping_interval == self.scalping_intervals[0]:
                    trigger_threshold = 10
                elif scalping_interval == self.scalping_intervals[1]:
                    trigger_threshold = 20
                else:
                    # If EMA status is not positive or negative, no action required
                    log_message = "StochRSI Signal: ||No Action|| - (Trigger Threshold Unreached)"
                    print(log_message)
                    logging.info(log_message)
                    return 'No Signal'

                # Only trigger the mechanism when current_k <= trigger_threshold
                if current_k <= trigger_threshold or lowest_k_reached is not None:
                    if lowest_k_reached is None or current_k < lowest_k_reached:
                        lowest_k_reached = current_k
                        save_flag_to_file(scalping_interval, 'lowest_k_reached', lowest_k_reached)
                        log_message = f"Updated Lowest StochRSI K Value: ||{lowest_k_reached}||"
                        print(log_message)
                        logging.info(log_message)

                # Check if %K has reversed significantly from the lowest value
                if lowest_k_reached is not None and current_k > lowest_k_reached:
                    reversal_threshold = lowest_k_reached * 2
                    if current_k > reversal_threshold:
                        reset_flag(scalping_interval, 'lowest_k_reached')
                        log_message = f"StochRSI Signal: ||Oversold Reversal Detected|| - (Lowest K: {lowest_k_reached}, Current K: {current_k})"
                        print(log_message)
                        logging.info(log_message)
                        return 'oversold'

                # Check if %K is at an overbought level
                elif current_k > 80:
                    log_message = f"StochRSI Signal: ||overbought||"
                    print(log_message)
                    logging.info(log_message)
                    return 'overbought'

            else:
                log_message = "StochRSI Cross Signal: ||Error|| - (Insufficient data points)"
                print(log_message)
                logging.info(log_message)

            log_message = "StochRSI Signal: ||No Signal||"
            print(log_message)
            logging.info(log_message)
            return 'No Signal'

        def rsi_signal(interval):
            """
            Generate a signal based on RSI values (6, 14, 24).
            :return: Signal 'RSI_Down', 'RSI_Up', or 'No Signal'.
            """
            data = all_features['latest'].get(interval, pd.DataFrame())

            # Get RSI values for the specified interval
            rsi_12 = data.get('RSI_12', None)
            rsi_24 = data.get('RSI_24', None)
            rsi_48 = data.get('RSI_48', None)

            # Check if all RSI values are available
            if rsi_12 is None or rsi_24 is None or rsi_48 is None:
                log_message = f"RSI Signal: ||Error|| - (Missing RSI values for interval: {interval})"
                print(log_message)
                logging.info(log_message)
                return 'No Signal'

            # Determine the signal based on RSI conditions with dynamic thresholds
            if rsi_12 < rsi_24 and rsi_12 < rsi_48:
                log_message = f"RSI Signal: ||RSI_Down|| - (RSI_12: {round(rsi_12)}, RSI_24: {round(rsi_24)}, RSI_48: {round(rsi_48)})"
                print(log_message)
                logging.info(log_message)
                return 'RSI_Down'
            elif rsi_12 > rsi_24 > rsi_48:
                log_message = f"RSI Signal: ||RSI_Up|| - (RSI_12: {round(rsi_12)}, RSI_24: {round(rsi_24)}, RSI_48: {round(rsi_48)}"
                print(log_message)
                logging.info(log_message)
                return 'RSI_Up'

            log_message = "RSI Signal: ||No Signal||"
            print(log_message)
            logging.info(log_message)
            return 'No Signal'

        def gain_trailing_lock(max_gain_reached):
            """
            Lock gain and provide a trailing sell signal based on entry_gain_loss,
            using adjusted stop loss thresholds after locking profit upon upper band crossing.
            """
            if entry_gain_loss is None or current_price is None or entry_price is None:
                return 'No Signal'

            # Ensure required data is available
            if middle_band is None or upper_band is None or lower_band is None:
                log_message = "Gain Trailing Lock: ||Error|| - (Missing band data for trailing lock calculation)"
                print(log_message)
                logging.info(log_message)
                return 'No Signal'

            # Lock the maximum gain if the price crosses the upper band
            if current_price > fake_upper_band:
                if max_gain_reached is None or current_price > max_gain_reached:
                    max_gain_reached = entry_gain_loss
                    save_flag_to_file(scalping_interval, 'max_gain_reached', max_gain_reached)
                    log_message = f"Gain Trailing Lock: ||Max Gain Locked|| - (Current Price: {current_price}, Max Gain: {max_gain_reached:.2f}%)"
                    print(log_message)
                    logging.info(log_message)

            # Calculate adjusted stop loss thresholds
            adjusted_stop_loss_middle = self.calculate_adjusted_stop_middle(
                entry_price=entry_price,
                upper_band_loss=upper_band,
                middle_band_loss=middle_band,
                trading_type='scalping'
            )

            adjusted_stop_loss_lower = self.calculate_adjusted_stop_lower(
                entry_price=entry_price,
                lower_band_loss=lower_band,
                middle_band_loss=middle_band,
                trading_type='scalping'
            )

            # Determine which threshold to use based on entry_price position
            if lower_band <= entry_price < middle_band:
                active_threshold = adjusted_stop_loss_lower
                threshold_type = "Lower Band"
            elif middle_band <= entry_price < upper_band:
                active_threshold = adjusted_stop_loss_middle
                threshold_type = "Middle Band"
            else:
                active_threshold = 0
                threshold_type = "Out of Range"

            # Log the determined threshold
            log_message = f"Gain Trailing Lock: ||Threshold Determined|| - (Entry Price: {entry_price}, Threshold Type: {threshold_type}, Active Threshold: {active_threshold:.2f})"
            print(log_message)
            logging.info(log_message)

            # Trigger a trailing sell signal if the price reverses below the active threshold
            if max_gain_reached is not None and active_threshold is not None:
                # Calculate the range between fake_upper_band and max_gain_reached
                price_range = abs(max_gain_reached - fake_upper_band)
                reversal_percentage = (current_price - max_gain_reached) / price_range * 100
                log_message = f"Gain Trailing Lock: ||Reversal Checking|| - (Reverse-%: {reversal_percentage:.2f}, Active Threshold: {active_threshold:.2f})"
                print(log_message)
                logging.info(log_message)
                if reversal_percentage <= active_threshold:
                    reset_flag(scalping_interval, 'max_gain_reached')
                    log_message = f"Gain Trailing Lock: ||Trailing Sell Signal Activated|| - (Reversal to Active Threshold: {active_threshold:.2f}%)"
                    print(log_message)
                    logging.info(log_message)
                    return 'trailing_sell'

            # If no conditions are met, return 'No Signal'
            log_message = f"Gain Trailing Lock: ||Trailing Sell Testing|| - (Entry Gain: {entry_gain_loss:.2f}%, Active Threshold: {active_threshold:.2f}%)"
            print(log_message)
            logging.info(log_message)
            return 'No Signal'

        def macd_positive_angle(interval):
            """
            Calculate the angle of the MACD line and check if it's positive.

            **MACD Angle Rule**:
               1. Use the last 3 records of MACD data.
               2. Calculate the angle of the MACD line based on its trend.
               3. If the angle is positive, return True; otherwise, return False.

            :param interval: The trading interval to fetch historical MACD data for.
            :return: Boolean indicating whether the MACD angle is positive.
            """
            try:
                # Extract data for the specified interval from the historical context
                interval_data = all_features['history'].get(interval, None)

                if interval_data is None or interval_data.empty:
                    log_message = f"MACD Angle: ||Error|| - (Historical data for interval {interval} is not available or empty)"
                    print(log_message)
                    logging.info(log_message)
                    return False

                # Use only the last 3 records from the historical context
                interval_data = interval_data.tail(4)

                # Extract MACD values
                macd = interval_data['MACD'].values

                if len(macd) < 2:
                    log_message = f"MACD Angle: ||Error|| - (Not enough data to calculate MACD angle)"
                    print(log_message)
                    logging.info(log_message)
                    return False

                # Normalize MACD values
                normalized_macd = macd / np.max(np.abs(macd))  # Normalize to [0, 1]
                x = np.arange(len(normalized_macd))
                slope, _ = np.polyfit(x, normalized_macd, 1)

                # Convert slope to angle in degrees
                macd_angle = np.degrees(np.arctan(slope))

                # Check if the MACD angle is positive
                if macd_angle >= 0:
                    log_message = f"MACD Angle: ||Positive Angle|| - (MACD angle |{macd_angle:.2f}°| >= 0)"
                    print(log_message)
                    logging.info(log_message)
                    return True
                else:
                    log_message = f"MACD Angle: ||Negative Angle|| - (MACD angle |{macd_angle:.2f}°| < 0)"
                    print(log_message)
                    logging.info(log_message)
                    return False

            except Exception as e:
                log_message = f"MACD Angle: ||Error|| - (An error occurred while calculating MACD angle: {e})"
                print(log_message)
                logging.info(log_message)
                return False

        def testing_uptrend_momentum(uptrend_counter):
            if uptrend_momentum(scalping_interval, current_price):
                uptrend_counter += 1
                save_flag_to_file(scalping_interval, 'uptrend_counter', uptrend_counter)
            else:
                uptrend_counter = 0
                save_flag_to_file(scalping_interval, 'uptrend_counter', uptrend_counter)

            # Proceed with buying decision only if the counter reaches 30
            log_message = f"Testing Uptrend Momentum: ||Uptrend Count: {uptrend_counter} of 60||"
            print(log_message)
            logging.info(log_message)

            if uptrend_counter >= 60:
                return True
            else:
                return False

        def macd_negative_rising(interval):
            """
            Check if:
              1) MACD histogram is negative but rising between the last two recorded points:
                 - macd_hist[-3] < 0 and macd_hist[-2] < 0
                 - macd_hist[-2] > macd_hist[-3]
              2) RSI_12 < RSI_24 < RSI_48 (all taken from the most recent row)
              3) rsi_dip = (RSI_12 <= 40)  <-- OLD logic
                 **OR** (RSI_12 at the minimum MACD histogram point is below some threshold) <-- NEW logic

            If both conditions are met, send a notification using self.notifier.
            Additionally, find the RSI at the minimum MACD histogram of the latest negative run,
            and log or notify based on that RSI dip value.

            :param interval: Interval string (e.g. '1m', '5m') to look up data.
            """
            # Retrieve historical DataFrame for the chosen interval
            interval_data = all_features['history'].get(interval, pd.DataFrame())
            if interval_data.empty:
                logging.info(f"[macd_negative_rising] No historical data for interval '{interval}'.")
                return

            required_cols = ['MACD_hist_fast', 'RSI_12', 'RSI_24', 'RSI_48']
            for col in required_cols:
                if col not in interval_data.columns:
                    logging.info(
                        f"[macd_negative_rising] Missing '{col}' in data for '{interval}'."
                    )
                    return

            macd_hist = interval_data['MACD_hist_fast']
            rsi_12_series = interval_data['RSI_12']
            rsi_24_series = interval_data['RSI_24']
            rsi_48_series = interval_data['RSI_48']

            if len(macd_hist) < 3:
                logging.info(
                    f"[macd_negative_rising] Not enough MACD_hist data for '{interval}'. Need at least 3 data points."
                )
                return

            # ------------------------------------------------------------------------
            # 1) Original logic: Check if MACD histogram is negative but rising
            #    between the last two bars: macd_hist[-3], macd_hist[-2].
            # ------------------------------------------------------------------------
            hist_neg2 = macd_hist.iloc[-2]  # 3rd from last
            hist_neg1 = macd_hist.iloc[-1]  # 2nd from last

            macd_negative_rising = (hist_neg2 < 0 and hist_neg1 < 0 and hist_neg1 > hist_neg2)

            # 2) Check RSI chain from the last row
            rsi_12 = rsi_12_series.iloc[-1]
            rsi_24 = rsi_24_series.iloc[-1]
            rsi_48 = rsi_48_series.iloc[-1]

            rsi_chain = (rsi_12 < rsi_24 < rsi_48)

            # ------------------------------------------------------------------------
            # 3) NEW logic: Identify the "latest downtrend" in MACD_hist
            #    and find the candle with the minimum MACD_hist in that run.
            # ------------------------------------------------------------------------
            negative_macd_indices = macd_hist[macd_hist < 0].index
            if len(negative_macd_indices) < 1:
                # No negative MACD histogram at all
                logging.info(f"[macd_negative_rising] No negative MACD_hist run for '{interval}'.")
                return

            # Find the last consecutive run of negative MACD values
            # Starting from the end, group backward until a gap is found
            latest_downtrend = [negative_macd_indices[-1]]
            for idx in range(len(negative_macd_indices) - 2, -1, -1):
                if negative_macd_indices[idx + 1] - negative_macd_indices[idx] == 1:
                    # Consecutive
                    latest_downtrend.append(negative_macd_indices[idx])
                else:
                    # Gap found, stop
                    break

            # Because we appended in reverse, reorder the list ascending by time
            latest_downtrend.sort()

            # Find the index with the minimum MACD_hist value within this run
            min_macd_hist_index = macd_hist[latest_downtrend].idxmin()
            min_macd_value = macd_hist[min_macd_hist_index]

            # ------------------------------------------------------------------------
            # 4) Grab the RSI at that same candle
            # ------------------------------------------------------------------------
            rsi_at_min_macd = rsi_12_series[min_macd_hist_index]
            # Decide how you want to define an “RSI dip”
            # For example, let's say RSI < 35 means it's a “dip”
            rsi_dip_threshold = 40
            rsi_dip_condition = (rsi_at_min_macd <= rsi_dip_threshold)

            # ------------------------------------------------------------------------
            # 5) Combine conditions:
            #    - MACD negative but rising on the last 2 bars
            #    - RSI chain on the last bar
            #    - RSI at the min MACD histogram is also a “dip”
            # ------------------------------------------------------------------------
            if macd_negative_rising and rsi_chain and rsi_dip_condition:
                message = (
                    f"MACD negative Rising on interval '{interval}'\n"
                    f"MACD Hist last two: (-2)={hist_neg2:.4f}, (-1)={hist_neg1:.4f}\n"
                    f"RSI_12 chain last bar: RSI_12={rsi_12:.2f}, RSI_24={rsi_24:.2f}, RSI_48={rsi_48:.2f}\n"
                    f"Latest Downtrend min MACD: {min_macd_value:.4f} at index {min_macd_hist_index}\n"
                    f"RSI@MinMACD={rsi_at_min_macd:.2f}, threshold={rsi_dip_threshold}"
                )
                logging.info("[macd_negative_rising] " + message)
                return message
            else:
                logging.info(
                    f"[macd_negative_rising] Condition not fully met at interval '{interval}'.\n"
                    f"  Last bars: MACD Hist (-2)={hist_neg2:.4f}, (-1)={hist_neg1:.4f};\n"
                    f"  RSI chain last bar: (12={rsi_12:.2f}, 24={rsi_24:.2f}, 48={rsi_48:.2f}), chain={rsi_chain};\n"
                    f"  RSI@MinMACD={rsi_at_min_macd:.2f} vs dip_threshold={rsi_dip_threshold} => dip_condition={rsi_dip_condition}."
                )
                return None

        def notify_macd_negative_rising(macd_counter):
            message = macd_negative_rising(scalping_interval)
            if message is not None:
                macd_counter += 1
                save_flag_to_file(scalping_interval, 'macd_counter', macd_counter)
            else:
                macd_counter = 0
                save_flag_to_file(scalping_interval, 'macd_counter', macd_counter)

            if macd_counter in range(1,4):
                self.notifier.send_notification("MACD Negative Rising", message)

        def notify_break_resistance(break_counter):
            latest_resistance = self.detect_latest_resistance(all_features=all_features,interval=scalping_interval, current_price=current_price)
            if latest_resistance is not None and current_price > latest_resistance:
                break_counter += 1
                save_flag_to_file(scalping_interval, 'break_counter', break_counter)
            else:
                break_counter = 0
                save_flag_to_file(scalping_interval, 'break_counter', break_counter)

            if break_counter in range(1,4):
                breakout_msg = (
                    f"Price has just broken above the latest resistance of "
                    f"{latest_resistance:.4f} on interval '{scalping_interval}'.\n"
                    f"Current Price: {current_price:.4f}")
                self.notifier.send_notification("Break Resistance", breakout_msg)


        # Get signals from the defined functions ----------------------------------------------------------------------
        # Test uptrend momentum for 30 consecutive cycles
        uptrend_signal = testing_uptrend_momentum(uptrend_counter)
        # stoch_signal = stoch_rsi_signal(lowest_k_reached)
        rsi_signal_value = rsi_signal(scalping_interval)
        trailing_signal = gain_trailing_lock(max_gain_reached)
        macd_positive_angle = macd_positive_angle(scalping_interval)
        notify_macd_negative_rising(macd_counter)
        notify_break_resistance(break_counter)

        # Decision logic based on Uptrend ,StochRSI, RSI, and gain trailing signals
        # Sell
        if scalping_positions:

            stop_loss_scalping_value = self.loading_stop_loss_scalping()

            # if stoch_signal == 'overbought' or overbought_reached == True:
            #     overbought_reached = True
            #     save_flag_to_file(scalping_interval, 'overbought_reached', overbought_reached)
            #     log_message = "Scalping Decision: ||Overbought Reached|| - (StochRSI: overbought)"
            #     print(log_message)
            #     logging.info(log_message)
            #     # Wait for RSI to give RSI_Down signal
            #     if rsi_signal_value != 'RSI_Up':
            #         log_message = "Scalping Decision: ||Sell_Sc|| - (RSI: RSI_Down after overbought)"
            #         print(log_message)
            #         logging.info(log_message)
            #         reset_flag(scalping_interval, 'overbought_reached')
            #         reset_flag(scalping_interval, 'max_gain_reached')
            #         reset_flag(scalping_interval, 'lowest_k_reached')
            #         return 'Sell_Sc'


            # if trailing_signal == 'trailing_sell':
            #     log_message = "Scalping Decision: ||Sell_Sc|| - (Trailing Gain Signal Activated)"
            #     print(log_message)
            #     logging.info(log_message)
            #     reset_flag(scalping_interval, 'overbought_reached')
            #     reset_flag(scalping_interval, 'max_gain_reached')
            #     reset_flag(scalping_interval, 'lowest_k_reached')
            #     return 'Sell_Sc'

            if rsi_signal_value == 'RSI_Down' and not macd_positive_angle:
                log_message = "Scalping Decision: ||Sell_Sc|| - (RSI Down Signal Activated)"
                print(log_message)
                logging.info(log_message)
                reset_flag(scalping_interval, 'overbought_reached')
                reset_flag(scalping_interval, 'max_gain_reached')
                reset_flag(scalping_interval, 'lowest_k_reached')
                return 'Sell_Sc'

        # Buy
        else:

            if uptrend_signal and rsi_signal_value == 'RSI_Up' and macd_positive_angle and market_stable:
                log_message = "Scalping Decision: ||Buy_Sc|| - (StochRSI: oversold, RSI: RSI_Down)"
                reset_flag(scalping_interval, 'lowest_k_reached')
                reset_flag(scalping_interval, 'uptrend_counter')
                print(log_message)
                logging.info(log_message)
                return 'Buy_Sc'

        # If no conditions are met, return 'Hold'
        log_message = "Scalping Decision: ||Hold|| - (No definitive conditions met for Buy or Sell)"
        print(log_message)
        logging.info(log_message)
        return 'Hold'









