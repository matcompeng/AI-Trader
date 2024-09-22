import json
import os


class DecisionMaker:
    def __init__(self, risk_tolerance=None, base_stop_loss=None, base_take_profit=None, profit_interval=None,
                 loose_interval=None, dip_interval=None, amount_rsi_interval=None, amount_atr_interval=None, min_stable_intervals=None, gain_sell_threshold=None, data_directory='data'):
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
        only for positions where the dip_flag is not equal to position['dip'].
        This method uses the calculate_gain_loose and invested_budget methods from BotManager class.
        :param bot_manager: Instance of BotManager to access existing methods.
        :param current_price: The current market price to compare with entry prices.
        :return: Total portfolio gain in percentage for positions where dip_flag != position['dip'].
        """
        stable_invested, dip_invested, total_invested = bot_manager.invested_budget()  # Use the existing invested_budget method
        total_gain = 0.0

        # Iterate over each position and calculate gain/loss only if dip_flag != position['dip']
        for position_id, position in bot_manager.position_manager.get_positions().items():
            dip_flag = position.get('dip', None)

            # Check if the dip_flag condition is met
            if dip_flag == 0:
                entry_price = float(position['entry_price'])
                amount = float(position['amount'])
                gain_loss = bot_manager.calculate_gain_loose(entry_price, current_price)

                invested_amount = entry_price * amount
                total_gain += (gain_loss / 100) * invested_amount

        # Calculate the overall percentage gain/loss
        if stable_invested > 0:
            portfolio_gain_percent = (total_gain / stable_invested) * 100
        else:
            portfolio_gain_percent = 0.0
        print(f"Portfolio Gain Percentage : {round(portfolio_gain_percent, 2)}%")

        return portfolio_gain_percent

    def check_for_sell_due_to_reversal(self, bot_manager, current_price):
        """
        Check if the portfolio gain has reached a maximum and lost 25% of that gain, triggering a sell decision.
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
            if total_portfolio_gain < self.max_gain * (1 - self.sell_threshold):
                print(f"Market has reversed. Current gain: {total_portfolio_gain:.2f}%, Max gain: {self.max_gain:.2f}%")
                self.max_gain = 0  # Reset The Maximum Gain
                self.save_max_gain()  # Save the reset max gain to the file
                return "Sell"
            else:
                print(f"Current Portfolio Gain: {total_portfolio_gain:.2f}%, Max Gain Reached{self.max_gain:.2f}% ,No reversal detected.")
                return "Hold"

        except Exception as e:
            print(f"Error in checking for sell due to reversal: {e}")
            return "Hold"


    def calculate_buy_amount(self, all_features ,amount_rsi_interval, amount_atr_interval, capital):
        """
        Calculate buy amount based on ATR (from 30m or 1h) and RSI (from 5m or 15m).

        :param capital:
        :param amount_atr_interval:
        :param amount_rsi_interval:
        :param all_features: A dictionary of dataframes for different intervals (e.g., '1m', '5m', '15m', '30m', '1h', '1d')
        :return: Recommended buy amount
        """
        # TODO: develop fear and greed index returns with amount calculation

        # Extract data for each interval
        current_atr = all_features[self.amount_atr_interval].get('ATR', None)
        current_rsi = all_features[self.amount_rsi_interval].get('RSI', None)

        # Example logic to adjust buy amount based on volatility and momentum
        volatility_factor = 1 / current_atr
        momentum_factor = 1.2 if current_rsi < 40 else 0.5 if current_rsi > 60 else 1.0

        # Adjust the buy amount based on both volatility and momentum factors
        adjusted_risk = self.risk_tolerance * volatility_factor * momentum_factor
        buy_amount = capital * adjusted_risk

        print(f"ATR ({amount_atr_interval}): {current_atr:.2f}, RSI ({amount_rsi_interval}): {current_rsi:.2f}, Buy Amount: {buy_amount:.2f}")
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


    def make_decision(self, prediction, current_price, entry_price, all_features, position_expired):
        """
        Make a final trading decision based on the prediction and risk management rules.
        :param position_expired:
        :param prediction: The initial prediction from the Predictor (Buy, Sell, Hold).
        :param current_price: The current market price of the asset.
        :param entry_price: The price at which the current position was entered (if any).
        :param all_features: Dictionary containing features for multiple intervals.
        :return: Final decision (Buy, Sell, Hold), adjusted_stop_loss, adjusted_take_profit.
        """
        # Get the necessary data
        lower_band_profit = all_features[self.profit_interval].get('lower_band', None)
        upper_band_profit = all_features[self.profit_interval].get('upper_band', None)
        lower_band_loss = all_features[self.loose_interval].get('lower_band', None)
        middle_band_loss = all_features[self.loose_interval].get('middle_band', None)
        upper_band_loss = all_features[self.loose_interval].get('upper_band', None)

        # Adjust stop_loss based
        adjusted_stop_loss_middle = self.calculate_adjusted_stop_middle(entry_price, upper_band_loss, middle_band_loss)
        adjusted_stop_loss_lower = self.calculate_adjusted_stop_lower(entry_price, lower_band_loss, middle_band_loss)

        # adjust take_profit base
        adjusted_take_profit = self.calculate_adjusted_take_profit(entry_price, upper_band_profit, lower_band_profit)

        if prediction == "Buy":
            if self.market_stable(all_features):
                return "Buy", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit
            elif self.is_there_dip(all_features):
                return "Buy_Dip", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit
            else:
                return "Hold", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

        elif prediction == "Hold" and self.is_there_dip(all_features) and not self.market_downtrend_stable(
                all_features):
            return "Buy_Dip", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

        elif prediction == "Hold" and entry_price:
            if self.should_sell(current_price, entry_price, adjusted_stop_loss_lower, adjusted_stop_loss_middle,
                                adjusted_take_profit, middle_band_loss, lower_band_loss, all_features, position_expired):
                return "Sell", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit
            else:
                return "Hold", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

        elif prediction == "Sell" and entry_price:
            return "Sell", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

        else:
            return "Hold", adjusted_stop_loss_lower, adjusted_stop_loss_middle, adjusted_take_profit

    def market_stable(self, all_features):
        """
        Check if the market is stable based on volatility and other criteria across multiple intervals.
        :param all_features: Dictionary containing features for multiple intervals.
        :return: True if the market is stable, False otherwise.
        """
        stable_intervals = 0
        total_intervals = len(all_features)

        for interval, features in all_features.items():
            # Check ATR (Average True Range) to measure volatility
            # atr = features.get('ATR', None)
            # close_price = features.get('close', None)
            # if atr and close_price:
            #     relative_atr = atr / close_price
            #     if relative_atr <= self.volatility_threshold:
            #         stable_intervals += 1

            # Additional checks for stability could include:
            rsi = features.get('RSI', None)
            if rsi and 30 <= rsi <= 70:
                stable_intervals += 1

            close_price = features.get('close', None)
            upper_band = features.get('upper_band', None)
            lower_band = features.get('lower_band', None)
            if upper_band and lower_band and close_price:
                if lower_band <= close_price <= upper_band:
                    stable_intervals += 1

        # Consider the market stable if a majority of intervals indicate stability
        # if stable_intervals >= (total_intervals * 2 * 0.75):  # e.g., 4 out of 5 intervals must be stable
        if total_intervals - (total_intervals - (stable_intervals / 2)) >= self.min_stable_intervals:  # e.g., 5 out of 6 intervals must be stable
            return True

        return False

    def market_downtrend_stable(self, all_features):
        """
        Check if the market is stable based on volatility and other criteria across multiple intervals.
        :param all_features: Dictionary containing features for multiple intervals.
        :return: True if the market is stable, False otherwise.
        """
        stable_intervals = 0
        total_intervals = len(all_features)

        for interval, features in all_features.items():
            # Check ATR (Average True Range) to measure volatility
            # atr = features.get('ATR', None)
            # close_price = features.get('close', None)
            # if atr and close_price:
            #     relative_atr = atr / close_price
            #     if relative_atr <= self.volatility_threshold:
            #         stable_intervals += 1

            # Additional checks for stability could include:
            rsi = features.get('RSI', None)
            if rsi >= 30:
                stable_intervals += 1

            close_price = features.get('close', None)
            lower_band = features.get('lower_band', None)
            if  lower_band and close_price:
                if close_price >= lower_band:
                    stable_intervals += 1

        # Consider the market stable if a majority of intervals indicate stability
        # if stable_intervals >= (total_intervals * 2 * 0.75):  # e.g., 4 out of 5 intervals must be stable
        if total_intervals - (total_intervals - (stable_intervals / 2)) >= self.min_stable_intervals:  # e.g., 5 out of 6 intervals must be stable
            return True

        return False

    def is_there_dip(self, all_features):

        interval_lower_band = all_features[self.dip_interval].get('lower_band', None)
        interval_close_price = all_features[self.dip_interval].get('close', None)

        if interval_close_price < interval_lower_band:
            return True
        return False


    def should_sell(self, current_price, entry_price, adjusted_stop_loss_lower, adjusted_stop_loss_middle,
                    adjusted_take_profit, middle_band_loss, lower_band_loss, all_features, position_expired):
        # Calculate the percentage change from the entry price
        price_change = ((current_price - entry_price) / entry_price) * 100

        # Check if the price has hit the take-profit threshold
        if price_change >= adjusted_take_profit:
            return True

        #Check is the market has unstable downtrend condition for position settlement
        elif not self.market_downtrend_stable(all_features):
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
