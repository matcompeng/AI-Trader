class DecisionMaker:
    def __init__(self, base_risk_tolerance=0.02, base_stop_loss=0.0005, base_take_profit=None,
                 volatility_threshold=0.01, profit_interval=None, loose_interval=None):
        self.base_risk_tolerance = base_risk_tolerance
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.volatility_threshold = volatility_threshold
        self.profit_interval = profit_interval
        self.loose_interval = loose_interval

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
            band_width_15m = upper_band_profit - lower_band_profit

            if band_width_15m == 0:
                return self.base_take_profit  # Avoid division by zero

            # Calculate the price change ratio using entry price
            price_change_ratio = ((upper_band_profit - entry_price) / band_width_15m)

            # Calculate the adjusted take profit
            adjusted_take_profit = self.base_take_profit * (1 + price_change_ratio)
            if adjusted_take_profit < self.base_take_profit:
                return self.base_take_profit

            return adjusted_take_profit

        return self.base_take_profit

    def make_decision(self, prediction, current_price, entry_price, all_features):
        """
        Make a final trading decision based on the prediction and risk management rules.
        :param prediction: The initial prediction from the Predictor (Buy, Sell, Hold).
        :param current_price: The current market price of the asset.
        :param entry_price: The price at which the current position was entered (if any).
        :param all_features: Dictionary containing features for multiple intervals.
        :return: Final decision (Buy, Sell, Hold), adjusted_stop_loss, adjusted_take_profit.
        """
        # Get the necessary data
        lower_band_profit = all_features[self.profit_interval].get('lower_band', None)
        upper_band_profit = all_features[self.profit_interval].get('upper_band', None)
        lower_band_loose = all_features[self.loose_interval].get('lower_band', None)
        middle_band_loose = all_features[self.loose_interval].get('middle_band', None)

        # Adjust stop_loss based on the lower band of 15m interval
        adjusted_stop_loss_middle = middle_band_loose
        adjusted_stop_loss_lower = lower_band_loose

        # Calculate adjusted take profit using entry price and 15m bands
        adjusted_take_profit = self.calculate_adjusted_take_profit(entry_price, upper_band_profit, lower_band_profit)

        if prediction == "Buy":
            if self.is_market_stable(all_features):
                return "Buy", adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit
            else:
                return "Hold", adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit
        elif prediction == "Hold" and entry_price:
            if self.should_sell(current_price, entry_price, adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit):
                return "Sell", adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit
            else:
                return "Hold", adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit
        elif prediction == "Sell" and entry_price:
            return "Sell", adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit
        else:
            return "Hold", adjusted_stop_loss_lower,adjusted_stop_loss_middle, adjusted_take_profit

    def is_market_stable(self, all_features):
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
                if lower_band < close_price < upper_band:
                    stable_intervals += 1

        # Consider the market stable if a majority of intervals indicate stability
        # if stable_intervals >= (total_intervals * 2 * 0.75):  # e.g., 4 out of 5 intervals must be stable
        if stable_intervals == (total_intervals * 2):  # e.g., 4 out of 5 intervals must be stable
            return True

        return False

    def should_sell(self, current_price, entry_price, adjusted_stop_loss_lower, adjusted_stop_loss_middle,
                    adjusted_take_profit):
        # Calculate the percentage change from the entry price
        price_change = ((current_price - entry_price) / entry_price) * 100

        # Check if the price has hit the take-profit threshold
        if price_change >= adjusted_take_profit:
            return True

        # Check stop-loss conditions
        if entry_price > adjusted_stop_loss_lower:
            # Sell only if current_price is below adjusted_stop_loss_lower
            if current_price < adjusted_stop_loss_lower:
                return True
        elif entry_price > adjusted_stop_loss_middle:
            # Sell only if current_price is below adjusted_stop_loss_middle
            if current_price < adjusted_stop_loss_middle:
                return True

        # If none of the above conditions are met, do not sell
        return False
