class DecisionMaker:
    def __init__(self, base_risk_tolerance=0.01, base_stop_loss=0.005, base_take_profit=0.0045, volatility_threshold=0.02):
        self.base_risk_tolerance = base_risk_tolerance
        self.base_stop_loss = base_stop_loss
        self.base_take_profit = base_take_profit
        self.volatility_threshold = volatility_threshold

    def calculate_dynamic_risk_tolerance(self, atr_1d, close_price):
        """
        Calculate dynamic risk tolerance based on the ATR of the 5-minute interval.
        :param atr_1d: The ATR value from the 1d interval.
        :param close_price: The current closing price.
        :return: Adjusted risk tolerance.
        """
        if atr_1d and close_price:
            # Adjust risk tolerance based on market volatility
            relative_atr = atr_1d / close_price
            dynamic_risk_tolerance = self.base_risk_tolerance * (1 + relative_atr)
            return dynamic_risk_tolerance
        return self.base_risk_tolerance

    def make_decision(self, prediction, current_price, entry_price, all_features):
        """
        Make a final trading decision based on the prediction and risk management rules.
        :param prediction: The initial prediction from the Predictor (Buy, Sell, Hold).
        :param current_price: The current market price of the asset.
        :param entry_price: The price at which the current position was entered (if any).
        :param all_features: Dictionary containing features for multiple intervals.
        :return: Final decision (Buy, Sell, Hold), adjusted_stop_loss, adjusted_take_profit.
        """
        # Get ATR and close price from the 5-minute interval
        atr_1d = all_features['1d'].get('ATR', None)
        close_price_1d = all_features['1d'].get('close', None)

        # Calculate dynamic risk tolerance based on the 5-minute ATR
        risk_tolerance = self.calculate_dynamic_risk_tolerance(atr_1d, close_price_1d)

        # Adjust stop_loss and take_profit based on the dynamic risk tolerance
        adjusted_stop_loss = self.base_stop_loss * (1 + risk_tolerance)
        adjusted_take_profit = self.base_take_profit * (1 + risk_tolerance)

        if prediction == "Buy":
            if self.is_market_stable(all_features):
                return "Buy", adjusted_stop_loss, adjusted_take_profit
            else:
                return "Hold", adjusted_stop_loss, adjusted_take_profit
        elif prediction == "Hold" and entry_price:
            if self.should_sell(current_price, entry_price, adjusted_stop_loss, adjusted_take_profit):
                return "Sell", adjusted_stop_loss, adjusted_take_profit
            else:
                return "Hold", adjusted_stop_loss, adjusted_take_profit
        elif prediction == "Sell" and entry_price:
            return "Sell", adjusted_stop_loss, adjusted_take_profit
        else:
            return "Hold", adjusted_stop_loss, adjusted_take_profit

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
            atr = features.get('ATR', None)
            close_price = features.get('close', None)
            if atr and close_price:
                relative_atr = atr / close_price
                if relative_atr <= self.volatility_threshold:
                    stable_intervals += 1

            # Additional checks for stability could include:
            rsi = features.get('RSI', None)
            if rsi and 30 <= rsi <= 70:
                stable_intervals += 1

            upper_band = features.get('upper_band', None)
            lower_band = features.get('lower_band', None)
            if upper_band and lower_band and close_price:
                if lower_band < close_price < upper_band:
                    stable_intervals += 1

        # Consider the market stable if a majority of intervals indicate stability
        if stable_intervals >= (total_intervals * 2 / 4):  # e.g., 4 out of 7 intervals must be stable
            return True

        return False

    def should_sell(self, current_price, entry_price, adjusted_stop_loss, adjusted_take_profit):
        # Calculate the percentage change from the entry price
        price_change = (current_price - entry_price) / entry_price

        # Check if the price has hit the stop-loss or take-profit threshold
        if price_change <= -adjusted_stop_loss or price_change >= adjusted_take_profit:
            return True
        return False
