class DecisionMaker:
    def __init__(self, risk_tolerance=0.01, stop_loss=0.005, take_profit=0.0045, volatility_threshold=0.02):
        self.risk_tolerance = risk_tolerance
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.volatility_threshold = volatility_threshold  # Threshold for market volatility

    def make_decision(self, prediction, current_price, entry_price, all_features):
        """
        Make a final trading decision based on the prediction and risk management rules.
        :param prediction: The initial prediction from the Predictor (Buy, Sell, Hold).
        :param current_price: The current market price of the asset.
        :param entry_price: The price at which the current position was entered (if any).
        :param all_features: Dictionary containing features for multiple intervals.
        :return: Final decision (Buy, Sell, Hold).
        """
        if prediction == "Buy":
            if self.is_market_stable(all_features):
                return "Buy"
            else:
                return "Hold"
        elif prediction == "Hold" and entry_price:
            if self.should_sell(current_price, entry_price):
                return "Sell"
            else:
                return "Hold"
        elif prediction == "Sell" and entry_price:
            return "Sell"
        else:
            return "Hold"

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

    def should_sell(self, current_price, entry_price):
        # Calculate the percentage change from the entry price
        price_change = (current_price - entry_price) / entry_price

        # Check if the price has hit the stop-loss or take-profit threshold
        if price_change >= self.take_profit:
            return True
        return False


# Example usage:
if __name__ == "__main__":
    decision_maker = DecisionMaker()

    # Simulate a prediction from the Predictor
    prediction = "Sell"

    # Simulate current and entry prices
    current_price = 45000
    entry_price = 44000

    # Make the final decision
    final_decision = decision_maker.make_decision(prediction, current_price, entry_price)

    print(f"Final Decision: {final_decision}")