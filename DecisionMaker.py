class DecisionMaker:
    def __init__(self, risk_tolerance=0.01, stop_loss=0.005, take_profit=0.5):
        self.risk_tolerance = risk_tolerance
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def make_decision(self, prediction, current_price, entry_price):
        """
        Make a final trading decision based on the prediction and risk management rules.
        :param prediction: The initial prediction from the Predictor (Buy, Sell, Hold).
        :param current_price: The current market price of the asset.
        :param entry_price: The price at which the current position was entered (if any).
        :return: Final decision (Buy, Sell, Hold).
        """
        if prediction == "Buy":
            # Example: Only buy if the market is not too volatile
            if self.is_market_stable(current_price):
                return "Buy"
            else:
                return "Hold"
        elif prediction == "Sell" and entry_price:
            # Example: Implement stop-loss and take-profit
            if self.should_sell(current_price, entry_price):
                return "Sell"
            else:
                return "Hold"
        else:
            return "Hold"

    def is_market_stable(self, current_price):
        # Placeholder for logic to check market stability, e.g., based on volatility
        # For now, we'll just assume the market is stable
        return True

    def should_sell(self, current_price, entry_price):
        # Calculate the percentage change from the entry price
        price_change = (current_price - entry_price) / entry_price

        # Check if the price has hit the stop-loss or take-profit threshold
        if price_change <= -self.stop_loss or price_change >= self.take_profit:
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