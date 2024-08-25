import os
import ccxt

class Trader:
    def __init__(self, symbol):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API key and secret must be set in environment variables.")

        self.symbol = symbol
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
        })

    def execute_trade(self, decision, usdt_amount):
        try:
            # Get the current price of Currency in USDT
            current_price = self.get_current_price()
            if not current_price:
                return "Error", "Failed to fetch current price."

            # Convert the USDT amount to Currency amount
            trading_amount = usdt_amount / current_price

            # Round the Currency amount to 5 decimal places
            adjusted_trading_amount = round(trading_amount, 5)

            # Execute the trade based on the decision
            if decision == "Buy":
                order = self.exchange.create_market_buy_order(self.symbol, adjusted_trading_amount)
                print(f"Buy Order Executed: {order}")
                return "Success", order
            elif decision == "Sell":
                order = self.exchange.create_market_sell_order(self.symbol, adjusted_trading_amount)
                print(f"Sell Order Executed: {order}")
                return "Success", order
            else:
                print("No trade executed. Decision was to Hold.")
                return "NoAction", None
        except Exception as e:
            print(f"Error executing trade: {e}")
            return "Error", str(e)

    def get_current_price(self):
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None


# Example usage:
if __name__ == "__main__":
    # Initialize the Trader with environment variables
    trader = Trader(symbol='BTC/USDT')

    # Example: Simulate a trading decision
    decision = "Sell"  # Example decision from Predictor

    # Execute the trade based on the decision with a USDT amount
    usdt_amount = 10  # Example USDT amount to trade
    trader.execute_trade(decision, usdt_amount)
