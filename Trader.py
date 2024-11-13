import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

class Trader:
    def __init__(self, symbol):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API key and secret must be set in environment variables.")

        self.symbol = symbol
        self.client = Client(self.api_key, self.api_secret)

    def execute_trade(self, decision, amount):
        try:
            # Get the current price of Currency in USDT
            current_price = self.get_current_price()
            if not current_price:
                return "Error", "Failed to fetch current price."

            # Execute the trade based on the decision
            if decision == "Buy" or decision == "Buy_Dip":
                order = self.client.order_market_buy(symbol=self.symbol, quantity=amount)
                print(f"Buy Order Executed: {order}")
                return "Success", order
            elif decision == "Sell":
                order = self.client.order_market_sell(symbol=self.symbol, quantity=amount)
                print(f"Sell Order Executed: {order}")
                return "Success", order
            else:
                print("No trade executed. Decision was to Hold.")
                return "NoAction", None
        except BinanceAPIException as e:
            print(f"Binance API Error executing trade: {e}")
            return "Error", str(e)
        except BinanceOrderException as e:
            print(f"Binance Order Error executing trade: {e}")
            return "Error", str(e)
        except Exception as e:
            print(f"Error executing trade: {e}")
            return "Error", str(e)

    def get_current_price(self):
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            print(f"Error fetching current price: {e}")
            raise Exception(f"Failed to get current price for {self.symbol}: {e}")

# Example usage:
if __name__ == "__main__":
    # Initialize the Trader with environment variables
    trader = Trader(symbol='BTCUSDT')

    # Example: Simulate a trading decision
    decision = "Sell"  # Example decision from Predictor

    # Execute the trade based on the decision with a BTC amount
    btc_amount = 0.001  # Example BTC amount to trade
    trader.execute_trade(decision, btc_amount)
