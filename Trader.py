import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from decimal import Decimal, ROUND_DOWN

class Trader:
    def __init__(self, symbol):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance API key and secret must be set in environment variables.")

        self.symbol = symbol
        self.client = Client(self.api_key, self.api_secret)
        self.step_size = self.get_step_size()

    def get_step_size(self):
        try:
            # Fetch exchange information to get the LOT_SIZE filter for the symbol
            exchange_info = self.client.get_symbol_info(self.symbol)
            for filter in exchange_info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    return float(filter['stepSize'])
        except Exception as e:
            raise Exception(f"Error fetching LOT_SIZE for {self.symbol}: {e}")

    def adjust_quantity(self, quantity):
        step_size_decimal = Decimal(str(self.step_size))
        quantity_decimal = Decimal(str(quantity))
        adjusted_quantity = (quantity_decimal // step_size_decimal) * step_size_decimal
        return float(adjusted_quantity)

    def execute_trade(self, decision, amount):
        try:
            # Get the current price of Currency in USDT
            current_price = self.get_current_price()
            if not current_price:
                return "Error", "Failed to fetch current price."

            # Adjust the amount to comply with Binance's LOT_SIZE
            adjusted_amount = self.adjust_quantity(amount)
            print(f"Adjusted trade amount {adjusted_amount}")
            # Check if the adjusted amount is valid (greater than zero)
            if adjusted_amount <= 0:
                return "Error", f"Adjusted trade amount {adjusted_amount} is not valid."

            # Execute the trade based on the decision
            if decision == "Buy" or decision == "Buy_Sc":
                order = self.client.order_market_buy(symbol=self.symbol, quantity=adjusted_amount)
                print(f"Buy Order Executed: {order}")
                return "Success", order
            elif decision == "Sell" or decision == "Sell_Sc":
                order = self.client.order_market_sell(symbol=self.symbol, quantity=adjusted_amount)
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
