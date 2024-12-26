AI-Trader Bot 📈🤖

Welcome to the AI-Trader Bot repository! This bot is an advanced cryptocurrency trading solution that leverages technical indicators, trading algorithms, and real-time market data to make informed trading decisions. It is designed to work seamlessly with Binance’s API for executing trades while implementing robust error handling and risk management mechanisms.

🚀 Features
	1.	Automated Trading:
	•	Executes market buy and sell orders on Binance based on dynamic trading strategies.
	2.	Technical Analysis:
	•	Calculates technical indicators such as RSI, MACD, Bollinger Bands, Stochastic RSI, VWAP, and more.
	3.	Dynamic Risk Management:
	•	Implements stop-loss, take-profit, and trailing profit mechanisms.
	4.	Scalping and Swing Trading:
	•	Supports both short-term scalping and long-term trading strategies.
	5.	Real-Time Notifications:
	•	Alerts on critical market events (e.g., resistance breakouts, MACD signals) using a notification service.
	6.	AI Integration:
	•	Utilizes OpenAI’s GPT API for validating and enhancing trading decisions.


 🛠️ Classes Overview:

1. BotManager

The central manager for coordinating the bot’s operations.
	•	Handles Initialization:
	•	Sets up all components (Trader, Notifier, DecisionMaker, etc.).
	•	Orchestrates Workflow:
	•	Oversees data collection, feature processing, decision-making, and trade execution.
	•	Error Management:
	•	Captures errors from various components and ensures the bot continues running robustly.
	•	Logging:
	•	Provides detailed logs of trading decisions, errors, and market activity.

2. DataCollector

Collects real-time market data from Binance.
	•	OHLCV Data:
	•	Fetches historical candlestick data for multiple intervals.
	•	Order Book:
	•	Captures bid/ask volumes to identify order flow dynamics.
	•	Persistence:
	•	Saves raw and processed data locally for future use.

3. FeatureProcessor

Processes raw market data into actionable features.
	•	Technical Indicator Calculation:
	•	Computes RSI, MACD, Bollinger Bands, ATR, VWAP, and more.
	•	Order Book Analysis:
	•	Identifies significant support and resistance levels from the order book.
	•	Feature Enrichment:
	•	Enhances data with trend and momentum features for scalping and swing trading.

4. DecisionMaker

The core decision-making engine for the bot.
	•	Trading Signal Detection:
	•	Analyzes market conditions to generate Buy, Sell, or Hold decisions.
	•	Scalping Strategies:
	•	Detects short-term opportunities using Bollinger Bands, MACD, and Stochastic RSI.
	•	Dynamic Risk Management:
	•	Adjusts stop-loss and take-profit levels based on market volatility.
	•	Notifications:
	•	Sends alerts for events like resistance breakouts and MACD rising signals.
	•	Resistance/Support Detection:
	•	Dynamically calculates resistance and support levels based on price action.

5. ChatGPTClient

A utility class for interacting with the OpenAI GPT API.
	•	Prompt Management:
	•	Sends prompts for trading strategy validation or prediction generation.
	•	Error Handling:
	•	Retries failed API calls and logs issues for transparency.
	•	Customizable Output:

6. Predictor

Integrates OpenAI’s GPT API for decision validation and enhancement.
	•	Dynamic Prompt Generation:
	•	Creates prompts based on the trading strategy and market data.
	•	Prediction and Validation:
	•	Sends predictions to OpenAI for validation, ensuring alignment with the strategy.
	•	Error Handling:
	•	Retries API requests and logs results for transparency.


7. Trader

Handles trade execution on Binance.
	•	Fetches Exchange Information:
	•	Retrieves the LOT_SIZE filter for the trading pair to adjust trade quantities.
	•	Executes Market Trades:
	•	Places buy or sell orders based on decisions (Buy, Sell, Hold).
	•	Error Handling:
	•	Manages Binance API and order-related exceptions.
	•	Fetches Current Price:
	•	Ensures trades are executed with the latest ticker price.


8. Notifier

Sends real-time notifications to alert the user of critical market events.
	•	Pushover Integration:
	•	Delivers notifications for resistance breakouts, MACD signals, and trade executions.
	•	Error Alerts:
	•	Notifies on API failures or unexpected conditions.


9. FearGreedIndex

Fetches the Fear & Greed Index from the alternative.me API.
	•	Market Sentiment:
	•	Provides insights into market psychology (e.g., “Extreme Fear” or “Greed”).
	•	Notifications:
	•	Alerts the user when extreme sentiment is detected.



 📈 Workflow
	1.	Data Collection:
	•	DataCollector fetches market data (OHLCV, order book).
	2.	Feature Processing:
	•	FeatureProcessor calculates indicators and extracts features.
	3.	Trading Decision:
	•	DecisionMaker uses features to generate trading signals.
	•	Predictor validates signals using AI integration.
	4.	Trade Execution:
	•	Trader places buy or sell orders on Binance.
	5.	Notifications:
	•	Notifier alerts the user about key events.

 📄 Installation and Setup

Prerequisites
	•	Python 3.8+
	•	Binance account with API key and secret
	•	Pushover account for notifications
	•	OpenAI API key for decision validation

Installation
	1.	Clone the repository: 
 git clone https://github.com/matcompeng/AI-Trader
cd AI-Trader-Bot

	2.	Install dependencies:
pip install -r requirements.txt

	3.	Set up environment variables:
 export BINANCE_API_KEY='your_api_key'
export BINANCE_API_SECRET='your_api_secret'
export OPENAI_API_KEY='your_openai_api_key'

	4.	Run the bot:
 python BotManager.py


 🔧 Configuration

Configure settings:
	•	Trading pairs: Specify trading symbols (e.g., BTCUSDT).
	•	Intervals: Set analysis intervals (e.g., 1m, 5m, 15m).
	•	Risk parameters: Adjust stop-loss and take-profit thresholds.

 ⚠️ Disclaimer

This bot is provided as-is and should be used at your own risk. The authors are not responsible for any financial losses incurred. Always test in a simulated environment before deploying to live trading.


💡 Future Enhancements
	•	Add machine learning models for predictive analytics.
	•	Support additional trading platforms beyond Binance.
	•	Implement advanced backtesting and simulation tools.
