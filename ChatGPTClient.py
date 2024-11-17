import openai
import os


class ChatGPTClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found. Make sure to set the OPENAI_API_KEY environment variable.")
        openai.api_key = self.api_key

    def get_prediction(self, prompt, temperature=0.0):  # Set default temperature to 0 for deterministic responses
        try:
            # Ensure the prompt is a string
            if not isinstance(prompt, str):
                raise ValueError("Prompt must be a string.")

            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Explicitly use GPT-4o
                messages=[
                    {"role": "system",
                     "content": "Imagine you are an analyst who is double-checking every piece of data provided by a prompt for accuracy. Take your time to analyze each indicator and rule consider their interplay before making a decision."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature  # Set temperature parameter here
            )
            # Return the content of the response message
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error in ChatGPT API call: {e}")
            return str(e)

    def get_validation(self, strategy_and_response, temperature=0.0):
        """
        Perform cross-check validation of the GPT's prior analysis against the user's submitted trading strategy.

        Args:
            strategy_and_response (str): The user-provided prompt containing both the trading strategy and the GPT's prior analysis.
            temperature (float): Temperature for controlling response randomness (default: 0.0 for deterministic response).

        Returns:
            str: Validation results indicating discrepancies or confirming the accuracy of the GPT's prior response.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Explicitly use GPT-4o
                messages=[
                    {"role": "system",
                     "content": "You are a critical validator. Your task is to cross-check a previously generated GPT response against the provided trading strategy. Ensure every rule, condition, and decision aligns accurately with the strategy and flag any discrepancies or misinterpretations."},
                    {"role": "user", "content": strategy_and_response},
                ],
                temperature=temperature  # Set temperature parameter here
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error in ChatGPT API call: {e}")
            return str(e)
