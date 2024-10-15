import os
import requests

class Notifier:
    def __init__(self):
        self.user_key = os.getenv("PUSHOVER_USER_KEY")
        self.api_token = os.getenv("PUSHOVER_API_TOKEN")
        if not self.user_key or not self.api_token:
            raise ValueError("Pushover user key and API token must be set in environment variables.")
        self.api_url = "https://api.pushover.net/1/messages.json"

    def send_notification(self, title, message, sound="pushover"):
        try:
            payload = {
                "token": self.api_token,
                "user": self.user_key,
                "title": title,
                "message": message,
                "sound": sound
            }
            response = requests.post(self.api_url, data=payload)
            if response.status_code == 200:
                print(f"Notification sent: {title} - {message}")
            else:
                print(f"Failed to send notification: {title} - {message}. Status Code: {response.status_code}")
        except Exception as e:
            print(f"Error in sending notification: {e}")

    def send_test_notification(self):
        self.send_notification("Test Notification", "This is a test message from the BTC bot.")

# Example usage:
if __name__ == "__main__":
    notifier = Notifier()

    # Send a test notification
    notifier.send_test_notification()

    # Example notification for a trade execution
    notifier.send_notification("Trade Executed", "Bought 0.001 BTC at $45,000")