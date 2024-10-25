import os
import requests
import logging
from Notifier import Notifier

class FearGreedIndex:
    def __init__(self):
        self.api_url = "https://api.alternative.me/fng/"
        self.index_value = None
        self.index_classification = None
        self.notifier = Notifier()

    def fetch_index(self, timeout=5):
        try:
            response = requests.get(f"{self.api_url}?limit=1&format=json", timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    index_data = data['data'][0]
                    self.index_value = int(index_data['value'])
                    self.index_classification = index_data['value_classification']
                    return self.index_value, self.index_classification
                else:
                    logging.error("No data available in response.")
            else:
                logging.error(f"Failed to fetch data from API. Status Code: {response.status_code}")
        except requests.RequestException as e:
            logging.error(f"Error fetching Fear & Greed Index: {e}")
            self.index_value = 'NA'
            self.index_classification = 'NA'
            self.notifier.send_notification("API Timeout Notification", "Error: Fear & Greed Index API request timed out.")
            return self.index_value, self.index_classification

    def get_index(self):
        if self.index_value is not None:
            return self.index_value, self.index_classification
        else:
            return self.fetch_index()


# Example usage:
if __name__ == "__main__":
    fear_greed_index = FearGreedIndex()
    index_value, index_classification = fear_greed_index.get_index()
    if index_value is not None:
        print(f"Fear & Greed Index Value: {index_value}")
        print(f"Classification: {index_classification}")
    else:
        print("Unable to retrieve Fear & Greed Index.")