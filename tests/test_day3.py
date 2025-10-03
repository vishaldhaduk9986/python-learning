import unittest
from unittest.mock import patch
import src.day3 as day3

class TestDay3WeatherAPI(unittest.TestCase):
    @patch("src.day3.requests.get")
    def test_weather_api_response(self, mock_get):
        mock_get.return_value.json.return_value = {
            "weather": [{"description": "clear sky"}],
            "main": {"temp": 30}
        }
        response = day3.requests.get(day3.url)
        data = response.json()
        self.assertIn("weather", data)
        self.assertIn("main", data)
        self.assertEqual(data["main"]["temp"], 30)
        self.assertEqual(data["weather"][0]["description"], "clear sky")

    @patch("src.day3.requests.get")
    def test_invalid_api_key(self, mock_get):
        mock_get.return_value.json.return_value = {
            "cod": 401,
            "message": "Invalid API key."
        }
        response = day3.requests.get(day3.url)
        data = response.json()
        self.assertEqual(data["cod"], 401)
        self.assertIn("Invalid API key", data["message"])

if __name__ == "__main__":
    unittest.main()
