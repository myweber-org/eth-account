
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return {'error': f'Failed to fetch weather data: {str(e)}'}
        except KeyError as e:
            return {'error': f'Invalid response format: {str(e)}'}
    
    def _parse_weather_data(self, data):
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def display_weather(self, weather_data):
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
            return
        
        print("\n" + "="*40)
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print("="*40)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"Last Updated: {weather_data['timestamp']}")
        print("="*40)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Example: python fetch_weather.py London")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(api_key)
    weather_data = fetcher.get_weather(city_name)
    fetcher.display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.units = "metric"

    def get_weather(self, city_name):
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENWEATHER_API_KEY environment variable.")

        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': self.units
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch weather data: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid response format: {e}")

    def display_weather(self, city_name):
        try:
            weather = self.get_weather(city_name)
            print(f"Weather in {weather['city']}, {weather['country']}:")
            print(f"  Temperature: {weather['temperature']}°C")
            print(f"  Feels like: {weather['feels_like']}°C")
            print(f"  Conditions: {weather['description'].title()}")
            print(f"  Humidity: {weather['humidity']}%")
            print(f"  Pressure: {weather['pressure']} hPa")
            print(f"  Wind Speed: {weather['wind_speed']} m/s")
            print(f"  Last Updated: {weather['timestamp']}")
            return weather
        except Exception as e:
            print(f"Error: {e}")
            return None

def main():
    fetcher = WeatherFetcher()
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\n{'='*40}")
        fetcher.display_weather(city)

if __name__ == "__main__":
    main()