
import requests
import json
import time
from datetime import datetime, timedelta

class WeatherFetcher:
    def __init__(self, api_key, cache_duration=300):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_duration = cache_duration
        self.cache = {}

    def _get_cache_key(self, city):
        return city.lower()

    def _is_cache_valid(self, cache_entry):
        if not cache_entry:
            return False
        timestamp = cache_entry.get('timestamp', 0)
        return time.time() - timestamp < self.cache_duration

    def fetch_weather(self, city):
        cache_key = self._get_cache_key(city)
        
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            print(f"Returning cached data for {city}")
            return self.cache[cache_key]['data']

        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            weather_data = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed']
            }

            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': weather_data
            }

            return weather_data

        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing weather data: {e}")
            return None

    def get_cached_cities(self):
        return list(self.cache.keys())

    def clear_cache(self):
        self.cache.clear()
        print("Cache cleared")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key, cache_duration=300)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}:")
        weather = fetcher.fetch_weather(city)
        
        if weather:
            print(f"Temperature: {weather['temperature']}°C")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Conditions: {weather['description']}")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
        else:
            print("Failed to fetch weather data")
    
    print(f"\nCached cities: {fetcher.get_cached_cities()}")

if __name__ == "__main__":
    main()