import requests
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, cache_duration: int = 300):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_duration = cache_duration
        self.cache: Dict[str, Dict[str, Any]] = {}
        
    def get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        cache_key = city.lower()
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                print(f"Returning cached data for {city}")
                return cached_data['data']
        
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.cache[cache_key] = {
                'data': processed_data,
                'timestamp': time.time()
            }
            
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def clear_cache(self):
        self.cache.clear()
        print("Weather cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            'cache_size': len(self.cache),
            'cached_cities': list(self.cache.keys())
        }

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}:")
        weather = fetcher.get_weather(city)
        if weather:
            for key, value in weather.items():
                print(f"  {key}: {value}")
    
    print(f"\nCache stats: {fetcher.get_cache_stats()}")

if __name__ == "__main__":
    main()
import requests
import sys

def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data is None:
        print("No weather data to display.")
        return
    try:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description}")
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)