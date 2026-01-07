
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class WeatherFetcher:
    def __init__(self, api_key, cache_dir="weather_cache"):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = timedelta(hours=1)

    def _get_cache_path(self, city):
        return self.cache_dir / f"{city.lower().replace(' ', '_')}.json"

    def _is_cache_valid(self, cache_path):
        if not cache_path.exists():
            return False
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - cache_time < self.cache_duration

    def fetch_weather(self, city):
        cache_path = self._get_cache_path(city)
        
        if self._is_cache_valid(cache_path):
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            cached_data['source'] = 'cache'
            return cached_data

        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            
            processed_data = {
                'city': city,
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'description': weather_data['weather'][0]['description'],
                'timestamp': datetime.now().isoformat(),
                'source': 'api'
            }
            
            with open(cache_path, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None

    def get_multiple_cities(self, cities):
        results = {}
        for city in cities:
            weather = self.fetch_weather(city)
            if weather:
                results[city] = weather
            time.sleep(0.5)
        return results

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    print("Fetching weather data...")
    weather_results = fetcher.get_multiple_cities(cities)
    
    for city, data in weather_results.items():
        if data:
            print(f"\n{city}:")
            print(f"  Temperature: {data['temperature']}°C")
            print(f"  Humidity: {data['humidity']}%")
            print(f"  Conditions: {data['description']}")
            print(f"  Source: {data['source']}")

if __name__ == "__main__":
    main()import requests
import json
import os

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        print("No weather data available.")
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
    print("="*40)

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Please set OPENWEATHER_API_KEY environment variable")
        return
    
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty.")
        return
    
    weather = get_weather(city, api_key)
    display_weather(weather)

if __name__ == "__main__":
    main()