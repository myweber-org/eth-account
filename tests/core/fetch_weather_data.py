
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
    main()