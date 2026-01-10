
import requests
import json
import time
from datetime import datetime, timedelta
import hashlib
import os

class WeatherFetcher:
    def __init__(self, api_key, cache_dir='./weather_cache'):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = cache_dir
        self.cache_duration = 3600
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _get_cache_key(self, city, country_code):
        data_string = f"{city.lower()}_{country_code.lower()}"
        return hashlib.md5(data_string.encode()).hexdigest()
    
    def _read_from_cache(self, cache_key):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cache_time < timedelta(seconds=self.cache_duration):
                    return cache_data['data']
        return None
    
    def _write_to_cache(self, cache_key, data):
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    def fetch_weather(self, city, country_code='us'):
        cache_key = self._get_cache_key(city, country_code)
        cached_data = self._read_from_cache(cache_key)
        
        if cached_data:
            print(f"Returning cached data for {city}, {country_code}")
            return cached_data
        
        params = {
            'q': f"{city},{country_code}",
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            weather_data = response.json()
            
            processed_data = {
                'city': weather_data['name'],
                'country': weather_data['sys']['country'],
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'pressure': weather_data['main']['pressure'],
                'description': weather_data['weather'][0]['description'],
                'wind_speed': weather_data['wind']['speed'],
                'timestamp': datetime.now().isoformat()
            }
            
            self._write_to_cache(cache_key, processed_data)
            print(f"Fetched fresh data for {city}, {country_code}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_temperature_trend(self, city_list, country_code='us'):
        temperatures = []
        for city in city_list:
            data = self.fetch_weather(city, country_code)
            if data:
                temperatures.append({
                    'city': city,
                    'temp': data['temperature'],
                    'timestamp': data['timestamp']
                })
            time.sleep(0.5)
        
        if temperatures:
            avg_temp = sum(t['temp'] for t in temperatures) / len(temperatures)
            hottest = max(temperatures, key=lambda x: x['temp'])
            coldest = min(temperatures, key=lambda x: x['temp'])
            
            return {
                'average_temperature': round(avg_temp, 2),
                'hottest_city': hottest['city'],
                'hottest_temp': hottest['temp'],
                'coldest_city': coldest['city'],
                'coldest_temp': coldest['temp'],
                'samples': len(temperatures)
            }
        return None

def main():
    api_key = os.environ.get('WEATHER_API_KEY', 'your_api_key_here')
    fetcher = WeatherFetcher(api_key)
    
    cities = ['London', 'New York', 'Tokyo', 'Sydney', 'Berlin']
    
    print("Fetching weather data for multiple cities...")
    trend = fetcher.get_temperature_trend(cities)
    
    if trend:
        print(f"\nWeather Analysis:")
        print(f"Average Temperature: {trend['average_temperature']}°C")
        print(f"Hottest: {trend['hottest_city']} ({trend['hottest_temp']}°C)")
        print(f"Coldest: {trend['coldest_city']} ({trend['coldest_temp']}°C)")
        print(f"Total cities analyzed: {trend['samples']}")
    
    single_city_data = fetcher.fetch_weather('Paris', 'fr')
    if single_city_data:
        print(f"\nSingle City Data (Paris, FR):")
        print(f"Temperature: {single_city_data['temperature']}°C")
        print(f"Humidity: {single_city_data['humidity']}%")
        print(f"Weather: {single_city_data['description']}")

if __name__ == "__main__":
    main()