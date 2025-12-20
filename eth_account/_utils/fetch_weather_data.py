
import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        
    def get_current_weather(self, city_name, country_code=None):
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            processed_data = self._process_weather_data(data)
            logging.info(f"Weather data fetched for {query}")
            return processed_data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch weather data: {e}")
            return None
    
    def _process_weather_data(self, raw_data):
        return {
            'location': raw_data.get('name'),
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(raw_data['dt']).isoformat(),
            'sunrise': datetime.fromtimestamp(raw_data['sys']['sunrise']).isoformat(),
            'sunset': datetime.fromtimestamp(raw_date['sys']['sunset']).isoformat()
        }
    
    def save_to_file(self, data, filename="weather_data.json"):
        if data:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Weather data saved to {filename}")
            return True
        return False

def main():
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    all_weather_data = []
    
    for city, country in cities:
        weather_data = fetcher.get_current_weather(city, country)
        if weather_data:
            all_weather_data.append(weather_data)
            print(f"Current weather in {city}: {weather_data['temperature']}°C, {weather_data['weather']}")
    
    if all_weather_data:
        fetcher.save_to_file(all_weather_data, "multi_city_weather.json")

if __name__ == "__main__":
    main()