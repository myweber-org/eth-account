import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_current_weather(self, city_name, country_code=None):
        location = f"{city_name},{country_code}" if country_code else city_name
        endpoint = f"{self.base_url}/weather"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                logging.error(f"API Error: {data.get('message', 'Unknown error')}")
                return None
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}")
            return None

    def _parse_weather_data(self, raw_data):
        parsed = {
            'location': raw_data.get('name'),
            'country': raw_data.get('sys', {}).get('country'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather_description': raw_data.get('weather', [{}])[0].get('description'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'visibility': raw_data.get('visibility'),
            'sunrise': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunrise')).isoformat() if raw_data.get('sys', {}).get('sunrise') else None,
            'sunset': datetime.fromtimestamp(raw_data.get('sys', {}).get('sunset')).isoformat() if raw_data.get('sys', {}).get('sunset') else None,
            'timestamp': datetime.now().isoformat()
        }
        return parsed

    def save_to_file(self, data, filename="weather_data.json"):
        if not data:
            logging.warning("No data to save")
            return False
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            logging.info(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            logging.error(f"Failed to save data to file: {e}")
            return False

def main():
    API_KEY = "your_api_key_here"  # Replace with actual API key
    fetcher = WeatherFetcher(API_KEY)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    all_weather_data = []
    
    for city, country in cities:
        logging.info(f"Fetching weather for {city}, {country}")
        weather_data = fetcher.get_current_weather(city, country)
        
        if weather_data:
            all_weather_data.append(weather_data)
            print(f"Weather in {weather_data['location']}, {weather_data['country']}:")
            print(f"  Temperature: {weather_data['temperature']}°C")
            print(f"  Conditions: {weather_data['weather_description']}")
            print(f"  Humidity: {weather_data['humidity']}%")
            print("-" * 40)
    
    if all_weather_data:
        fetcher.save_to_file(all_weather_data, "multi_city_weather.json")

if __name__ == "__main__":
    main()