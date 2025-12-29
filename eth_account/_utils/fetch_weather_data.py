
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
import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_weather(self, city_name):
        try:
            params = {
                'q': city_name,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather_data(data)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {city_name}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response for {city_name}: {e}")
            return None
        except KeyError as e:
            self.logger.error(f"Unexpected data structure for {city_name}: {e}")
            return None

    def _parse_weather_data(self, data):
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
        }
        return weather_info

    def save_to_file(self, weather_data, filename):
        if weather_data:
            try:
                with open(filename, 'w') as f:
                    json.dump(weather_data, f, indent=2)
                self.logger.info(f"Weather data saved to {filename}")
                return True
            except IOError as e:
                self.logger.error(f"Failed to save to {filename}: {e}")
                return False
        return False

def main():
    API_KEY = "your_api_key_here"
    fetcher = WeatherFetcher(API_KEY)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather(city)
        if weather:
            print(f"Temperature in {weather['city']}, {weather['country']}: {weather['temperature']}°C")
            print(f"Weather: {weather['weather']} ({weather['description']})")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
            
            filename = f"weather_{city.lower().replace(' ', '_')}.json"
            fetcher.save_to_file(weather, filename)
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()