import requests
import json
from datetime import datetime

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
            data = response.json()
            
            if data.get('cod') != 200:
                return self._format_error(data)
                
            return self._format_response(data)
            
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except json.JSONDecodeError:
            return "Invalid response from server"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def _format_response(self, data):
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        
        return {
            'city': data.get('name'),
            'country': data.get('sys', {}).get('country'),
            'temperature': main.get('temp'),
            'feels_like': main.get('feels_like'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'description': weather.get('description'),
            'wind_speed': data.get('wind', {}).get('speed'),
            'timestamp': datetime.fromtimestamp(data.get('dt')).strftime('%Y-%m-%d %H:%M:%S')
        }

    def _format_error(self, data):
        error_msg = data.get('message', 'Unknown error')
        return f"Weather API error: {error_msg}"

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        result = fetcher.get_weather(city)
        
        if isinstance(result, dict):
            print(f"City: {result['city']}, {result['country']}")
            print(f"Temperature: {result['temperature']}°C (Feels like: {result['feels_like']}°C)")
            print(f"Conditions: {result['description']}")
            print(f"Humidity: {result['humidity']}% | Pressure: {result['pressure']} hPa")
            print(f"Wind Speed: {result['wind_speed']} m/s")
            print(f"Last updated: {result['timestamp']}")
        else:
            print(f"Error: {result}")

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
                self.logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return None
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            return None

    def _parse_weather_data(self, raw_data):
        parsed = {
            'timestamp': datetime.fromtimestamp(raw_data['dt']).isoformat(),
            'location': raw_data['name'],
            'country': raw_data['sys']['country'],
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['main'],
            'description': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'wind_direction': raw_data['wind'].get('deg', 'N/A'),
            'visibility': raw_data.get('visibility', 'N/A'),
            'cloudiness': raw_data['clouds']['all']
        }
        return parsed

    def format_weather_report(self, weather_data):
        if not weather_data:
            return "Weather data unavailable"
        
        report_lines = [
            f"Weather Report for {weather_data['location']}, {weather_data['country']}",
            f"Time: {weather_data['timestamp']}",
            f"Conditions: {weather_data['weather']} - {weather_data['description']}",
            f"Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)",
            f"Humidity: {weather_data['humidity']}%",
            f"Pressure: {weather_data['pressure']} hPa",
            f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°",
            f"Visibility: {weather_data['visibility']} meters",
            f"Cloud cover: {weather_data['cloudiness']}%"
        ]
        
        return '\n'.join(report_lines)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    for city, country in cities:
        print(f"\nFetching weather for {city}, {country}...")
        weather = fetcher.get_current_weather(city, country)
        
        if weather:
            report = fetcher.format_weather_report(weather)
            print(report)
        else:
            print(f"Failed to retrieve weather data for {city}")

if __name__ == "__main__":
    main()