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
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    def __init__(self, api_key: str, base_url: str = "http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_current_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        location = f"{city},{country_code}" if country_code else city
        params = {
            'q': location,
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
            
            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat()
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch weather data: {str(e)}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Invalid response format: {str(e)}")
    
    def get_forecast(self, city: str, days: int = 5) -> Dict[str, Any]:
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            forecast = []
            for item in data['list'][:days*8:8]:
                forecast.append({
                    'date': datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d'),
                    'temperature': item['main']['temp'],
                    'weather': item['weather'][0]['description'],
                    'humidity': item['main']['humidity']
                })
            
            return {
                'city': data['city']['name'],
                'country': data['city']['country'],
                'forecast': forecast
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch forecast: {str(e)}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Invalid forecast response: {str(e)}")

def save_to_json(data: Dict[str, Any], filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    try:
        current = fetcher.get_current_weather("London", "UK")
        print(f"Current weather in {current['city']}:")
        print(f"Temperature: {current['temperature']}°C")
        print(f"Weather: {current['weather']}")
        print(f"Humidity: {current['humidity']}%")
        
        save_to_json(current, "london_weather.json")
        
        forecast = fetcher.get_forecast("London", 3)
        print(f"\n3-day forecast for {forecast['city']}:")
        for day in forecast['forecast']:
            print(f"{day['date']}: {day['temperature']}°C, {day['weather']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()import requests
import json
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
        return
        
    print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
    print(f"  Temperature: {weather_data['temperature']}°C")
    print(f"  Feels like: {weather_data['feels_like']}°C")
    print(f"  Conditions: {weather_data['weather']}")
    print(f"  Humidity: {weather_data['humidity']}%")
    print(f"  Pressure: {weather_data['pressure']} hPa")
    print(f"  Wind Speed: {weather_data['wind_speed']} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather.py <api_key> <city>")
        print("Example: python fetch_weather.py abc123 London")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()