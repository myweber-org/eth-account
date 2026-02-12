
import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()
        
    def get_weather(self, city_name, units='metric'):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None
    
    def _parse_weather_data(self, data):
        weather_info = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
        }
        return weather_info
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available")
            return
        
        print("\n" + "="*50)
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print("="*50)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Last updated: {weather_data['timestamp']}")
        print("="*50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        print("Example: python fetch_weather_data.py London")
        sys.exit(1)
    
    city_name = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    
    if api_key == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key")
        print("Get a free API key at: https://openweathermap.org/api")
        sys.exit(1)
    
    fetcher = WeatherFetcher(api_key)
    weather_data = fetcher.get_weather(city_name)
    
    if weather_data:
        fetcher.display_weather(weather_data)
        
        with open(f"weather_{city_name.replace(' ', '_')}.json", 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"\nWeather data saved to weather_{city_name.replace(' ', '_')}.json")
    else:
        print(f"Could not fetch weather data for {city_name}")

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime, timedelta
import os

class WeatherFetcher:
    def __init__(self, api_key, cache_dir='./weather_cache'):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_path(self, city):
        safe_city = "".join(c for c in city if c.isalnum())
        return os.path.join(self.cache_dir, f"{safe_city}.json")

    def _is_cache_valid(self, cache_path):
        if not os.path.exists(cache_path):
            return False
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time < timedelta(minutes=30)

    def fetch_weather(self, city):
        cache_path = self._get_cache_path(city)
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                    print(f"Using cached data for {city}")
                    return cached_data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Cache read error: {e}")

        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('cod') != 200:
                raise ValueError(f"API error: {data.get('message', 'Unknown error')}")

            processed_data = {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.now().isoformat()
            }

            try:
                with open(cache_path, 'w') as f:
                    json.dump(processed_data, f, indent=2)
            except IOError as e:
                print(f"Cache write error: {e}")

            return processed_data

        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Data processing error: {e}")
            return None

def main():
    api_key = os.environ.get('WEATHER_API_KEY')
    if not api_key:
        print("Error: WEATHER_API_KEY environment variable not set")
        return

    fetcher = WeatherFetcher(api_key)
    cities = ['London', 'Tokyo', 'New York']

    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.fetch_weather(city)
        if weather:
            print(f"Temperature: {weather['temperature']}°C")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Conditions: {weather['description']}")
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime

def get_weather_data(api_key, city_name):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
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
        
        if data.get('cod') != 200:
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
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_data):
    """
    Display weather information in a readable format.
    """
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*40)
    print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Time: {weather_data['timestamp']}")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Weather: {weather_data['weather'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print("="*40)

def main():
    # Replace with your actual OpenWeatherMap API key
    API_KEY = "your_api_key_here"
    
    if API_KEY == "your_api_key_here":
        print("Please replace 'your_api_key_here' with your actual OpenWeatherMap API key.")
        print("You can get a free API key from: https://openweathermap.org/api")
        return
    
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty.")
        return
    
    print(f"\nFetching weather data for {city}...")
    weather_data = get_weather_data(API_KEY, city)
    
    if weather_data:
        display_weather(weather_data)
        
        # Optional: Save to JSON file
        save_option = input("\nSave to JSON file? (y/n): ").lower()
        if save_option == 'y':
            filename = f"weather_{weather_data['city'].lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(weather_data, f, indent=2)
            print(f"Weather data saved to {filename}")

if __name__ == "__main__":
    main()