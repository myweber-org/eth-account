
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
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def save_to_file(self, weather_data, filename='weather_data.json'):
        if weather_data:
            try:
                with open(filename, 'w') as f:
                    json.dump(weather_data, f, indent=4)
                print(f"Weather data saved to {filename}")
            except IOError as e:
                print(f"Error saving to file: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather(city)
        
        if weather:
            print(f"City: {weather['city']}")
            print(f"Temperature: {weather['temperature']}°C")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Conditions: {weather['description']}")
            print(f"Wind Speed: {weather['wind_speed']} m/s")
            print(f"Last Updated: {weather['timestamp']}")
            
            filename = f"{city.lower().replace(' ', '_')}_weather.json"
            fetcher.save_to_file(weather, filename)

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime

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
        
        weather_info = {
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
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except KeyError as e:
        print(f"Unexpected API response format: {e}")
        return None

def display_weather(weather_data):
    if weather_data:
        print("\n" + "="*40)
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print("="*40)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"Last Updated: {weather_data['timestamp']}")
        print("="*40 + "\n")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather(API_KEY, CITY)
    display_weather(weather)
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch weather data from a public API."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        """Initialize the weather fetcher with an API key."""
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_weather_by_city(self, city_name: str, country_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch weather data for a given city."""
        query = city_name
        if country_code:
            query = f"{city_name},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_weather_by_coordinates(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Fetch weather data using latitude and longitude coordinates."""
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def _parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure the raw weather API response."""
        parsed_data = {
            'location': raw_data.get('name', 'Unknown'),
            'country': raw_data.get('sys', {}).get('country', 'Unknown'),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'weather_description': raw_data.get('weather', [{}])[0].get('description', 'Unknown'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_direction': raw_data.get('wind', {}).get('deg'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'visibility': raw_data.get('visibility'),
            'sunrise': self._convert_timestamp(raw_data.get('sys', {}).get('sunrise')),
            'sunset': self._convert_timestamp(raw_data.get('sys', {}).get('sunset')),
            'data_timestamp': self._convert_timestamp(raw_data.get('dt')),
            'raw_data': raw_data
        }
        return parsed_data
    
    def _convert_timestamp(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert Unix timestamp to readable datetime string."""
        if timestamp:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return None
    
    def display_weather(self, weather_data: Dict[str, Any]) -> None:
        """Display weather information in a readable format."""
        if not weather_data:
            print("No weather data available.")
            return
        
        print("\n" + "="*50)
        print(f"Weather in {weather_data['location']}, {weather_data['country']}")
        print("="*50)
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Weather: {weather_data['weather_description'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print(f"Data collected at: {weather_data['data_timestamp']}")
        print("="*50 + "\n")

def main():
    """Example usage of the WeatherFetcher class."""
    api_key = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(api_key)
    
    # Example: Get weather by city
    print("Fetching weather for London, UK...")
    london_weather = fetcher.get_weather_by_city("London", "UK")
    fetcher.display_weather(london_weather)
    
    # Example: Get weather by coordinates (New York)
    print("Fetching weather for New York coordinates...")
    ny_weather = fetcher.get_weather_by_coordinates(40.7128, -74.0060)
    fetcher.display_weather(ny_weather)
    
    # Save weather data to JSON file
    if london_weather:
        with open('london_weather.json', 'w') as f:
            json.dump(london_weather, f, indent=2)
        print("Weather data saved to 'london_weather.json'")

if __name__ == "__main__":
    main()