import requests
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherFetcher:
    """A class to fetch weather data from OpenWeatherMap API."""
    
    BASE_URL = "http://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key):
        """Initialize the fetcher with an API key."""
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_weather_by_city(self, city_name, units='metric'):
        """Fetch weather data for a given city."""
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        
        try:
            logger.info(f"Fetching weather data for city: {city_name}")
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant information
            weather_info = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
            logger.info(f"Successfully fetched weather data for {city_name}")
            return weather_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while fetching weather data: {e}")
            return None
        except KeyError as e:
            logger.error(f"Unexpected data structure in API response: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            return None
    
    def format_weather_output(self, weather_data):
        """Format weather data into a readable string."""
        if not weather_data:
            return "No weather data available."
        
        return f"""
Weather Report for {weather_data['city']}, {weather_data['country']}
--------------------------------------------------
Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)
Conditions: {weather_data['weather'].title()}
Humidity: {weather_data['humidity']}%
Pressure: {weather_data['pressure']} hPa
Wind Speed: {weather_data['wind_speed']} m/s
Report Time: {weather_data['timestamp']}
"""

def main():
    """Main function to demonstrate weather fetching."""
    # Note: In production, use environment variables or config files for API keys
    API_KEY = "your_api_key_here"  # Replace with actual API key
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Example cities to fetch weather for
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        weather_data = fetcher.get_weather_by_city(city)
        
        if weather_data:
            print(fetcher.format_weather_output(weather_data))
        else:
            print(f"Failed to fetch weather data for {city}")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
import requests
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
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed']

        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Conditions: {weather_desc}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    else:
        print("City not found or invalid API response.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)