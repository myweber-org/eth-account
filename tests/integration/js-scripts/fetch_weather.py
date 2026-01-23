
import requests
import os
from datetime import datetime

def get_weather(city_name, api_key):
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
        
        if data['cod'] != 200:
            return f"Error: {data.get('message', 'Unknown error')}"
        
        main = data['main']
        weather = data['weather'][0]
        wind = data['wind']
        
        result = f"""
Weather in {data['name']}, {data['sys']['country']}:
----------------------------------------
Temperature: {main['temp']}°C (Feels like: {main['feels_like']}°C)
Condition: {weather['main']} - {weather['description']}
Humidity: {main['humidity']}%
Pressure: {main['pressure']} hPa
Wind Speed: {wind['speed']} m/s
Visibility: {data.get('visibility', 'N/A')} meters
Sunrise: {datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S')}
Sunset: {datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S')}
----------------------------------------
Last updated: {datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')}
"""
        return result
        
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"Data parsing error: {str(e)}"

def main():
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("Error: OPENWEATHER_API_KEY environment variable not set")
        print("Get your API key from: https://openweathermap.org/api")
        return
    
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty")
        return
    
    weather_info = get_weather(city, api_key)
    print(weather_info)

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
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data is None:
        print("No data to display.")
        return
    try:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        wind_speed = data['wind']['speed']

        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description}")
        print(f"  Wind Speed: {wind_speed} m/s")
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city):
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.RequestException as e:
            return {'error': f'Network error: {str(e)}'}
        except json.JSONDecodeError:
            return {'error': 'Invalid response from server'}
        except KeyError as e:
            return {'error': f'Unexpected data format: {str(e)}'}
    
    def _parse_response(self, data):
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
    
    def display_weather(self, weather_data):
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
            return
        
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
        print("="*40)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather.py <city_name>")
        print("Example: python fetch_weather.py London")
        sys.exit(1)
    
    city = ' '.join(sys.argv[1:])
    
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    print(f"Fetching weather data for {city}...")
    weather_data = fetcher.get_weather(city)
    fetcher.display_weather(weather_data)

if __name__ == "__main__":
    main()