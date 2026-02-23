import requests
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
    except json.JSONDecodeError:
        print("Error decoding JSON response.")
        return None

def display_weather(data):
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        feels_like = data['main']['feels_like']
        humidity = data['main']['humidity']
        weather_desc = data['weather'][0]['description']
        wind_speed = data['wind']['speed']

        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (Feels like: {feels_like}°C)")
        print(f"  Conditions: {weather_desc.capitalize()}")
        print(f"  Humidity: {humidity}%")
        print(f"  Wind Speed: {wind_speed} m/s")
    else:
        error_msg = data.get('message', 'Unknown error') if data else 'No data received'
        print(f"Could not retrieve weather data. Error: {error_msg}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = sys.argv[2]

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    def get_weather_by_city(self, city_name, country_code=None):
        if not self.api_key:
            raise ValueError("API key not provided")
            
        query = city_name
        if country_code:
            query += f",{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def _parse_weather_data(self, data):
        if data.get('cod') != 200:
            return None
            
        return {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind'].get('deg', 0),
            'visibility': data.get('visibility', 0),
            'clouds': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available")
            return
            
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
        print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_deg']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Cloudiness: {weather_data['clouds']}%")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")
        print(f"Last updated: {weather_data['timestamp']}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\n{'='*50}")
        weather = fetcher.get_weather_by_city(city)
        if weather:
            fetcher.display_weather(weather)
        else:
            print(f"Could not fetch weather for {city}")
        print(f"{'='*50}")

if __name__ == "__main__":
    main()