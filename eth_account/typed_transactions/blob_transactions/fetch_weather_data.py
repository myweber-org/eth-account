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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if weather_data and weather_data.get('cod') == 200:
        main = weather_data['main']
        weather = weather_data['weather'][0]
        print(f"City: {weather_data['name']}")
        print(f"Temperature: {main['temp']}°C")
        print(f"Humidity: {main['humidity']}%")
        print(f"Weather: {weather['description']}")
        print(f"Pressure: {main['pressure']} hPa")
    else:
        message = weather_data.get('message', 'Unknown error') if weather_data else 'No data received'
        print(f"Failed to get weather data: {message}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = sys.argv[2]
    
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather_by_city(self, city_name):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_weather_data(data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None

    def _parse_weather_data(self, data):
        parsed_data = {
            'city': data.get('name'),
            'country': data.get('sys', {}).get('country'),
            'temperature': data.get('main', {}).get('temp'),
            'feels_like': data.get('main', {}).get('feels_like'),
            'humidity': data.get('main', {}).get('humidity'),
            'pressure': data.get('main', {}).get('pressure'),
            'weather_description': data.get('weather', [{}])[0].get('description'),
            'wind_speed': data.get('wind', {}).get('speed'),
            'wind_direction': data.get('wind', {}).get('deg'),
            'visibility': data.get('visibility'),
            'cloudiness': data.get('clouds', {}).get('all'),
            'sunrise': self._convert_timestamp(data.get('sys', {}).get('sunrise')),
            'sunset': self._convert_timestamp(data.get('sys', {}).get('sunset')),
            'timestamp': datetime.now().isoformat()
        }
        return parsed_data

    def _convert_timestamp(self, timestamp):
        if timestamp:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        return None

    def save_to_file(self, data, filename='weather_data.json'):
        if data:
            try:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                print(f"Weather data saved to {filename}")
            except IOError as e:
                print(f"Error saving to file: {e}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.get_weather_by_city(city)
        
        if weather_data:
            print(f"Temperature in {weather_data['city']}: {weather_data['temperature']}°C")
            print(f"Weather: {weather_data['weather_description']}")
            print(f"Humidity: {weather_data['humidity']}%")
            
            filename = f"weather_{city.lower().replace(' ', '_')}.json"
            fetcher.save_to_file(weather_data, filename)
        else:
            print(f"Failed to fetch weather data for {city}")

if __name__ == "__main__":
    main()