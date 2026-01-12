
import requests
import json
from datetime import datetime
import sys

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
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
        except KeyError as e:
            return f"Unexpected API response format: {str(e)}"
    
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
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info
    
    def display_weather(self, weather_data):
        if isinstance(weather_data, dict):
            print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Feels like: {weather_data['feels_like']}°C")
            print(f"Conditions: {weather_data['weather'].title()}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Pressure: {weather_data['pressure']} hPa")
            print(f"Wind Speed: {weather_data['wind_speed']} m/s")
            print(f"Last updated: {weather_data['timestamp']}")
        else:
            print(weather_data)

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_weather_data.py <city_name>")
        sys.exit(1)
    
    api_key = "your_api_key_here"
    city_name = ' '.join(sys.argv[1:])
    
    fetcher = WeatherFetcher(api_key)
    weather_data = fetcher.get_weather(city_name)
    fetcher.display_weather(weather_data)

if __name__ == "__main__":
    main()
import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url

    def get_weather_by_city(self, city_name, units="metric"):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return f"Error fetching weather data: {e}"

    def _parse_weather_data(self, data):
        if data.get('cod') != 200:
            return f"API Error: {data.get('message', 'Unknown error')}"

        main_info = data.get('main', {})
        weather_info = data.get('weather', [{}])[0]
        wind_info = data.get('wind', {})

        parsed_data = {
            'city': data.get('name'),
            'country': data.get('sys', {}).get('country'),
            'temperature': main_info.get('temp'),
            'feels_like': main_info.get('feels_like'),
            'humidity': main_info.get('humidity'),
            'pressure': main_info.get('pressure'),
            'weather': weather_info.get('description'),
            'wind_speed': wind_info.get('speed'),
            'wind_direction': wind_info.get('deg'),
            'timestamp': datetime.fromtimestamp(data.get('dt')).isoformat(),
            'sunrise': datetime.fromtimestamp(data.get('sys', {}).get('sunrise')).time().isoformat(),
            'sunset': datetime.fromtimestamp(data.get('sys', {}).get('sunset')).time().isoformat()
        }
        return parsed_data

    def save_to_file(self, data, filename="weather_data.json"):
        if isinstance(data, dict):
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return f"Weather data saved to {filename}"
        return "Invalid data format, cannot save to file"

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.get_weather_by_city(city)
        
        if isinstance(weather_data, dict):
            print(f"Temperature in {city}: {weather_data['temperature']}°C")
            print(f"Weather: {weather_data['weather']}")
            print(f"Humidity: {weather_data['humidity']}%")
            
            if city == "London":
                fetcher.save_to_file(weather_data, f"{city.lower()}_weather.json")
        else:
            print(weather_data)

if __name__ == "__main__":
    main()