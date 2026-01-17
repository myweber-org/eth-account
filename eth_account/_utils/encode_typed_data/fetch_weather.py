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
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description.capitalize()}")
    except KeyError as e:
        print(f"Unexpected data structure: missing key {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <API_KEY> <CITY>")
        sys.exit(1)
    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_data = get_weather(api_key, city)
    display_weather(weather_data)
import requests
import json
import time
from datetime import datetime, timedelta
import os

CACHE_FILE = "weather_cache.json"
CACHE_DURATION = 300  # 5 minutes in seconds

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                if time.time() - data.get('timestamp', 0) < CACHE_DURATION:
                    return data.get('weather_data')
        except (json.JSONDecodeError, KeyError):
            pass
    return None

def save_cache(weather_data):
    cache_data = {
        'timestamp': time.time(),
        'weather_data': weather_data
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f)

def fetch_weather_data(city, api_key):
    cached_data = load_cache()
    if cached_data and cached_data.get('city') == city:
        print(f"Using cached data for {city}")
        return cached_data
    
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        weather_info = {
            'city': city,
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'timestamp': datetime.now().isoformat()
        }
        
        save_cache(weather_info)
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing weather data: {e}")
        return None

def display_weather(weather_data):
    if not weather_data:
        print("No weather data available")
        return
    
    print("\n" + "="*40)
    print(f"Weather Report for {weather_data['city']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")
    print(f"Last Updated: {weather_data['timestamp']}")
    print("="*40)

def main():
    api_key = os.environ.get('WEATHER_API_KEY')
    if not api_key:
        print("Please set WEATHER_API_KEY environment variable")
        return
    
    city = input("Enter city name: ").strip()
    if not city:
        city = "London"
    
    print(f"\nFetching weather data for {city}...")
    weather_data = fetch_weather_data(city, api_key)
    display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
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
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"  Temperature: {temp}°C")
        print(f"  Humidity: {humidity}%")
        print(f"  Conditions: {description}")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather.py <API_KEY> <CITY_NAME>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city_name = sys.argv[2]
    weather_data = get_weather(api_key, city_name)
    display_weather(weather_data)