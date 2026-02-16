import requests
import json
from datetime import datetime
import sys

def fetch_weather_data(api_key, city):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}", file=sys.stderr)
        return None

def parse_weather_data(weather_json):
    """
    Parse and extract relevant information from weather JSON response.
    """
    if not weather_json:
        return None
    
    try:
        main_data = weather_json['main']
        weather_info = weather_json['weather'][0]
        
        parsed_data = {
            'temperature': main_data['temp'],
            'feels_like': main_data['feels_like'],
            'humidity': main_data['humidity'],
            'pressure': main_data['pressure'],
            'description': weather_info['description'],
            'city': weather_json['name'],
            'country': weather_json['sys']['country'],
            'timestamp': datetime.fromtimestamp(weather_json['dt']).isoformat()
        }
        return parsed_data
    except (KeyError, IndexError) as e:
        print(f"Error parsing weather data: {e}", file=sys.stderr)
        return None

def display_weather_info(weather_data):
    """
    Display weather information in a readable format.
    """
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*40)
    print(f"Weather in {weather_data['city']}, {weather_data['country']}")
    print("="*40)
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Conditions: {weather_data['description'].title()}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Last updated: {weather_data['timestamp']}")
    print("="*40)

def save_weather_data(weather_data, filename='weather_log.json'):
    """
    Save weather data to a JSON file for logging purposes.
    """
    if not weather_data:
        return False
    
    try:
        with open(filename, 'a') as f:
            json.dump(weather_data, f)
            f.write('\n')
        return True
    except IOError as e:
        print(f"Error saving weather data: {e}", file=sys.stderr)
        return False

def main():
    """
    Main function to demonstrate weather data fetching functionality.
    """
    # In a real application, this should be stored securely
    API_KEY = "your_api_key_here"  # Replace with actual API key
    CITY = "London"
    
    print(f"Fetching weather data for {CITY}...")
    
    # Fetch raw weather data
    weather_json = fetch_weather_data(API_KEY, CITY)
    
    if weather_json:
        # Parse the data
        weather_data = parse_weather_data(weather_json)
        
        # Display the information
        display_weather_info(weather_data)
        
        # Save to log file
        if weather_data and save_weather_data(weather_data):
            print(f"Weather data saved to weather_log.json")
    else:
        print("Failed to retrieve weather data.")

if __name__ == "__main__":
    main()