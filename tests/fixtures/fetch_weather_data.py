import requests
import json
from datetime import datetime

def get_weather_data(api_key, city, units='metric'):
    """
    Fetch current weather data for a given city using OpenWeatherMap API.
    
    Args:
        api_key (str): OpenWeatherMap API key
        city (str): City name
        units (str): Units of measurement ('metric', 'imperial', 'standard')
    
    Returns:
        dict: Weather data dictionary or None if request fails
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        'q': city,
        'appid': api_key,
        'units': units
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('cod') != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
        
        processed_data = {
            'city': data['name'],
            'country': data['sys']['country'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['main'],
            'description': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 'N/A'),
            'visibility': data.get('visibility', 'N/A'),
            'cloudiness': data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M:%S'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M:%S'),
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
            'units': '°C' if units == 'metric' else '°F'
        }
        
        return processed_data
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather_data(weather_data):
    """
    Display weather data in a formatted way.
    
    Args:
        weather_data (dict): Weather data dictionary
    """
    if not weather_data:
        print("No weather data available.")
        return
    
    print("\n" + "="*50)
    print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
    print("="*50)
    print(f"Current Time: {weather_data['timestamp']}")
    print(f"Weather: {weather_data['weather']} ({weather_data['description']})")
    print(f"Temperature: {weather_data['temperature']}{weather_data['units']}")
    print(f"Feels Like: {weather_data['feels_like']}{weather_data['units']}")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
    print(f"Visibility: {weather_data['visibility']} meters")
    print(f"Cloudiness: {weather_data['cloudiness']}%")
    print(f"Sunrise: {weather_data['sunrise']}")
    print(f"Sunset: {weather_data['sunset']}")
    print("="*50)

def save_weather_data(weather_data, filename='weather_data.json'):
    """
    Save weather data to a JSON file.
    
    Args:
        weather_data (dict): Weather data dictionary
        filename (str): Output filename
    """
    if not weather_data:
        print("No data to save.")
        return
    
    try:
        with open(filename, 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"Weather data saved to {filename}")
    except IOError as e:
        print(f"Failed to save data: {e}")

def main():
    """
    Main function to demonstrate weather data fetching.
    """
    api_key = "your_api_key_here"  # Replace with your actual API key
    city = "London"
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather_data(api_key, city)
    
    if weather_data:
        display_weather_data(weather_data)
        save_weather_data(weather_data)
        
        # Example of additional processing
        if weather_data['temperature'] > 25:
            print("It's a warm day!")
        elif weather_data['temperature'] < 10:
            print("It's a cold day!")
        else:
            print("Moderate temperature today.")
    else:
        print("Failed to fetch weather data.")

if __name__ == "__main__":
    main()