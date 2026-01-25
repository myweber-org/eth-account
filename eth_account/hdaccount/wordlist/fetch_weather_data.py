
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
        
        if data['cod'] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid response from server")
        return None

def display_weather(weather_data):
    if not weather_data:
        return
        
    main = weather_data['main']
    weather = weather_data['weather'][0]
    wind = weather_data['wind']
    
    print(f"\nWeather in {weather_data['name']}, {weather_data['sys']['country']}:")
    print(f"  Temperature: {main['temp']}°C (Feels like: {main['feels_like']}°C)")
    print(f"  Conditions: {weather['description'].title()}")
    print(f"  Humidity: {main['humidity']}%")
    print(f"  Pressure: {main['pressure']} hPa")
    print(f"  Wind: {wind['speed']} m/s, Direction: {wind.get('deg', 'N/A')}°")
    print(f"  Cloudiness: {weather_data['clouds']['all']}%")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <api_key> <city>")
        print("Example: python fetch_weather_data.py your_api_key \"New York\"")
        sys.exit(1)
    
    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(api_key, city)
    
    if weather_data:
        display_weather(weather_data)

if __name__ == "__main__":
    main()