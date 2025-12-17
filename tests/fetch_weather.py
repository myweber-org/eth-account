
import requests
import json
import os
from datetime import datetime

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(complete_url)
        response.raise_for_status()
        data = response.json()
        
        if data["cod"] != 200:
            print(f"Error: {data.get('message', 'Unknown error')}")
            return None
            
        main_data = data["main"]
        weather_data = data["weather"][0]
        sys_data = data["sys"]
        
        weather_info = {
            "city": data["name"],
            "country": sys_data["country"],
            "temperature": main_data["temp"],
            "feels_like": main_data["feels_like"],
            "humidity": main_data["humidity"],
            "pressure": main_data["pressure"],
            "weather": weather_data["main"],
            "description": weather_data["description"],
            "timestamp": datetime.fromtimestamp(data["dt"]).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Data parsing error: {e}")
        return None

def display_weather(weather_info):
    if not weather_info:
        return
        
    print("\n" + "="*40)
    print(f"Weather in {weather_info['city']}, {weather_info['country']}")
    print("="*40)
    print(f"Current Time: {weather_info['timestamp']}")
    print(f"Temperature: {weather_info['temperature']}°C")
    print(f"Feels Like: {weather_info['feels_like']}°C")
    print(f"Weather: {weather_info['weather']} ({weather_info['description']})")
    print(f"Humidity: {weather_info['humidity']}%")
    print(f"Pressure: {weather_info['pressure']} hPa")
    print("="*40)

def main():
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    
    if not api_key:
        print("Error: OPENWEATHER_API_KEY environment variable not set.")
        print("Please set your API key: export OPENWEATHER_API_KEY='your_api_key'")
        return
    
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty.")
        return
    
    print(f"Fetching weather data for {city}...")
    weather_data = get_weather(city, api_key)
    
    if weather_data:
        display_weather(weather_data)
        
        save_option = input("\nSave to file? (y/n): ").strip().lower()
        if save_option == 'y':
            filename = f"weather_{city.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(weather_data, f, indent=2)
            print(f"Weather data saved to {filename}")

if __name__ == "__main__":
    main()