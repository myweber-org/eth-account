
import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()
    
    def get_weather_by_city(self, city_name, units="metric"):
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            return None
    
    def _parse_weather_data(self, raw_data):
        if raw_data.get("cod") != 200:
            return None
        
        weather_info = {
            "city": raw_data.get("name"),
            "country": raw_data.get("sys", {}).get("country"),
            "temperature": raw_data.get("main", {}).get("temp"),
            "feels_like": raw_data.get("main", {}).get("feels_like"),
            "humidity": raw_data.get("main", {}).get("humidity"),
            "pressure": raw_data.get("main", {}).get("pressure"),
            "weather": raw_data.get("weather", [{}])[0].get("description"),
            "wind_speed": raw_data.get("wind", {}).get("speed"),
            "wind_direction": raw_data.get("wind", {}).get("deg"),
            "visibility": raw_data.get("visibility"),
            "cloudiness": raw_data.get("clouds", {}).get("all"),
            "sunrise": datetime.fromtimestamp(raw_data.get("sys", {}).get("sunrise")),
            "sunset": datetime.fromtimestamp(raw_data.get("sys", {}).get("sunset")),
            "timestamp": datetime.fromtimestamp(raw_data.get("dt"))
        }
        
        return weather_info
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available")
            return
        
        print(f"Weather in {weather_data['city']}, {weather_data['country']}")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Conditions: {weather_data['weather'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Sunrise: {weather_data['sunrise'].strftime('%H:%M')}")
        print(f"Sunset: {weather_data['sunset'].strftime('%H:%M')}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\n{'='*40}")
        weather = fetcher.get_weather_by_city(city)
        if weather:
            fetcher.display_weather(weather)
        else:
            print(f"Could not fetch weather for {city}")
        print(f"{'='*40}")

if __name__ == "__main__":
    main()