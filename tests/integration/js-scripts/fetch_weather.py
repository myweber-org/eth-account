
import requests
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city_name, country_code=''):
        if not self.api_key:
            raise ValueError("API key not provided")
        
        query = f"{city_name},{country_code}" if country_code else city_name
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def display_weather(self, weather_data):
        if not weather_data:
            print("No weather data available")
            return
        
        print(f"Weather in {weather_data['city']}, {weather_data['country']}:")
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Feels like: {weather_data['feels_like']}°C")
        print(f"Conditions: {weather_data['description'].title()}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"Last updated: {weather_data['timestamp']}")

def main():
    fetcher = WeatherFetcher()
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather = fetcher.get_weather(city)
        if weather:
            fetcher.display_weather(weather)
        else:
            print(f"Failed to fetch weather for {city}")

if __name__ == "__main__":
    main()