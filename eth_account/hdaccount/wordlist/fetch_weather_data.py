import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.session = requests.Session()

    def get_weather(self, city_name, units='metric'):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('cod') != 200:
                raise ValueError(f"API Error: {data.get('message', 'Unknown error')}")
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch weather data: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError("Invalid response from weather API")

    def _parse_weather_data(self, data):
        weather_info = {
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
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return weather_info

    def display_weather(self, weather_data):
        print(f"Weather Report for {weather_data['city']}, {weather_data['country']}")
        print(f"Timestamp: {weather_data['timestamp']}")
        print(f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)")
        print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Pressure: {weather_data['pressure']} hPa")
        print(f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°")
        print(f"Visibility: {weather_data['visibility']} meters")
        print(f"Cloudiness: {weather_data['cloudiness']}%")
        print(f"Sunrise: {weather_data['sunrise']}")
        print(f"Sunset: {weather_data['sunset']}")

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    try:
        city = input("Enter city name: ").strip()
        if not city:
            print("City name cannot be empty")
            return
            
        weather = fetcher.get_weather(city)
        fetcher.display_weather(weather)
        
    except (ValueError, ConnectionError) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")

if __name__ == "__main__":
    main()