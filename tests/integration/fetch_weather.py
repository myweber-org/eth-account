import requests
import json
from datetime import datetime

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
            data = response.json()
            
            if data.get('cod') != 200:
                return self._format_error(data)
                
            return self._format_response(data)
            
        except requests.exceptions.RequestException as e:
            return f"Network error: {str(e)}"
        except json.JSONDecodeError:
            return "Invalid response from server"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def _format_response(self, data):
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        
        return {
            'city': data.get('name'),
            'country': data.get('sys', {}).get('country'),
            'temperature': main.get('temp'),
            'feels_like': main.get('feels_like'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'description': weather.get('description'),
            'wind_speed': data.get('wind', {}).get('speed'),
            'timestamp': datetime.fromtimestamp(data.get('dt')).strftime('%Y-%m-%d %H:%M:%S')
        }

    def _format_error(self, data):
        error_msg = data.get('message', 'Unknown error')
        return f"Weather API error: {error_msg}"

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "New York", "Tokyo", "Paris"]
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        result = fetcher.get_weather(city)
        
        if isinstance(result, dict):
            print(f"City: {result['city']}, {result['country']}")
            print(f"Temperature: {result['temperature']}°C (Feels like: {result['feels_like']}°C)")
            print(f"Conditions: {result['description']}")
            print(f"Humidity: {result['humidity']}% | Pressure: {result['pressure']} hPa")
            print(f"Wind Speed: {result['wind_speed']} m/s")
            print(f"Last updated: {result['timestamp']}")
        else:
            print(f"Error: {result}")

if __name__ == "__main__":
    main()