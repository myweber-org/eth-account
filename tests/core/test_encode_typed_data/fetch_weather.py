
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

class WeatherFetcher:
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        query = city
        if country_code:
            query += f",{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'success': True,
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'raw_data': data
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Network error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            return {
                'success': False,
                'error': f"Data parsing error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }
    
    def format_weather_report(self, weather_data: Dict[str, Any]) -> str:
        if not weather_data['success']:
            return f"Weather data unavailable: {weather_data['error']}"
        
        report_lines = [
            f"Weather Report for {weather_data['city']}, {weather_data['country']}",
            f"Time: {weather_data['timestamp']}",
            f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)",
            f"Conditions: {weather_data['description'].title()}",
            f"Humidity: {weather_data['humidity']}%",
            f"Pressure: {weather_data['pressure']} hPa",
            f"Wind Speed: {weather_data['wind_speed']} m/s"
        ]
        
        return '\n'.join(report_lines)

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = [
        ("London", "GB"),
        ("New York", "US"),
        ("Tokyo", "JP")
    ]
    
    for city, country in cities:
        print(f"\nFetching weather for {city}, {country}...")
        weather = fetcher.get_weather(city, country)
        report = fetcher.format_weather_report(weather)
        print(report)

if __name__ == "__main__":
    main()