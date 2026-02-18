import requests
import json

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
            return f"Error: {data.get('message', 'Unknown error')}"
        
        weather_info = {
            'city': data['name'],
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description'],
            'wind_speed': data['wind']['speed']
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except (KeyError, json.JSONDecodeError) as e:
        return f"Data parsing error: {str(e)}"

def display_weather(weather_data):
    if isinstance(weather_data, dict):
        print(f"Weather in {weather_data['city']}:")
        print(f"  Temperature: {weather_data['temperature']}°C")
        print(f"  Feels like: {weather_data['feels_like']}°C")
        print(f"  Humidity: {weather_data['humidity']}%")
        print(f"  Pressure: {weather_data['pressure']} hPa")
        print(f"  Conditions: {weather_data['weather']}")
        print(f"  Wind speed: {weather_data['wind_speed']} m/s")
    else:
        print(weather_data)

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    CITY = "London"
    
    weather = get_weather(API_KEY, CITY)
    display_weather(weather)
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
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(data):
    if data is None:
        print("No data to display.")
        return
    if data.get('cod') != 200:
        print(f"Error: {data.get('message', 'Unknown error')}")
        return

    city = data['name']
    country = data['sys']['country']
    temp = data['main']['temp']
    feels_like = data['main']['feels_like']
    humidity = data['main']['humidity']
    weather_desc = data['weather'][0]['description']
    wind_speed = data['wind']['speed']

    print(f"Weather in {city}, {country}:")
    print(f"  Temperature: {temp}°C (feels like {feels_like}°C)")
    print(f"  Conditions: {weather_desc}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        print("Example: python fetch_weather_data.py abc123 London")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()
import requests
import json
from datetime import datetime
import logging
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
    
    def get_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        query = city
        if country_code:
            query = f"{city},{country_code}"
        
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_weather_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching weather for {query}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response for {query}: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected API response structure for {query}: {e}")
            raise
    
    def _parse_weather_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'city': raw_data['name'],
            'country': raw_data['sys']['country'],
            'temperature': raw_data['main']['temp'],
            'feels_like': raw_data['main']['feels_like'],
            'humidity': raw_data['main']['humidity'],
            'pressure': raw_data['main']['pressure'],
            'weather': raw_data['weather'][0]['description'],
            'wind_speed': raw_data['wind']['speed'],
            'wind_direction': raw_data['wind'].get('deg', 0),
            'visibility': raw_data.get('visibility', 0),
            'cloudiness': raw_data['clouds']['all'],
            'sunrise': datetime.fromtimestamp(raw_data['sys']['sunrise']).isoformat(),
            'sunset': datetime.fromtimestamp(raw_data['sys']['sunset']).isoformat(),
            'timestamp': datetime.fromtimestamp(raw_data['dt']).isoformat(),
            'coordinates': {
                'lon': raw_data['coord']['lon'],
                'lat': raw_data['coord']['lat']
            }
        }
    
    def format_weather_report(self, weather_data: Dict[str, Any]) -> str:
        report_lines = [
            f"Weather Report for {weather_data['city']}, {weather_data['country']}",
            f"Timestamp: {weather_data['timestamp']}",
            f"Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)",
            f"Weather: {weather_data['weather'].title()}",
            f"Humidity: {weather_data['humidity']}%",
            f"Pressure: {weather_data['pressure']} hPa",
            f"Wind: {weather_data['wind_speed']} m/s at {weather_data['wind_direction']}°",
            f"Visibility: {weather_data['visibility']} meters",
            f"Cloudiness: {weather_data['cloudiness']}%",
            f"Sunrise: {weather_data['sunrise']}",
            f"Sunset: {weather_data['sunset']}",
            f"Coordinates: {weather_data['coordinates']['lat']:.4f}, {weather_data['coordinates']['lon']:.4f}"
        ]
        return "\n".join(report_lines)

def main():
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    cities = [
        ("London", "UK"),
        ("New York", "US"),
        ("Tokyo", "JP"),
        ("Sydney", "AU")
    ]
    
    for city, country in cities:
        try:
            print(f"\n{'='*50}")
            weather_data = fetcher.get_weather(city, country)
            report = fetcher.format_weather_report(weather_data)
            print(report)
            logger.info(f"Successfully fetched weather for {city}, {country}")
        except Exception as e:
            logger.error(f"Failed to fetch weather for {city}, {country}: {e}")
            continue

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime
import os
from typing import Optional, Dict, Any

class WeatherFetcher:
    """A class to fetch and process weather data from OpenWeatherMap API."""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the WeatherFetcher with an API key."""
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as OPENWEATHER_API_KEY environment variable")
    
    def get_current_weather(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """Fetch current weather data for a given city."""
        query = f"{city},{country_code}" if country_code else city
        params = {
            "q": query,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._process_current_weather(data)
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to fetch weather data: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Invalid response from weather API"}
    
    def get_weather_forecast(self, city: str, country_code: Optional[str] = None) -> Dict[str, Any]:
        """Fetch 5-day weather forecast for a given city."""
        query = f"{city},{country_code}" if country_code else city
        params = {
            "q": query,
            "appid": self.api_key,
            "units": "metric"
        }
        
        try:
            response = requests.get(self.FORECAST_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._process_forecast_data(data)
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to fetch forecast data: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Invalid response from forecast API"}
    
    def _process_current_weather(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw current weather data into a structured format."""
        if data.get("cod") != 200:
            return {"error": f"Weather API error: {data.get('message', 'Unknown error')}"}
        
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        sys = data.get("sys", {})
        
        processed_data = {
            "location": {
                "city": data.get("name"),
                "country": sys.get("country"),
                "coordinates": {
                    "latitude": data.get("coord", {}).get("lat"),
                    "longitude": data.get("coord", {}).get("lon")
                }
            },
            "temperature": {
                "current": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "min": main.get("temp_min"),
                "max": main.get("temp_max")
            },
            "conditions": {
                "main": weather.get("main"),
                "description": weather.get("description"),
                "icon": weather.get("icon")
            },
            "additional": {
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "wind_speed": wind.get("speed"),
                "wind_direction": wind.get("deg"),
                "cloudiness": data.get("clouds", {}).get("all"),
                "visibility": data.get("visibility"),
                "sunrise": self._timestamp_to_datetime(sys.get("sunrise")),
                "sunset": self._timestamp_to_datetime(sys.get("sunset"))
            },
            "timestamp": self._timestamp_to_datetime(data.get("dt")),
            "timezone": data.get("timezone")
        }
        
        return processed_data
    
    def _process_forecast_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw forecast data into a structured format."""
        if data.get("cod") != "200":
            return {"error": f"Forecast API error: {data.get('message', 'Unknown error')}"}
        
        city_info = data.get("city", {})
        forecast_list = data.get("list", [])
        
        processed_forecast = {
            "location": {
                "city": city_info.get("name"),
                "country": city_info.get("country"),
                "coordinates": {
                    "latitude": city_info.get("coord", {}).get("lat"),
                    "longitude": city_info.get("coord", {}).get("lon")
                }
            },
            "forecast": []
        }
        
        for forecast in forecast_list[:10]:  # Limit to first 10 entries (approx 2.5 days)
            main = forecast.get("main", {})
            weather = forecast.get("weather", [{}])[0]
            wind = forecast.get("wind", {})
            
            forecast_entry = {
                "timestamp": self._timestamp_to_datetime(forecast.get("dt")),
                "temperature": {
                    "current": main.get("temp"),
                    "feels_like": main.get("feels_like"),
                    "min": main.get("temp_min"),
                    "max": main.get("temp_max")
                },
                "conditions": {
                    "main": weather.get("main"),
                    "description": weather.get("description"),
                    "icon": weather.get("icon")
                },
                "additional": {
                    "humidity": main.get("humidity"),
                    "pressure": main.get("pressure"),
                    "wind_speed": wind.get("speed"),
                    "wind_direction": wind.get("deg"),
                    "cloudiness": forecast.get("clouds", {}).get("all"),
                    "precipitation": forecast.get("pop", 0)  # Probability of precipitation
                }
            }
            processed_forecast["forecast"].append(forecast_entry)
        
        return processed_forecast
    
    def _timestamp_to_datetime(self, timestamp: Optional[int]) -> Optional[str]:
        """Convert Unix timestamp to ISO format datetime string."""
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp).isoformat()
    
    def save_weather_data(self, data: Dict[str, Any], filename: str = "weather_data.json") -> bool:
        """Save weather data to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except (IOError, TypeError) as e:
            print(f"Failed to save data: {str(e)}")
            return False

def main():
    """Example usage of the WeatherFetcher class."""
    # Replace with your actual API key or set OPENWEATHER_API_KEY environment variable
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Fetch current weather for London
    print("Fetching current weather for London...")
    london_weather = fetcher.get_current_weather("London", "GB")
    
    if "error" not in london_weather:
        print(f"Current temperature in London: {london_weather['temperature']['current']}°C")
        print(f"Conditions: {london_weather['conditions']['description']}")
        
        # Save to file
        if fetcher.save_weather_data(london_weather, "london_weather.json"):
            print("Weather data saved to london_weather.json")
    else:
        print(f"Error: {london_weather['error']}")
    
    # Fetch forecast for New York
    print("\nFetching forecast for New York...")
    ny_forecast = fetcher.get_weather_forecast("New York", "US")
    
    if "error" not in ny_forecast and ny_forecast.get("forecast"):
        print(f"Forecast for {ny_forecast['location']['city']}:")
        for i, forecast in enumerate(ny_forecast["forecast"][:3], 1):
            print(f"  {i}. {forecast['timestamp']}: {forecast['temperature']['current']}°C, {forecast['conditions']['description']}")

if __name__ == "__main__":
    main()