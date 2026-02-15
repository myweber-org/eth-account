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
    main()import requests
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
    print(f"  Temperature: {temp}°C (Feels like: {feels_like}°C)")
    print(f"  Conditions: {weather_desc.capitalize()}")
    print(f"  Humidity: {humidity}%")
    print(f"  Wind Speed: {wind_speed} m/s")

def main():
    if len(sys.argv) < 3:
        print("Usage: python fetch_weather_data.py <API_KEY> <CITY_NAME>")
        print("Example: python fetch_weather_data.py your_api_key_here London")
        sys.exit(1)

    api_key = sys.argv[1]
    city = ' '.join(sys.argv[2:])

    weather_data = get_weather(api_key, city)
    display_weather(weather_data)

if __name__ == "__main__":
    main()import requests
import json
from datetime import datetime
import sys

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_current_weather(self, city_name, country_code=None):
        """Fetch current weather data for a given city."""
        query = city_name
        if country_code:
            query += f",{country_code}"
            
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/weather",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def get_weather_forecast(self, city_name, days=5):
        """Fetch weather forecast for multiple days."""
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/forecast",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast: {e}")
            return None
    
    def format_weather_data(self, weather_data):
        """Format weather data for display."""
        if not weather_data:
            return "No weather data available"
        
        main = weather_data.get('main', {})
        weather = weather_data.get('weather', [{}])[0]
        wind = weather_data.get('wind', {})
        
        formatted = f"""
Weather Report:
---------------
Location: {weather_data.get('name', 'Unknown')}
Temperature: {main.get('temp', 'N/A')}°C
Feels Like: {main.get('feels_like', 'N/A')}°C
Humidity: {main.get('humidity', 'N/A')}%
Pressure: {main.get('pressure', 'N/A')} hPa
Conditions: {weather.get('description', 'N/A').title()}
Wind Speed: {wind.get('speed', 'N/A')} m/s
Wind Direction: {wind.get('deg', 'N/A')}°
Visibility: {weather_data.get('visibility', 'N/A')} meters
"""
        return formatted
    
    def save_to_file(self, data, filename=None):
        """Save weather data to a JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"weather_data_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Weather data saved to {filename}")
            return True
        except IOError as e:
            print(f"Error saving to file: {e}")
            return False

def main():
    # Replace with your actual API key
    API_KEY = "your_api_key_here"
    
    if API_KEY == "your_api_key_here":
        print("Please set your API key in the script")
        sys.exit(1)
    
    fetcher = WeatherFetcher(API_KEY)
    
    # Example usage
    city = "London"
    
    print(f"Fetching current weather for {city}...")
    current_weather = fetcher.get_current_weather(city)
    
    if current_weather:
        print(fetcher.format_weather_data(current_weather))
        fetcher.save_to_file(current_weather)
    
    print(f"\nFetching 5-day forecast for {city}...")
    forecast = fetcher.get_weather_forecast(city, days=5)
    
    if forecast:
        fetcher.save_to_file(forecast, "weather_forecast.json")
        print(f"Forecast data saved successfully")

if __name__ == "__main__":
    main()