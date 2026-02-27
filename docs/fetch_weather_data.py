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
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def display_weather(weather_data):
    if weather_data and weather_data.get('cod') == 200:
        main = weather_data['main']
        weather = weather_data['weather'][0]
        print(f"City: {weather_data['name']}")
        print(f"Temperature: {main['temp']}°C")
        print(f"Humidity: {main['humidity']}%")
        print(f"Weather: {weather['description']}")
        print(f"Pressure: {main['pressure']} hPa")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_weather_data.py <api_key> <city_name>")
        sys.exit(1)

    api_key = sys.argv[1]
    city = sys.argv[2]
    weather_info = get_weather(api_key, city)
    display_weather(weather_info)import requests
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()

    def get_current_weather(self, city_name, units="metric"):
        endpoint = f"{self.base_url}/weather"
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "feels_like": data["main"]["feels_like"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "weather": data["weather"][0]["description"],
                "wind_speed": data["wind"]["speed"],
                "timestamp": datetime.fromtimestamp(data["dt"]).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {city_name}: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format for {city_name}: {e}")
            return None

    def get_forecast(self, city_name, days=5, units="metric"):
        endpoint = f"{self.base_url}/forecast"
        params = {
            "q": city_name,
            "appid": self.api_key,
            "units": units,
            "cnt": days * 8
        }
        
        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecast = []
            for item in data["list"]:
                forecast.append({
                    "datetime": datetime.fromtimestamp(item["dt"]).isoformat(),
                    "temperature": item["main"]["temp"],
                    "feels_like": item["main"]["feels_like"],
                    "humidity": item["main"]["humidity"],
                    "weather": item["weather"][0]["description"],
                    "wind_speed": item["wind"]["speed"]
                })
            
            return forecast
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Forecast request failed for {city_name}: {e}")
            return None
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid forecast response format for {city_name}: {e}")
            return None

def save_to_json(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Data saved to {filename}")
        return True
    except IOError as e:
        logger.error(f"Failed to save data to {filename}: {e}")
        return False

def main():
    api_key = "your_api_key_here"
    cities = ["London", "New York", "Tokyo", "Paris", "Sydney"]
    
    fetcher = WeatherFetcher(api_key)
    
    all_weather_data = {}
    
    for city in cities:
        logger.info(f"Fetching weather for {city}")
        current = fetcher.get_current_weather(city)
        forecast = fetcher.get_forecast(city, days=3)
        
        if current and forecast:
            all_weather_data[city] = {
                "current": current,
                "forecast": forecast
            }
            logger.info(f"Successfully fetched data for {city}")
        else:
            logger.warning(f"Failed to fetch complete data for {city}")
    
    if all_weather_data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weather_data_{timestamp}.json"
        save_to_json(all_weather_data, filename)

if __name__ == "__main__":
    main()import requests

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
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
    if data and data.get('cod') == 200:
        city = data['name']
        country = data['sys']['country']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        description = data['weather'][0]['description']
        print(f"Weather in {city}, {country}:")
        print(f"Temperature: {temp}°C")
        print(f"Humidity: {humidity}%")
        print(f"Conditions: {description}")
    else:
        print("City not found or invalid data received.")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city = input("Enter city name: ")
    weather_data = get_weather(city, API_KEY)
    display_weather(weather_data)import requests
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
import logging

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_current_weather(self, city_name, country_code=None):
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
            data = response.json()
            
            return {
                'city': data['name'],
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'description': data['weather'][0]['description'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            self.logger.error(f"Data parsing error: {e}")
            return None

    def get_forecast(self, city_name, days=5):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/forecast/daily",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            forecast = []
            for item in data['list']:
                forecast.append({
                    'date': datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d'),
                    'temp_day': item['temp']['day'],
                    'temp_night': item['temp']['night'],
                    'humidity': item['humidity'],
                    'description': item['weather'][0]['description']
                })
            
            return forecast
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Forecast request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            self.logger.error(f"Forecast data parsing error: {e}")
            return None

def save_to_json(data, filename):
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        logging.error(f"Failed to save data: {e}")
        return False

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    
    fetcher = WeatherFetcher(API_KEY)
    
    current = fetcher.get_current_weather("London", "UK")
    if current:
        print(f"Current weather in {current['city']}:")
        print(f"Temperature: {current['temperature']}°C")
        print(f"Humidity: {current['humidity']}%")
        print(f"Conditions: {current['description']}")
        
        save_to_json(current, "current_weather.json")
    
    forecast = fetcher.get_forecast("London", 3)
    if forecast:
        print("\n3-day forecast:")
        for day in forecast:
            print(f"{day['date']}: {day['temp_day']}°C, {day['description']}")
        
        save_to_json(forecast, "weather_forecast.json")import requests
import json
import os
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        
    def get_weather(self, city_name, country_code=''):
        if not self.api_key:
            raise ValueError("API key not provided. Set OPENWEATHER_API_KEY environment variable.")
        
        query = f"{city_name},{country_code}" if country_code else city_name
        params = {
            'q': query,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.exceptions.RequestException as e:
            return {'error': f'Failed to fetch weather data: {str(e)}'}
    
    def _parse_response(self, data):
        if data.get('cod') != 200:
            return {'error': data.get('message', 'Unknown error')}
        
        return {
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
    
    def save_to_file(self, data, filename='weather_data.json'):
        if 'error' not in data:
            with open(filename, 'a') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
            return True
        return False

def main():
    fetcher = WeatherFetcher()
    
    cities = ['London', 'New York', 'Tokyo', 'Paris']
    
    for city in cities:
        print(f"\nFetching weather for {city}...")
        weather_data = fetcher.get_weather(city)
        
        if 'error' in weather_data:
            print(f"Error: {weather_data['error']}")
        else:
            print(f"Temperature: {weather_data['temperature']}°C")
            print(f"Weather: {weather_data['weather']} - {weather_data['description']}")
            print(f"Humidity: {weather_data['humidity']}%")
            print(f"Wind Speed: {weather_data['wind_speed']} m/s")
            
            if fetcher.save_to_file(weather_data):
                print(f"Data saved to file")

if __name__ == "__main__":
    main()