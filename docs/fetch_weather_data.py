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
    display_weather(weather_data)