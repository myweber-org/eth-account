import requests
import json
from datetime import datetime

class WeatherFetcher:
    def __init__(self, api_key, base_url="http://api.openweathermap.org/data/2.5/weather"):
        self.api_key = api_key
        self.base_url = base_url

    def get_weather_by_city(self, city_name, units="metric"):
        params = {
            'q': city_name,
            'appid': self.api_key,
            'units': units
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return self._parse_weather_data(response.json())
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to fetch weather data: {e}"}

    def _parse_weather_data(self, data):
        if data.get("cod") != 200:
            return {"error": data.get("message", "Unknown error")}

        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})

        return {
            "city": data.get("name"),
            "country": data.get("sys", {}).get("country"),
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "weather": weather.get("description"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "timestamp": datetime.fromtimestamp(data.get("dt")).isoformat()
        }

def main():
    api_key = "your_api_key_here"
    fetcher = WeatherFetcher(api_key)
    
    cities = ["London", "Tokyo", "New York"]
    for city in cities:
        print(f"\nFetching weather for {city}...")
        result = fetcher.get_weather_by_city(city)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()