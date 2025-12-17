import requests
import json

def get_weather(city_name, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}&units=metric"
    
    try:
        response = requests.get(complete_url)
        response.raise_for_status()
        data = response.json()
        
        if data["cod"] != 200:
            return {"error": "City not found"}
        
        main = data["main"]
        weather_desc = data["weather"][0]["description"]
        
        result = {
            "city": data["name"],
            "temperature": main["temp"],
            "feels_like": main["feels_like"],
            "humidity": main["humidity"],
            "description": weather_desc,
            "pressure": main["pressure"],
            "wind_speed": data["wind"]["speed"]
        }
        
        return result
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {str(e)}"}
    except (KeyError, json.JSONDecodeError) as e:
        return {"error": f"Data parsing error: {str(e)}"}

def display_weather(weather_data):
    if "error" in weather_data:
        print(f"Error: {weather_data['error']}")
        return
    
    print(f"Weather in {weather_data['city']}:")
    print(f"Temperature: {weather_data['temperature']}°C")
    print(f"Feels like: {weather_data['feels_like']}°C")
    print(f"Humidity: {weather_data['humidity']}%")
    print(f"Conditions: {weather_data['description']}")
    print(f"Pressure: {weather_data['pressure']} hPa")
    print(f"Wind Speed: {weather_data['wind_speed']} m/s")

if __name__ == "__main__":
    API_KEY = "your_api_key_here"
    city = input("Enter city name: ").strip()
    
    if not city:
        print("City name cannot be empty")
    else:
        weather = get_weather(city, API_KEY)
        display_weather(weather)