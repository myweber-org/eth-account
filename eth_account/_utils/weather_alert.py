import requests
import json
import time
from datetime import datetime

class WeatherAlert:
    def __init__(self, api_key, city, threshold_temp=30):
        self.api_key = api_key
        self.city = city
        self.threshold_temp = threshold_temp
        self.alert_log = []
        
    def fetch_weather(self):
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': self.city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def check_temperature(self, weather_data):
        if weather_data and 'main' in weather_data:
            current_temp = weather_data['main']['temp']
            feels_like = weather_data['main']['feels_like']
            
            alert_triggered = current_temp > self.threshold_temp
            
            if alert_triggered:
                alert_msg = f"ALERT: Temperature {current_temp}°C exceeds threshold {self.threshold_temp}°C"
                self.log_alert(alert_msg, current_temp, feels_like)
                return True, alert_msg
            else:
                return False, f"Temperature {current_temp}°C is within safe range"
        
        return False, "Unable to read temperature data"
    
    def log_alert(self, message, temp, feels_like):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'temperature': temp,
            'feels_like': feels_like,
            'city': self.city
        }
        self.alert_log.append(log_entry)
        
        with open('weather_alerts.json', 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    
    def monitor(self, interval_seconds=300):
        print(f"Starting weather monitor for {self.city}")
        print(f"Alert threshold: {self.threshold_temp}°C")
        print(f"Check interval: {interval_seconds} seconds")
        
        try:
            while True:
                weather_data = self.fetch_weather()
                
                if weather_data:
                    alert, message = self.check_temperature(weather_data)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    if alert:
                        print(f"[{timestamp}] ⚠️  {message}")
                    else:
                        print(f"[{timestamp}] ✅ {message}")
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self.print_summary()
    
    def print_summary(self):
        print(f"\n--- Monitoring Summary ---")
        print(f"Total alerts triggered: {len(self.alert_log)}")
        print(f"City: {self.city}")
        print(f"Threshold: {self.threshold_temp}°C")
        
        if self.alert_log:
            print("\nRecent alerts:")
            for alert in self.alert_log[-3:]:
                print(f"  {alert['timestamp']}: {alert['message']}")

def main():
    api_key = "your_api_key_here"
    city = "London"
    
    alert_system = WeatherAlert(api_key, city, threshold_temp=25)
    alert_system.monitor(interval_seconds=600)

if __name__ == "__main__":
    main()