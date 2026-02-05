import json

def calculate_average_temperature(data_file):
    """
    Reads weather data from a JSON file and calculates the average temperature.
    """
    try:
        with open(data_file, 'r') as file:
            data = json.load(file)

        if not data or 'readings' not in data:
            return None

        temperatures = [reading['temp'] for reading in data['readings'] if 'temp' in reading]
        
        if not temperatures:
            return None

        average_temp = sum(temperatures) / len(temperatures)
        return round(average_temp, 2)

    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{data_file}' contains invalid JSON.")
        return None
    except KeyError as e:
        print(f"Error: Missing expected key in data: {e}")
        return None

if __name__ == "__main__":
    result = calculate_average_temperature('weather_data.json')
    if result is not None:
        print(f"Average Temperature: {result}°C")
    else:
        print("Could not calculate average temperature.")