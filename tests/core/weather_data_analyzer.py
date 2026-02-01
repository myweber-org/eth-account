import json

def calculate_average_temperature(data_file):
    """
    Calculate the average temperature from weather data stored in a JSON file.
    The JSON file should contain a list of temperature readings.
    """
    try:
        with open(data_file, 'r') as file:
            weather_data = json.load(file)
        
        if not isinstance(weather_data, list):
            raise ValueError("Data should be a list of temperature readings")
        
        if not weather_data:
            return 0.0
        
        total = sum(weather_data)
        average = total / len(weather_data)
        return round(average, 2)
    
    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{data_file}' contains invalid JSON.")
        return None
    except (TypeError, ValueError) as e:
        print(f"Error processing data: {e}")
        return None

def generate_temperature_report(data_file, output_file=None):
    """
    Generate a report of temperature statistics.
    Optionally save the report to a file.
    """
    avg_temp = calculate_average_temperature(data_file)
    
    if avg_temp is None:
        return
    
    report = f"Weather Data Analysis Report\n"
    report += f"=============================\n"
    report += f"Data Source: {data_file}\n"
    report += f"Average Temperature: {avg_temp}°C\n"
    report += f"Analysis Complete\n"
    
    print(report)
    
    if output_file:
        try:
            with open(output_file, 'w') as file:
                file.write(report)
            print(f"Report saved to: {output_file}")
        except IOError as e:
            print(f"Error saving report: {e}")

if __name__ == "__main__":
    # Example usage
    data_file = "temperature_readings.json"
    generate_temperature_report(data_file, "weather_report.txt")