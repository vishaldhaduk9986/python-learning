import requests
import json

API_KEY = "9e159c3418ae797c5409fe6efa20edb0"  
city = "Rajkot"       
url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()
print("Response Data",data)
try:
    with open("weather.json", "w") as file:
        json.dump(data, file)
except FileNotFoundError:
    print("File not found")
except Exception as e:
    print("Error:", e)

print("Reading data back from file")

try:
    with open("weather.json", "r") as file:
        weather = json.load(file)
except FileNotFoundError:
    print("File not found")
except Exception as e:
    print("Error:", e)
temperature = weather["main"]["temp"]
humidity = weather["main"]["humidity"]


print(f"City: {city} ,Temperature: {temperature}°C")
print(f"Humidity: {humidity}%")




