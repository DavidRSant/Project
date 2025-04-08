from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import requests

OPENWEATHER_API_KEY = "95afa775ec36e1de4632cdb5ab1fd4aa"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WeatherResponse(BaseModel):
    location: str
    temperature: float
    description: str
    feels_like: float
    humidity: int
    wind_speed: float
    icon: str

class ForecastDay(BaseModel):
    date: str
    temperature: float
    feels_like: float
    description: str
    icon: str

class ForecastResponse(BaseModel):
    location: str
    forecast: List[ForecastDay]

@app.get("/weather", response_model=WeatherResponse)
def get_weather(city: Optional[str] = Query(None)):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    res = requests.get(url, params=params)
    data = res.json()
    return WeatherResponse(
        location=f"{data['name']}, {data['sys']['country']}",
        temperature=data["main"]["temp"],
        description=data["weather"][0]["description"],
        feels_like=data["main"]["feels_like"],
        humidity=data["main"]["humidity"],
        wind_speed=data["wind"]["speed"],
        icon=f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png"
    )

@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(city: Optional[str] = Query(None)):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    res = requests.get(url, params=params)
    data = res.json()
    seen = set()
    days = []
    for entry in data["list"]:
        date = entry["dt_txt"].split(" ")[0]
        if date not in seen:
            seen.add(date)
            days.append(ForecastDay(
                date=date,
                temperature=entry["main"]["temp"],
                feels_like=entry["main"]["feels_like"],
                description=entry["weather"][0]["description"],
                icon=f"http://openweathermap.org/img/wn/{entry['weather'][0]['icon']}@2x.png"
            ))
        if len(days) == 5:
            break
    return ForecastResponse(location=f"{data['city']['name']}, {data['city']['country']}", forecast=days)