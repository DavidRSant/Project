from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime
import sqlite3
import requests
import json

OPENWEATHER_API_KEY = "95afa775ec36e1de4632cdb5ab1fd4aa"
DB_NAME = "weather.db"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WeatherCreate(BaseModel):
    location: str
    start_date: date
    end_date: date

class WeatherRecord(BaseModel):
    id: int
    location: str
    start_date: str
    end_date: str
    requested_at: str
    weather_data: dict

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS weather (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            requested_at TEXT NOT NULL,
            weather_data TEXT NOT NULL
        )
        ''')
init_db()

@app.post("/weather", response_model=WeatherRecord)
def create_weather(data: WeatherCreate):
    if data.start_date > data.end_date:
        raise HTTPException(status_code=400, detail="Invalid date range")

    weather_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": data.location, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    res = requests.get(weather_url, params=params)

    if res.status_code != 200:
        raise HTTPException(status_code=404, detail="Location not found")

    weather_data = res.json()
    now = datetime.utcnow().isoformat()

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO weather (location, start_date, end_date, requested_at, weather_data)
        VALUES (?, ?, ?, ?, ?)
        ''', (data.location, str(data.start_date), str(data.end_date), now, json.dumps(weather_data)))
        conn.commit()
        record_id = cursor.lastrowid

    return WeatherRecord(
        id=record_id,
        location=data.location,
        start_date=str(data.start_date),
        end_date=str(data.end_date),
        requested_at=now,
        weather_data=weather_data
    )

@app.get("/weather", response_model=List[WeatherRecord])
def read_all_weather():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        rows = cursor.execute("SELECT * FROM weather").fetchall()
        return [
            WeatherRecord(
                id=row[0], location=row[1], start_date=row[2],
                end_date=row[3], requested_at=row[4],
                weather_data=json.loads(row[5])
            ) for row in rows
        ]

@app.put("/weather/{record_id}", response_model=WeatherRecord)
def update_weather(record_id: int, data: WeatherCreate):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM weather WHERE id=?", (record_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Record not found")

        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": data.location, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        res = requests.get(weather_url, params=params)

        if res.status_code != 200:
            raise HTTPException(status_code=404, detail="Location not found")

        weather_data = res.json()
        now = datetime.utcnow().isoformat()

        cursor.execute('''
        UPDATE weather SET location=?, start_date=?, end_date=?, requested_at=?, weather_data=?
        WHERE id=?
        ''', (data.location, str(data.start_date), str(data.end_date), now, json.dumps(weather_data), record_id))
        conn.commit()

        return WeatherRecord(
            id=record_id,
            location=data.location,
            start_date=str(data.start_date),
            end_date=str(data.end_date),
            requested_at=now,
            weather_data=weather_data
        )

@app.delete("/weather/{record_id}")
def delete_weather(record_id: int):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM weather WHERE id=?", (record_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        conn.commit()
        return {"detail": "Record deleted"}