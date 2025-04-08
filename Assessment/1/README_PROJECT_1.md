
# 🌤️ Real-Time Weather App

A simple real-time weather and 5-day forecast viewer using FastAPI and OpenWeatherMap API.

## 🚀 Features

- View current weather by city or ZIP
- 5-day weather forecast
- FastAPI-powered backend
- Simple HTML + JavaScript frontend

## 📦 Tech Stack

- **Backend**: FastAPI + OpenWeatherMap API
- **Frontend**: HTML, JavaScript
- **No database required**

---

## 🧰 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server

```bash
uvicorn main:app --reload --port 8001
```

Visit: `http://127.0.0.1:8001/docs` to test endpoints manually.

### 3. Open the frontend

Open the `index.html` file in your browser.

---

## ⚙️ API Endpoints

- `GET /weather?city=...` – Get current weather
- `GET /forecast?city=...` – Get 5-day forecast

---

## 🔑 Notes

- Requires internet access (to reach OpenWeather API)
- API key is embedded in `main.py` – replace with your own for production
