
# 🌦️ Weather Lookup & History Tracker

A full-stack weather tracker that stores user queries in SQLite and displays summary analytics with Chart.js.

## 🚀 Features

- Add weather queries by city and date range
- Store and retrieve records (SQLite DB)
- Export weather records to CSV
- Filter by city
- Visualize average temperature per city with Chart.js
- Integrated YouTube and Google Maps city links

## 📦 Tech Stack

- **Backend**: FastAPI + SQLite + OpenWeatherMap API
- **Frontend**: HTML, JavaScript, Chart.js
- **Database**: SQLite (file-based)

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

Visit: `http://127.0.0.1:8001/docs` to test API endpoints.

---

### 3. Open the frontend

Just open `index.html` in your browser (no server needed for frontend).

---

## ✨ Functionality Overview

- `POST /weather` – Add weather data to database
- `GET /weather` – List all weather records
- `PUT /weather/{id}` – Update weather for a record
- `DELETE /weather/{id}` – Remove record from DB

---

## 📌 Notes

- Requires internet access to fetch data and load Chart.js
- OpenWeatherMap API key is hardcoded in `main.py` (you may change it)
