# synthetic_data_generator.py
from datetime import datetime, timedelta
import random
import numpy as np
from typing import List, Optional
from src.electricity_forecast.data_validation_model import FeaturesInput, PredictionRequest

# Starting point from real data
START_INDEX = 8763
START_TIME = datetime(2018, 1, 1, 0, 0)  # 2018-01-01 00:00:00

def generate_synthetic_entry(
    base_datetime: datetime,
    include_target: bool = False,
    drift: bool = False
) -> dict:
    """Generate one synthetic row matching real data format and ranges."""
    hour = base_datetime.hour
    month = base_datetime.month

    # Drift offsets
    temp_drift = 5.0 if drift else 0.0
    wind_drift = 3.0 if drift else 0.0

    # Daily temp pattern (in Kelvin, ~273K = 0°C)
    def daily_temp(base_celsius: float, amplitude: float) -> float:
        celsius = base_celsius + amplitude * np.sin(2 * np.pi * (hour - 10) / 24)
        return celsius + 273.15 + temp_drift

    # === Wind ===
    madrid_wind = max(0, np.random.normal(4 + wind_drift, 2))
    valencia_wind_deg = random.choice([
        "level_1", "level_2", "level_3", "level_4", "level_5",
        "level_6", "level_7", "level_8", "level_9", "level_10"
    ])
    bilbao_wind = max(0, np.random.normal(3 + wind_drift, 1.5))
    barcelona_wind = max(0, np.random.normal(4.5, 2))
    seville_wind = max(0, np.random.normal(2.5, 1.2))
    valencia_wind = max(0, np.random.normal(5 + wind_drift, 2))

    # === Rain/Snow ===
    rain_prob = 0.1 + (0.2 * (month in [10, 11, 12, 1]))  # higher in winter
    bilbao_rain = 0.0 if random.random() > rain_prob else abs(np.random.exponential(2))
    barcelona_rain = 0.0 if random.random() > rain_prob else abs(np.random.exponential(1.8))
    seville_rain = 0.0 if random.random() > rain_prob * 0.5 else abs(np.random.exponential(0.8))
    madrid_rain = 0.0 if random.random() > rain_prob else abs(np.random.exponential(1.5))

    barcelona_rain_3h = barcelona_rain * (1 + random.uniform(0, 2))
    seville_rain_3h = seville_rain * (1 + random.uniform(0, 2))

    bilbao_snow = 0.0 if random.random() > 0.03 else abs(np.random.exponential(0.3))
    valencia_snow = 0.0  # Almost never

    # === Humidity (%)
    madrid_humidity = np.clip(np.random.normal(70, 15), 0, 100)
    valencia_humidity = np.clip(np.random.normal(65, 12), 0, 100)
    seville_humidity = np.clip(np.random.normal(75, 18), 0, 100)

    # === Clouds (%)
    madrid_clouds = np.clip(np.random.normal(30, 25), 0, 100)
    bilbao_clouds = np.clip(np.random.normal(50, 30), 0, 100)
    seville_clouds = np.clip(np.random.normal(40, 35), 0, 100)

    # === Pressure (hPa)
    madrid_pressure = np.random.normal(1020, 10)
    bilbao_pressure = np.random.normal(1015, 12)
    barcelona_pressure = np.random.normal(1018, 11)
    valencia_pressure = np.random.normal(1022, 10)

    # === Temperature (Kelvin)
    seville_temp = daily_temp(18, 10)
    madrid_temp = daily_temp(12, 10)
    bilbao_temp = daily_temp(13, 8)
    barcelona_temp = daily_temp(14, 9)
    valencia_temp = daily_temp(16, 9)

    # === Weather ID (simplified)
    def weather_id_from_conditions(clouds, rain):
        if rain > 0.5:
            return 500  # rain
        elif clouds > 80:
            return 804  # overcast
        elif clouds > 50:
            return 803  # broken
        elif clouds > 20:
            return 802  # scattered
        else:
            return 800  # clear

    madrid_weather_id = weather_id_from_conditions(madrid_clouds, madrid_rain)
    bilbao_weather_id = weather_id_from_conditions(bilbao_clouds, bilbao_rain)
    barcelona_weather_id = weather_id_from_conditions(barcelona_wind, barcelona_rain)
    seville_weather_id = weather_id_from_conditions(seville_clouds, seville_rain)

    # === Wind direction (degrees)
    barcelona_wind_deg = random.uniform(0, 360)
    bilbao_wind_deg = random.uniform(0, 360)

    # === Categorical pressure
    seville_pressure = random.choice(["sp25", "sp50", "sp75", "sp100"])

    # === Target
    load_shortfall_3h = None
    if include_target:
        base = 50 + 20 * (
            np.sin(2 * np.pi * (hour - 8) / 24) +
            np.sin(2 * np.pi * (hour - 19) / 24)
        )
        load_shortfall_3h = base + np.random.normal(0, 8)

    return {
        "Unnamed: 0": int((base_datetime - datetime(2018, 1, 1)).total_seconds() // 3600) + 8760,
        "time": base_datetime.strftime("%Y-%m-%d %H:%M:%S"),

        "Madrid_wind_speed": round(madrid_wind, 10),
        "Valencia_wind_deg": valencia_wind_deg,
        "Bilbao_rain_1h": round(bilbao_rain, 10),
        "Valencia_wind_speed": round(valencia_wind, 10),
        "Seville_humidity": round(seville_humidity, 10),
        "Madrid_humidity": round(madrid_humidity, 10),
        "Valencia_humidity": round(valencia_humidity, 10),
        "Bilbao_clouds_all": round(bilbao_clouds, 10),
        "Bilbao_wind_speed": round(bilbao_wind, 10),
        "Seville_clouds_all": round(seville_clouds, 10),
        "Bilbao_wind_deg": round(bilbao_wind_deg, 10),
        "Barcelona_wind_speed": round(barcelona_wind, 10),
        "Barcelona_wind_deg": round(barcelona_wind_deg, 10),
        "Madrid_clouds_all": round(madrid_clouds, 10),
        "Seville_wind_speed": round(seville_wind, 10),
        "Barcelona_rain_1h": round(barcelona_rain, 10),
        "Seville_pressure": seville_pressure,
        "Seville_rain_1h": round(seville_rain, 10),
        "Bilbao_snow_3h": round(bilbao_snow, 10),
        "Barcelona_pressure": round(barcelona_pressure, 10),
        "Seville_rain_3h": round(seville_rain_3h, 10),
        "Madrid_rain_1h": round(madrid_rain, 10),
        "Barcelona_rain_3h": round(barcelona_rain_3h, 10),
        "Valencia_snow_3h": round(valencia_snow, 10),
        "Madrid_weather_id": madrid_weather_id,
        "Barcelona_weather_id": barcelona_weather_id,
        "Bilbao_pressure": round(bilbao_pressure, 10),
        "Seville_weather_id": seville_weather_id,
        "Valencia_pressure": round(valencia_pressure, 10),
        "Seville_temp_max": round(seville_temp + 2, 10),
        "Madrid_pressure": round(madrid_pressure, 10),
        "Valencia_temp_max": round(valencia_temp + 2, 10),
        "Valencia_temp": round(valencia_temp, 10),
        "Bilbao_weather_id": bilbao_weather_id,
        "Seville_temp": round(seville_temp, 10),
        "Valencia_humidity": round(valencia_humidity, 10),
        "Valencia_temp_min": round(valencia_temp - 2, 10),
        "Barcelona_temp_max": round(barcelona_temp + 2, 10),
        "Madrid_temp_max": round(madrid_temp + 2, 10),
        "Barcelona_temp": round(barcelona_temp, 10),
        "Bilbao_temp_min": round(bilbao_temp - 2, 10),
        "Bilbao_temp": round(bilbao_temp, 10),
        "Barcelona_temp_min": round(barcelona_temp - 2, 10),
        "Bilbao_temp_max": round(bilbao_temp + 2, 10),
        "Seville_temp_min": round(seville_temp - 2, 10),
        "Madrid_temp": round(madrid_temp, 10),
        "Madrid_temp_min": round(madrid_temp - 2, 10),

        "load_shortfall_3h": round(load_shortfall_3h, 10) if load_shortfall_3h is not None else None
    }


def generate_synthetic_dataset(
    n_timesteps: int = 16,  # e.g., 16 steps = 48 hours
    start_index: int = START_INDEX,
    start_time: datetime = START_TIME,
    include_target: bool = True,
    drift_after_timesteps: Optional[int] = None  # e.g., drift after 8 steps
) -> List[PredictionRequest]:
    """
    Generate synthetic data with 3-hour intervals.
    """
    requests = []
    
    for i in range(n_timesteps):
        current_index = start_index + i
        current_time = start_time + timedelta(hours=3 * i)  # 3-hour steps

        # Drift flag: after N timesteps
        drift = drift_after_timesteps is not None and i >= drift_after_timesteps

        raw_data = generate_synthetic_entry(
            base_datetime=current_time,
            include_target=include_target,
            drift=drift
        )

        # Fix: ensure index matches real data
        raw_data["Unnamed: 0"] = current_index

        try:
            request_obj = PredictionRequest(features=FeaturesInput(**raw_data))
            requests.append(request_obj)
        except Exception as e:
            print(f"Validation error at index {current_index} ({current_time}): {e}")
            continue

    return requests

# === Example Usage ===
if __name__ == "__main__":
    data = generate_synthetic_dataset(
        n_timesteps=48,
        start_index=START_INDEX,
        start_time=START_TIME,
        include_target=True,
        drift_after_timesteps=24
    )

    # Save as JSON
    output = [
        {
            "time": entry.features.time,
            "features": entry.features.model_dump(by_alias=True)
        }
        for entry in data
    ]

    import json
    with open("synthetic_test_data_aligned.json", "w") as f:
        json.dump(output, f, indent=2)

    print("✅ Generated 48 hours of realistic synthetic data. Saved to synthetic_test_data_aligned.json")