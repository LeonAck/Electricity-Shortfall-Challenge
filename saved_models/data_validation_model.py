from pydantic import BaseModel, field_validator, model_validator, ValidationInfo, Field
from typing import Dict, Any, Optional


class FeaturesInput(BaseModel):

    # Allow population by field name (replaces old allow_population_by_field_name)
    model_config = {"validate_by_name": True}  # or "populate_by_name" depending on your Pydantic version

    # IDs / indices
    Unnamed_0: Optional[int] = Field(None, alias="Unnamed: 0")

    # Timestamp
    time: Optional[str]

    # Wind
    Madrid_wind_speed: Optional[float]
    Valencia_wind_deg: Optional[str]
    Bilbao_wind_speed: Optional[float]
    Barcelona_wind_speed: Optional[float]
    Seville_wind_speed: Optional[float]
    Valencia_wind_speed: Optional[float]

    # Rain
    Bilbao_rain_1h: Optional[float]
    Barcelona_rain_1h: Optional[float]
    Seville_rain_1h: Optional[float]
    Seville_rain_3h: Optional[float]
    Barcelona_rain_3h: Optional[float]
    Madrid_rain_1h: Optional[float]

    # Snow
    Bilbao_snow_3h: Optional[float]
    Valencia_snow_3h: Optional[float]

    # Humidity
    Seville_humidity: Optional[float]
    Madrid_humidity: Optional[float]
    Valencia_humidity: Optional[float]

    # Clouds
    Bilbao_clouds_all: Optional[float]
    Seville_clouds_all: Optional[float]
    Madrid_clouds_all: Optional[float]

    # Pressure
    Barcelona_pressure: Optional[float]
    Bilbao_pressure: Optional[float]
    Madrid_pressure: Optional[float]
    Valencia_pressure: Optional[float] 

    # Temperature
    Seville_temp: Optional[float]
    Seville_temp_max: Optional[float]
    Seville_temp_min: Optional[float]
    Madrid_temp: Optional[float]
    Madrid_temp_max: Optional[float]
    Madrid_temp_min: Optional[float]
    Bilbao_temp: Optional[float]
    Bilbao_temp_min: Optional[float]
    Bilbao_temp_max: Optional[float]
    Barcelona_temp: Optional[float]
    Barcelona_temp_max: Optional[float]
    Barcelona_temp_min: Optional[float]
    Valencia_temp: Optional[float]
    Valencia_temp_max: Optional[float]
    Valencia_temp_min: Optional[float]

    # Weather ID
    Madrid_weather_id: Optional[float]
    Barcelona_weather_id: Optional[float]
    Bilbao_weather_id: Optional[float]
    Seville_weather_id: Optional[float]

    # Wind direction (degrees as float)
    Barcelona_wind_deg: Optional[float]
    Bilbao_wind_deg: Optional[float]

    # Categorical pressure
    Seville_pressure: Optional[str]

    load_shortfall_3h: Optional[float] = None

    # Validator: Non-negative wind speed
    @field_validator('Madrid_wind_speed', 'Bilbao_wind_speed', 'Barcelona_wind_speed', 'Seville_wind_speed')
    @classmethod
    def non_negative_wind_speed(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError(f'{info.field_name} must be non-negative')
        return v

    # Validator: Positive pressure
    @field_validator('Valencia_pressure', 'Barcelona_pressure', 'Bilbao_pressure', 'Madrid_pressure')
    @classmethod
    def positive_pressure(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError(f'{info.field_name} must be positive')
        return v

    # Root validator: Log unknown fields
    @model_validator(mode='before')
    @classmethod
    def remove_unnecessary_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return data

        known_fields = set(cls.model_fields.keys())
        known_aliases = {f.alias for f in cls.model_fields.values() if f.alias}
        allowed_keys = known_fields | known_aliases

        unknown = set(data.keys()) - allowed_keys
        if unknown:
            print(f"⚠️ Ignoring unexpected fields: {unknown}")
        return data


class PredictionRequest(BaseModel):
    features: FeaturesInput


# Test data
test_json = {
    "features": {
        "Unnamed: 0": 8763,
        "time": "2018-01-01 00:00:00",
        "Madrid_wind_speed": 5.0,
        "Valencia_wind_deg": "level_8",
        "Bilbao_rain_1h": 0.0,
        "Valencia_wind_speed": 5.0,
        "Seville_humidity": 87.0,
        "Madrid_humidity": 71.3333333333,
        "Bilbao_clouds_all": 20.0,
        "Bilbao_wind_speed": 3.0,
        "Seville_clouds_all": 0.0,
        "Bilbao_wind_deg": 193.3333333333,
        "Barcelona_wind_speed": 4.0,
        "Barcelona_wind_deg": 176.6666666667,
        "Madrid_clouds_all": 0.0,
        "Seville_wind_speed": 1.0,
        "Barcelona_rain_1h": 0.0,
        "Seville_pressure": "sp25",
        "Seville_rain_1h": 0.0,
        "Bilbao_snow_3h": 0,
        "Barcelona_pressure": 1017.3333333333,
        "Seville_rain_3h": 0.0,
        "Madrid_rain_1h": 0.0,
        "Barcelona_rain_3h": 0.0,
        "Valencia_snow_3h": 0,
        "Madrid_weather_id": 800.0,
        "Barcelona_weather_id": 800.0,
        "Bilbao_pressure": 1025.6666666667,
        "Seville_weather_id": 800.0,
        "Valencia_pressure": None,
        "Seville_temp_max": 284.4833333333,
        "Madrid_pressure": 1030.0,
        "Valencia_temp_max": 287.4833333333,
        "Valencia_temp": 287.4833333333,
        "Bilbao_weather_id": 801.0,
        "Seville_temp": 283.6733333333,
        "Valencia_humidity": 46.3333333333,
        "Valencia_temp_min": 287.4833333333,
        "Barcelona_temp_max": 287.8166666667,
        "Madrid_temp_max": 280.8166666667,
        "Barcelona_temp": 287.3566666667,
        "Bilbao_temp_min": 276.15,
        "Bilbao_temp": 280.38,
        "Barcelona_temp_min": 286.8166666667,
        "Bilbao_temp_max": 285.15,
        "Seville_temp_min": 283.15,
        "Madrid_temp": 279.8666666667,
        "Madrid_temp_min": 279.15
    }
}
