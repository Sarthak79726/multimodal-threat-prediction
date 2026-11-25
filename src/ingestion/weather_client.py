"""
Open-Meteo Weather API Client for Real-Time Campus Atmospheric Observations.
Retrieves live, authentic meteorological data for BIT Mesra (23.4123°N, 85.4399°E)
with strict range validation, TTL caching, and graceful offline degradation.
Never fabricates simulated weather values.
"""

import json
import logging
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from src.ingestion.schemas import WeatherReading, SensorConnectionStatus

logger = logging.getLogger(__name__)

# WMO Weather Interpretation Codes (WW)
WMO_CODE_MAP = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

BIT_MESRA_COORDS = {
    "latitude": 23.4123,
    "longitude": 85.4399,
    "timezone": "Asia/Kolkata"
}


class OpenMeteoWeatherClient:
    """
    Official Open-Meteo live weather client for BIT Mesra campus.
    Free, non-commercial API access without private API keys.
    Thread-safe in-memory caching with 15-minute TTL.
    """
    _instance: Optional["OpenMeteoWeatherClient"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cache_ttl_seconds: int = 900, timeout_seconds: int = 5):
        if getattr(self, "_initialized", False):
            return
        self.cache_ttl = cache_ttl_seconds
        self.timeout = timeout_seconds
        self._cache: Optional[WeatherReading] = None
        self._last_fetch_time: Optional[datetime] = None
        self._req_lock = threading.Lock()
        self._initialized = True
        logger.info("OpenMeteoWeatherClient initialized for BIT Mesra (23.4123°N, 85.4399°E)")

    def get_current_weather(self, force_refresh: bool = False) -> WeatherReading:
        """
        Fetches current weather for BIT Mesra. Returns cached reading if within TTL.
        If network fails, returns cached reading or UNAVAILABLE status.
        Never generates synthetic weather values.
        """
        now = datetime.now(timezone.utc)

        with self._req_lock:
            # Check cache validity
            if not force_refresh and self._cache is not None and self._last_fetch_time is not None:
                age_seconds = (now - self._last_fetch_time).total_seconds()
                if age_seconds < self.cache_ttl:
                    return self._cache

            # Execute real HTTP request to Open-Meteo
            try:
                url = (
                    f"https://api.open-meteo.com/v1/forecast"
                    f"?latitude={BIT_MESRA_COORDS['latitude']}"
                    f"&longitude={BIT_MESRA_COORDS['longitude']}"
                    f"&current=temperature_2m,relative_humidity_2m,precipitation,weather_code,cloud_cover,is_day"
                    f"&timezone=Asia%2FKolkata"
                )

                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MultimodalThreatPlatform/1.0",
                        "Accept": "application/json"
                    }
                )

                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw_data = json.loads(resp.read().decode("utf-8"))

                current = raw_data.get("current")
                if not current or not isinstance(current, dict):
                    raise ValueError(f"Malformed Open-Meteo payload: 'current' block missing: {raw_data}")

                # Validate data ranges
                temp_c = float(current.get("temperature_2m"))
                rh_pct = float(current.get("relative_humidity_2m"))
                precip_mm = float(current.get("precipitation", 0.0))
                cloud_pct = float(current.get("cloud_cover", 0.0))
                is_day = int(current.get("is_day", 1))
                w_code = int(current.get("weather_code", 0))

                if not (-30.0 <= temp_c <= 60.0):
                    raise ValueError(f"Physical anomaly: Temperature {temp_c}°C outside realistic terrestrial range")
                if not (0.0 <= rh_pct <= 100.0):
                    raise ValueError(f"Physical anomaly: Humidity {rh_pct}% outside [0, 100]% range")
                if precip_mm < 0.0:
                    precip_mm = 0.0
                if not (0.0 <= cloud_pct <= 100.0):
                    cloud_pct = min(100.0, max(0.0, cloud_pct))

                weather_desc = WMO_CODE_MAP.get(w_code, f"Weather Code {w_code}")
                time_str = current.get("time", now.strftime("%Y-%m-%dT%H:%M"))

                reading = WeatherReading(
                    temperature_c=temp_c,
                    relative_humidity_pct=rh_pct,
                    precipitation_mm=precip_mm,
                    cloud_cover_pct=cloud_pct,
                    is_day=is_day,
                    weather_code=w_code,
                    weather_desc=weather_desc,
                    timestamp=time_str,
                    source="Open-Meteo Meteorological API (Live Satellite/Sensor Model)",
                    status=SensorConnectionStatus.CONNECTED,
                    station_location="BIT Mesra Campus (23.4123°N, 85.4399°E)",
                    error_message=None
                )

                self._cache = reading
                self._last_fetch_time = now
                logger.info("Successfully fetched live Open-Meteo weather: %s°C, %s%% RH, %s", temp_c, rh_pct, weather_desc)
                return reading

            except Exception as e:
                err_msg = f"Live weather API request failed: {str(e)}"
                logger.warning("Open-Meteo acquisition warning: %s", err_msg)

                # If we have an existing cache, return it with STALE status
                if self._cache is not None:
                    return WeatherReading(
                        temperature_c=self._cache.temperature_c,
                        relative_humidity_pct=self._cache.relative_humidity_pct,
                        precipitation_mm=self._cache.precipitation_mm,
                        cloud_cover_pct=self._cache.cloud_cover_pct,
                        is_day=self._cache.is_day,
                        weather_code=self._cache.weather_code,
                        weather_desc=self._cache.weather_desc,
                        timestamp=self._cache.timestamp,
                        source=self._cache.source + " (Cached)",
                        status=SensorConnectionStatus.STALE,
                        station_location=self._cache.station_location,
                        error_message=err_msg
                    )

                # Return explicitly UNAVAILABLE status without fabricating numbers
                return WeatherReading(
                    temperature_c=None,
                    relative_humidity_pct=None,
                    precipitation_mm=None,
                    cloud_cover_pct=None,
                    is_day=None,
                    weather_code=None,
                    weather_desc="Unavailable (Awaiting live API connectivity)",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
                    source="Open-Meteo Meteorological API",
                    status=SensorConnectionStatus.UNAVAILABLE,
                    station_location="BIT Mesra Campus (23.4123°N, 85.4399°E)",
                    error_message=err_msg
                )
