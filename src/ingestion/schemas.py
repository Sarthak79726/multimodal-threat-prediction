"""
Real-Time Data Ingestion Schemas and Data Models.
Defines standardized data contracts, connection states, and provenance models
for real sensor ingestion, external APIs, and hardware connectors.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime


class SensorModality(str, Enum):
    PHYSIOLOGICAL = "PHYSIOLOGICAL"
    SPATIAL_GPS = "SPATIAL_GPS"
    ENVIRONMENTAL_WEATHER = "ENVIRONMENTAL_WEATHER"
    ENVIRONMENTAL_LIGHTING = "ENVIRONMENTAL_LIGHTING"
    ENVIRONMENTAL_CROWD = "ENVIRONMENTAL_CROWD"
    MULTIMODAL_FUSION = "MULTIMODAL_FUSION"


class SensorConnectionStatus(str, Enum):
    CONNECTED = "CONNECTED"                      # Live sensor streaming fresh data
    STREAMING = "STREAMING"                      # Active real-time packet reception
    STALE = "STALE"                              # Connected but data exceeds freshness timeout
    DISCONNECTED = "DISCONNECTED"                # Device configured but no packet received
    UNAVAILABLE = "UNAVAILABLE"                  # Integration failed or offline
    AWAITING_HARDWARE = "AWAITING_HARDWARE"      # Interface ready, physical sensor not yet deployed
    AUTHENTICATION_REQUIRED = "AUTH_REQUIRED"    # API endpoint requires credentials
    BENCHMARK_REPLAY = "BENCHMARK_REPLAY"        # Legitimate empirical research dataset replay


class DataSourceType(str, Enum):
    LIVE_API = "LIVE_API"
    PHYSICAL_IOT_SENSOR = "PHYSICAL_IOT_SENSOR"
    LIVE_BLE_WEARABLE = "LIVE_BLE_WEARABLE"
    BROWSER_GEOLOCATION = "BROWSER_GEOLOCATION"
    RESEARCH_BENCHMARK_DATASET = "RESEARCH_BENCHMARK_DATASET"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    SCHEDULE_BASELINE_MODEL = "SCHEDULE_BASELINE_MODEL"


@dataclass
class WeatherReading:
    """Live weather observation from official meteorological API."""
    temperature_c: Optional[float]
    relative_humidity_pct: Optional[float]
    precipitation_mm: Optional[float]
    cloud_cover_pct: Optional[float]
    is_day: Optional[int]
    weather_code: Optional[int]
    weather_desc: str
    timestamp: str
    source: str
    status: SensorConnectionStatus
    station_location: str = "BIT Mesra Campus (23.4123°N, 85.4399°E)"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature_c": self.temperature_c,
            "relative_humidity_pct": self.relative_humidity_pct,
            "precipitation_mm": self.precipitation_mm,
            "cloud_cover_pct": self.cloud_cover_pct,
            "is_day": self.is_day,
            "weather_code": self.weather_code,
            "weather_desc": self.weather_desc,
            "timestamp": self.timestamp,
            "source": self.source,
            "status": self.status.value,
            "station_location": self.station_location,
            "error_message": self.error_message
        }


@dataclass
class LightingSensorReading:
    """Direct photometer / lux measurement from physical IoT sensor."""
    lux: Optional[float]
    indoor_flag: bool = False
    sensor_model: str = "BH1750 / Ambient Photometer"
    timestamp: Optional[str] = None
    status: SensorConnectionStatus = SensorConnectionStatus.AWAITING_HARDWARE
    hardware_port: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lux": self.lux,
            "indoor_flag": self.indoor_flag,
            "sensor_model": self.sensor_model,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "hardware_port": self.hardware_port,
            "error_message": self.error_message
        }


@dataclass
class CrowdSensorReading:
    """Direct people count / occupancy from privacy-preserving sensor."""
    people_count: Optional[int]
    density_category: Optional[str] = None
    occupancy_rate: Optional[float] = None
    sensor_type: str = "PIR Motion Array / Optical Turnstile"
    timestamp: Optional[str] = None
    status: SensorConnectionStatus = SensorConnectionStatus.AWAITING_HARDWARE
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "people_count": self.people_count,
            "density_category": self.density_category,
            "occupancy_rate": self.occupancy_rate,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "error_message": self.error_message
        }


@dataclass
class GPSReading:
    """Live GPS coordinate observation from browser or mobile client."""
    latitude: Optional[float]
    longitude: Optional[float]
    accuracy_m: Optional[float]
    timestamp: Optional[str]
    is_manual_override: bool = False
    status: SensorConnectionStatus = SensorConnectionStatus.DISCONNECTED
    nearest_landmark: Optional[str] = None
    distance_to_landmark_m: Optional[float] = None
    geofence_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy_m": self.accuracy_m,
            "timestamp": self.timestamp,
            "is_manual_override": self.is_manual_override,
            "status": self.status.value,
            "nearest_landmark": self.nearest_landmark,
            "distance_to_landmark_m": self.distance_to_landmark_m,
            "geofence_status": self.geofence_status
        }


@dataclass
class WearableReading:
    """Biometric telemetry from physical wearable or WESAD research benchmark."""
    heart_rate_bpm: Optional[float]
    rmssd_s: Optional[float]
    eda_microsiemens: Optional[float]
    skin_temp_c: Optional[float]
    device_model: str = "Empatica E4 / RespiBAN"
    subject_id: Optional[str] = None
    condition_name: Optional[str] = None
    timestamp: Optional[str] = None
    status: SensorConnectionStatus = SensorConnectionStatus.BENCHMARK_REPLAY
    is_live_stream: bool = False
    stress_probability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "heart_rate_bpm": self.heart_rate_bpm,
            "rmssd_s": self.rmssd_s,
            "eda_microsiemens": self.eda_microsiemens,
            "skin_temp_c": self.skin_temp_c,
            "device_model": self.device_model,
            "subject_id": self.subject_id,
            "condition_name": self.condition_name,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "is_live_stream": self.is_live_stream,
            "stress_probability": self.stress_probability
        }


@dataclass
class ModalityProvenance:
    """Compact provenance metadata card for dashboard display."""
    modality_name: str
    source_name: str
    source_type: DataSourceType
    status: SensorConnectionStatus
    model_name: Optional[str]
    prediction_type: str
    last_update_str: str
    is_fresh: bool = True
    confidence_pct: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality_name": self.modality_name,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "status": self.status.value,
            "model_name": self.model_name,
            "prediction_type": self.prediction_type,
            "last_update_str": self.last_update_str,
            "is_fresh": self.is_fresh,
            "confidence_pct": self.confidence_pct,
            "note": self.note
        }
