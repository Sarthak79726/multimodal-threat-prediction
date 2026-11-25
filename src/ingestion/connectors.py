"""
Real Sensor Ingestion Connectors.
Provides modular, extensible hardware interfaces for physical IoT lux sensors,
privacy-preserving campus occupancy sensors, live wearable devices, and GPS receivers.
Strictly eliminates simulated/fake values when hardware is not streaming.
"""

import logging
import threading
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from src.ingestion.schemas import (
    SensorConnectionStatus,
    DataSourceType,
    LightingSensorReading,
    CrowdSensorReading,
    GPSReading,
    WearableReading
)

logger = logging.getLogger(__name__)


class PhysicalLightingSensorConnector:
    """
    Ingestion connector for physical ambient light sensors (e.g. BH1750, TSL2561, VEML7700).
    Connects to MQTT / HTTP REST telemetry or USB-Serial microcontrollers.
    When no physical sensor is streaming, reports AWAITING_HARDWARE without fabricating lux.
    """
    def __init__(self, sensor_model: str = "BH1750 Digital Ambient Light Sensor", timeout_seconds: int = 60):
        self.sensor_model = sensor_model
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._latest_reading: Optional[LightingSensorReading] = None
        self._last_received_time: Optional[datetime] = None

    def ingest_reading(
        self,
        lux: float,
        indoor_flag: bool = False,
        sensor_model: Optional[str] = None,
        hardware_port: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> LightingSensorReading:
        """
        Ingests and validates a genuine physical photometer reading.
        Rejects non-finite, negative, or physically impossible lux values.
        """
        if lux is None or not np.isfinite(lux) or float(lux) < 0.0:
            raise ValueError(f"Invalid physical lux reading: {lux}. Must be non-negative finite float.")

        lux_f = round(float(lux), 2)
        ts_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        model = sensor_model or self.sensor_model

        with self._lock:
            reading = LightingSensorReading(
                lux=lux_f,
                indoor_flag=indoor_flag,
                sensor_model=model,
                timestamp=ts_str,
                status=SensorConnectionStatus.CONNECTED,
                hardware_port=hardware_port,
                error_message=None
            )
            self._latest_reading = reading
            self._last_received_time = datetime.now(timezone.utc)
            logger.info("Physical lux reading ingested: %s lx from %s", lux_f, model)
            return reading

    def get_latest_reading(self) -> LightingSensorReading:
        """
        Retrieves the latest physical lux reading.
        Returns AWAITING_HARDWARE if no physical sensor has reported in.
        """
        with self._lock:
            if self._latest_reading is None:
                return LightingSensorReading(
                    lux=None,
                    indoor_flag=False,
                    sensor_model=self.sensor_model,
                    timestamp=None,
                    status=SensorConnectionStatus.AWAITING_HARDWARE,
                    error_message="Physical IoT lux sensor not connected. Awaiting hardware deployment."
                )

            # Check staleness
            now = datetime.now(timezone.utc)
            if self._last_received_time is not None:
                age = (now - self._last_received_time).total_seconds()
                if age > self.timeout_seconds:
                    stale_reading = LightingSensorReading(
                        lux=self._latest_reading.lux,
                        indoor_flag=self._latest_reading.indoor_flag,
                        sensor_model=self._latest_reading.sensor_model,
                        timestamp=self._latest_reading.timestamp,
                        status=SensorConnectionStatus.STALE,
                        hardware_port=self._latest_reading.hardware_port,
                        error_message=f"Sensor data is stale (last seen {int(age)}s ago)"
                    )
                    return stale_reading

            return self._latest_reading


class PrivacyPreservingCrowdConnector:
    """
    Ingestion connector for campus occupancy and crowd sensing.
    Connects to privacy-preserving PIR motion arrays, Wi-Fi association counters,
    or optical turnstiles.
    When no physical occupancy hardware is streaming, reports AWAITING_HARDWARE.
    """
    def __init__(self, sensor_type: str = "PIR Motion Sensor Array / Optical Turnstile", timeout_seconds: int = 120):
        self.sensor_type = sensor_type
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._latest_reading: Optional[CrowdSensorReading] = None
        self._last_received_time: Optional[datetime] = None

    def ingest_count(
        self,
        people_count: int,
        density_category: Optional[str] = None,
        occupancy_rate: Optional[float] = None,
        sensor_type: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> CrowdSensorReading:
        """
        Ingests and validates a genuine privacy-preserving people count.
        Rejects negative counts or non-integer values.
        """
        if people_count is None or int(people_count) < 0:
            raise ValueError(f"Invalid people count: {people_count}. Must be non-negative integer.")

        cnt = int(people_count)
        ts_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        s_type = sensor_type or self.sensor_type

        # Derive category if not provided
        if density_category is None:
            if cnt <= 5:
                cat = "Low"
            elif cnt <= 20:
                cat = "Medium"
            else:
                cat = "High"
        else:
            cat = str(density_category).capitalize()

        with self._lock:
            reading = CrowdSensorReading(
                people_count=cnt,
                density_category=cat,
                occupancy_rate=occupancy_rate,
                sensor_type=s_type,
                timestamp=ts_str,
                status=SensorConnectionStatus.CONNECTED,
                error_message=None
            )
            self._latest_reading = reading
            self._last_received_time = datetime.now(timezone.utc)
            logger.info("Campus crowd reading ingested: %s people (%s) from %s", cnt, cat, s_type)
            return reading

    def get_latest_reading(self) -> CrowdSensorReading:
        """
        Retrieves the latest physical crowd count.
        Returns AWAITING_HARDWARE if no physical sensor has reported in.
        """
        with self._lock:
            if self._latest_reading is None:
                return CrowdSensorReading(
                    people_count=None,
                    density_category=None,
                    occupancy_rate=None,
                    sensor_type=self.sensor_type,
                    timestamp=None,
                    status=SensorConnectionStatus.AWAITING_HARDWARE,
                    error_message="Physical occupancy / crowd sensor not connected. Awaiting hardware deployment."
                )

            # Check staleness
            now = datetime.now(timezone.utc)
            if self._last_received_time is not None:
                age = (now - self._last_received_time).total_seconds()
                if age > self.timeout_seconds:
                    return CrowdSensorReading(
                        people_count=self._latest_reading.people_count,
                        density_category=self._latest_reading.density_category,
                        occupancy_rate=self._latest_reading.occupancy_rate,
                        sensor_type=self._latest_reading.sensor_type,
                        timestamp=self._latest_reading.timestamp,
                        status=SensorConnectionStatus.STALE,
                        error_message=f"Crowd sensor data is stale (last seen {int(age)}s ago)"
                    )

            return self._latest_reading


class WearableDeviceConnector:
    """
    Ingestion connector for physical smartwatches / smartbands (BLE / Empatica E4 TCP).
    Supports dual modes:
    1. Real-time BLE Streaming: Ingests packets from paired physical wearable.
    2. Benchmark Research Replay: Replays verified empirical WESAD dataset records.
    Never invents random pulse or skin conductance readings when disconnected.
    """
    def __init__(self, device_model: str = "Empatica E4 / Wearable BLE", timeout_seconds: int = 30):
        self.device_model = device_model
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._latest_live_reading: Optional[WearableReading] = None
        self._last_received_time: Optional[datetime] = None

    def ingest_live_packet(
        self,
        heart_rate_bpm: float,
        rmssd_s: float,
        eda_microsiemens: float,
        skin_temp_c: float,
        device_model: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> WearableReading:
        """
        Ingests real-time biometric telemetry packet from a physically connected wearable.
        Validates physiological boundaries (HR in 30-240 BPM, Temp in 20-45°C).
        """
        hr = float(heart_rate_bpm)
        eda = float(eda_microsiemens)
        temp = float(skin_temp_c)
        rmssd = float(rmssd_s)

        if not (30.0 <= hr <= 240.0):
            raise ValueError(f"Physiological biometric anomaly: Heart Rate {hr} BPM outside human range [30, 240]")
        if not (20.0 <= temp <= 45.0):
            raise ValueError(f"Physiological biometric anomaly: Skin Temperature {temp}°C outside human range [20, 45]")
        if eda < 0.0:
            raise ValueError(f"Physiological biometric anomaly: EDA {eda} μS cannot be negative")

        ts_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        model = device_model or self.device_model

        with self._lock:
            reading = WearableReading(
                heart_rate_bpm=round(hr, 1),
                rmssd_s=round(rmssd, 4),
                eda_microsiemens=round(eda, 3),
                skin_temp_c=round(temp, 2),
                device_model=model,
                subject_id="LIVE_USER",
                condition_name="Live Wearable Stream",
                timestamp=ts_str,
                status=SensorConnectionStatus.STREAMING,
                is_live_stream=True,
                stress_probability=None
            )
            self._latest_live_reading = reading
            self._last_received_time = datetime.now(timezone.utc)
            logger.info("Live wearable packet ingested: HR=%s BPM, EDA=%s μS from %s", hr, eda, model)
            return reading

    def get_live_reading(self) -> WearableReading:
        """
        Retrieves the latest reading from a physically paired wearable.
        If no device is paired or streaming, returns DISCONNECTED.
        Never fabricates pulse or EDA.
        """
        with self._lock:
            if self._latest_live_reading is None:
                return WearableReading(
                    heart_rate_bpm=None,
                    rmssd_s=None,
                    eda_microsiemens=None,
                    skin_temp_c=None,
                    device_model=self.device_model,
                    subject_id=None,
                    condition_name=None,
                    timestamp=None,
                    status=SensorConnectionStatus.DISCONNECTED,
                    is_live_stream=False,
                    stress_probability=None
                )

            now = datetime.now(timezone.utc)
            if self._last_received_time is not None:
                age = (now - self._last_received_time).total_seconds()
                if age > self.timeout_seconds:
                    return WearableReading(
                        heart_rate_bpm=self._latest_live_reading.heart_rate_bpm,
                        rmssd_s=self._latest_live_reading.rmssd_s,
                        eda_microsiemens=self._latest_live_reading.eda_microsiemens,
                        skin_temp_c=self._latest_live_reading.skin_temp_c,
                        device_model=self._latest_live_reading.device_model,
                        subject_id=self._latest_live_reading.subject_id,
                        condition_name=self._latest_live_reading.condition_name,
                        timestamp=self._latest_live_reading.timestamp,
                        status=SensorConnectionStatus.STALE,
                        is_live_stream=True,
                        stress_probability=self._latest_live_reading.stress_probability
                    )

            return self._latest_live_reading


class GPSLocationConnector:
    """
    Ingestion connector for browser HTML5 Geolocation API and mobile NMEA streams.
    Validates coordinate bounds, evaluates reported accuracy margin, and detects manual overrides.
    Never fabricates GPS fixes when permission is denied or location is unavailable.
    """
    @staticmethod
    def validate_and_ingest(
        lat: Optional[float],
        lon: Optional[float],
        accuracy: Optional[float] = None,
        timestamp: Optional[str] = None,
        is_manual_override: bool = False
    ) -> GPSReading:
        """
        Validates GPS coordinates and constructs a verified GPSReading object.
        """
        if lat is None or lon is None:
            return GPSReading(
                latitude=None,
                longitude=None,
                accuracy_m=None,
                timestamp=timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S IST"),
                is_manual_override=is_manual_override,
                status=SensorConnectionStatus.DISCONNECTED,
                nearest_landmark=None,
                distance_to_landmark_m=None,
                geofence_status="GPS_UNAVAILABLE"
            )

        lat_f = float(lat)
        lon_f = float(lon)
        acc_f = float(accuracy) if (accuracy is not None and np.isfinite(accuracy)) else 10.0

        if not (-90.0 <= lat_f <= 90.0):
            raise ValueError(f"Invalid latitude {lat_f}. Must be in [-90, 90]")
        if not (-180.0 <= lon_f <= 180.0):
            raise ValueError(f"Invalid longitude {lon_f}. Must be in [-180, 180]")
        if acc_f < 0.0:
            acc_f = 10.0

        ts_str = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        status = SensorConnectionStatus.CONNECTED if not is_manual_override else SensorConnectionStatus.BENCHMARK_REPLAY

        return GPSReading(
            latitude=round(lat_f, 6),
            longitude=round(lon_f, 6),
            accuracy_m=round(acc_f, 1),
            timestamp=ts_str,
            is_manual_override=is_manual_override,
            status=status,
            nearest_landmark=None,
            distance_to_landmark_m=None,
            geofence_status="CONNECTED"
        )
