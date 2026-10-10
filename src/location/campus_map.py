import math
import sys
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from src.location.landmarks import get_all_landmarks_list

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on the Earth in meters using Haversine formula.
    """
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return R * c

class CampusMapAnalyzer:
    """
    Analyzes GPS coordinates within or around BIT Mesra Campus, identifying
    nearest landmark objects, evaluating detection radii, and calculating spatial metrics.
    Guarantees a clean, standardized data contract.
    """

    def __init__(self, landmarks: Optional[List[Dict[str, Any]]] = None):
        self.landmarks = landmarks if landmarks is not None else get_all_landmarks_list()

    def analyze_gps(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        accuracy: Optional[float] = None,
        time_of_day_hour: float = 14.0,
        timestamp: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Analyzes a GPS position (lat, lon) and accuracy, returning a guaranteed standardized result schema.
        If lat or lon is None, returns status='ERROR' and nearest_landmark=None.
        """
        if lat is None or lon is None:
            return {
                "status": "ERROR",
                "message": "GPS coordinates not available.",
                "current_location": None,
                "nearest_landmark": None,
                "detected_landmark": None,
                "location_risk_score": None
            }

        if not self.landmarks:
            self.landmarks = get_all_landmarks_list()

        # Calculate Haversine distance to all configured landmarks
        min_dist = float('inf')
        nearest_landmark = self.landmarks[0]

        for lm in self.landmarks:
            dist = haversine_distance(lat, lon, lm["latitude"], lm["longitude"])
            if dist < min_dist:
                min_dist = dist
                nearest_landmark = lm

        # Distance to Security Gate
        security_gate = next((lm for lm in self.landmarks if lm["id"] in ["main_gate", "security_gate_1"]), nearest_landmark)
        dist_to_security = haversine_distance(lat, lon, security_gate["latitude"], security_gate["longitude"])

        # Check configurable detection radius
        radius = float(nearest_landmark.get("radius", 120.0))
        is_inside = min_dist <= radius
        detected_name = nearest_landmark["name"] if is_inside else "No nearby landmark"

        # Environmental lighting calculation
        is_night = (time_of_day_hour >= 19.0) or (time_of_day_hour <= 5.0)
        is_twilight = (time_of_day_hour in [18.0, 19.0, 5.0, 6.0])
        base_landmark_lighting = float(nearest_landmark.get("lighting_level", 7.0))

        if is_night:
            effective_lighting = base_landmark_lighting * 0.6
        elif is_twilight:
            effective_lighting = base_landmark_lighting * 0.8
        else:
            effective_lighting = min(10.0, base_landmark_lighting + 2.0)

        # Derived spatial isolation
        center_dist = haversine_distance(lat, lon, config.BIT_MESRA_CENTER["latitude"], config.BIT_MESRA_CENTER["longitude"])
        base_isolation = float(nearest_landmark.get("isolation_index", 0.3))
        spatial_isolation = min(1.0, max(0.0, base_isolation + (center_dist / 1000.0) * 0.2))
        crime_rate = float(nearest_landmark.get("crime_rate_index", 0.1))

        # Contextual location risk calculation
        base_risk = float(nearest_landmark.get("base_risk_score", 0.2))
        night_penalty = 0.25 if is_night else 0.0
        calculated_risk = min(1.0, max(0.0, base_risk + night_penalty * spatial_isolation))

        acc_val = float(accuracy) if accuracy is not None else 10.0
        is_low_accuracy = acc_val > 100.0
        status_code = "LOW_ACCURACY" if is_low_accuracy else "CONNECTED"
        message_str = "GPS accuracy is low. Waiting for a better location..." if is_low_accuracy else "Location analyzed successfully."

        return {
            "status": status_code,
            "message": message_str,
            "current_location": {
                "latitude": round(float(lat), 6),
                "longitude": round(float(lon), 6),
                "accuracy_meters": round(acc_val, 1),
                "timestamp": timestamp or config.get_campus_now().isoformat()
            },
            "nearest_landmark": {
                "id": nearest_landmark["id"],
                "name": nearest_landmark["name"],
                "latitude": float(nearest_landmark["latitude"]),
                "longitude": float(nearest_landmark["longitude"]),
                "distance_meters": round(min_dist, 1)
            },
            "detected_landmark": {
                "name": detected_name,
                "is_inside": is_inside,
                "radius_meters": radius
            },
            "location_risk_score": round(calculated_risk, 4),
            # Guaranteed backward compatibility fields
            "nearest_landmark_name": nearest_landmark["name"],
            "nearest_landmark_id": nearest_landmark["id"],
            "distance_to_nearest_landmark_m": round(min_dist, 1),
            "distance_to_security_m": round(dist_to_security, 1),
            "location_type": nearest_landmark.get("location_type", "general_campus"),
            "lighting_level": round(effective_lighting, 2),
            "isolation_index": round(spatial_isolation, 4),
            "crime_rate_index": round(crime_rate, 4),
            "time_of_day_hour": float(time_of_day_hour),
            "description": nearest_landmark.get("description", "")
        }

if __name__ == "__main__":
    analyzer = CampusMapAnalyzer()
    res = analyzer.analyze_gps(23.417810, 85.434655, accuracy=10.0, time_of_day_hour=22.0)
    print("Test Campus Analysis for Hostel 13 (23.417810, 85.434655):")
    print("  Status:", res["status"])
    print("  Nearest Landmark:", res["nearest_landmark"])
    print("  Detected Landmark:", res["detected_landmark"])
    print("  Location Risk Score:", res["location_risk_score"])

