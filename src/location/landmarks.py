"""
BIT Mesra Campus Landmarks Configuration & Coordinate Database.
Centralized location database with verified geographic coordinates, detection radii,
location categories, and prototype contextual risk parameters.
"""

from typing import Dict, Any, List

BIT_MESRA_CENTER = {
    "latitude": 23.41230,
    "longitude": 85.43990,
    "name": "BIT Mesra Main Building (Clock Tower)"
}

BIT_MESRA_LANDMARKS_DB: Dict[str, Dict[str, Any]] = {
    "main_building": {
        "id": "main_building",
        "name": "Main Building",
        "latitude": 23.412300,
        "longitude": 85.439900,
        "radius": 150.0,
        "location_type": "administrative_hub",
        "base_risk_score": 0.08,
        "lighting_level": 9.5,
        "isolation_index": 0.05,
        "crime_rate_index": 0.02,
        "distance_to_security_m": 20.0,
        "description": "Central administrative block (Clock Tower) with active 24/7 security & CCTV."
    },
    "rnd": {
        "id": "rnd",
        "name": "R&D",
        "latitude": 23.413154,
        "longitude": 85.441643,
        "radius": 150.0,
        "location_type": "academic_zone",
        "base_risk_score": 0.12,
        "lighting_level": 8.5,
        "isolation_index": 0.15,
        "crime_rate_index": 0.05,
        "distance_to_security_m": 50.0,
        "description": "R&D Building & Central Library academic research center."
    },
    "h1": {
        "id": "h1",
        "name": "H1",
        "latitude": 23.414209,
        "longitude": 85.440313,
        "radius": 120.0,
        "location_type": "residential_hostel",
        "base_risk_score": 0.25,
        "lighting_level": 7.5,
        "isolation_index": 0.25,
        "crime_rate_index": 0.10,
        "distance_to_security_m": 120.0,
        "description": "Hostel 1 residential sector."
    },
    "h2": {
        "id": "h2",
        "name": "H2",
        "latitude": 23.413707,
        "longitude": 85.438484,
        "radius": 120.0,
        "location_type": "residential_hostel",
        "base_risk_score": 0.25,
        "lighting_level": 7.5,
        "isolation_index": 0.25,
        "crime_rate_index": 0.10,
        "distance_to_security_m": 130.0,
        "description": "Hostel 2 residential sector."
    },
    "h3": {
        "id": "h3",
        "name": "H3",
        "latitude": 23.416120,
        "longitude": 85.439653,
        "radius": 120.0,
        "location_type": "residential_hostel",
        "base_risk_score": 0.28,
        "lighting_level": 7.0,
        "isolation_index": 0.30,
        "crime_rate_index": 0.12,
        "distance_to_security_m": 150.0,
        "description": "Hostel 3 residential sector."
    },
    "h4": {
        "id": "h4",
        "name": "H4",
        "latitude": 23.415602,
        "longitude": 85.437798,
        "radius": 120.0,
        "location_type": "residential_hostel",
        "base_risk_score": 0.28,
        "lighting_level": 7.0,
        "isolation_index": 0.30,
        "crime_rate_index": 0.12,
        "distance_to_security_m": 170.0,
        "description": "Hostel 4 residential sector."
    },
    "girls_hostel": {
        "id": "girls_hostel",
        "name": "Girls Hostel Area / H8-H9",
        "latitude": 23.416180,
        "longitude": 85.442107,
        "radius": 150.0,
        "location_type": "residential_hostel",
        "base_risk_score": 0.20,
        "lighting_level": 8.5,
        "isolation_index": 0.20,
        "crime_rate_index": 0.08,
        "distance_to_security_m": 80.0,
        "description": "Girls Hostel Complex / H8-H9 with dedicated security checkpoints."
    },
    "pmc": {
        "id": "pmc",
        "name": "PMC",
        "latitude": 23.416205,
        "longitude": 85.440121,
        "radius": 150.0,
        "location_type": "unlit_industrial_alley",
        "base_risk_score": 0.65,
        "lighting_level": 2.5,
        "isolation_index": 0.80,
        "crime_rate_index": 0.68,
        "distance_to_security_m": 480.0,
        "description": "PMC (Polytechnic & Medical Outpost)."
    },
    "h6": {
        "id": "h6",
        "name": "H6",
        "latitude": 23.422811,
        "longitude": 85.433202,
        "radius": 120.0,
        "location_type": "residential_hostel",
        "base_risk_score": 0.30,
        "lighting_level": 6.8,
        "isolation_index": 0.35,
        "crime_rate_index": 0.15,
        "distance_to_security_m": 200.0,
        "description": "Hostel 6 residential sector."
    },
    "h7": {
        "id": "h7",
        "name": "H7",
        "latitude": 23.424604,
        "longitude": 85.432130,
        "radius": 120.0,
        "location_type": "residential_hostel",
        "base_risk_score": 0.32,
        "lighting_level": 6.5,
        "isolation_index": 0.40,
        "crime_rate_index": 0.18,
        "distance_to_security_m": 230.0,
        "description": "Hostel 7 residential sector."
    },
    "dispensary": {
        "id": "dispensary",
        "name": "Dispensary",
        "latitude": 23.410500,
        "longitude": 85.438800,
        "radius": 100.0,
        "location_type": "medical_center",
        "base_risk_score": 0.15,
        "lighting_level": 8.5,
        "isolation_index": 0.18,
        "crime_rate_index": 0.05,
        "distance_to_security_m": 100.0,
        "description": "Dispensary / Health Centre with night duty staff."
    },
    "h10": {
        "id": "h10",
        "name": "H10",
        "latitude": 23.419115,
        "longitude": 85.434912,
        "radius": 120.0,
        "location_type": "isolated_hostel_boundary",
        "base_risk_score": 0.45,
        "lighting_level": 4.5,
        "isolation_index": 0.70,
        "crime_rate_index": 0.40,
        "distance_to_security_m": 400.0,
        "description": "Hostel 10 perimeter sector."
    },
    "h11": {
        "id": "h11",
        "name": "H11",
        "latitude": 23.418032,
        "longitude": 85.435418,
        "radius": 120.0,
        "location_type": "isolated_forest_boundary",
        "base_risk_score": 0.50,
        "lighting_level": 3.5,
        "isolation_index": 0.78,
        "crime_rate_index": 0.50,
        "distance_to_security_m": 450.0,
        "description": "Hostel 11 residential sector."
    },
    "h12": {
        "id": "h12",
        "name": "H12",
        "latitude": 23.418919,
        "longitude": 85.434311,
        "radius": 120.0,
        "location_type": "isolated_forest_boundary",
        "base_risk_score": 0.55,
        "lighting_level": 2.5,
        "isolation_index": 0.85,
        "crime_rate_index": 0.60,
        "distance_to_security_m": 520.0,
        "description": "Hostel 12 sector adjacent to campus perimeter."
    },
    "h13": {
        "id": "h13",
        "name": "H13",
        "latitude": 23.417810,
        "longitude": 85.434655,
        "radius": 120.0,
        "location_type": "isolated_forest_boundary",
        "base_risk_score": 0.60,
        "lighting_level": 1.5,
        "isolation_index": 0.92,
        "crime_rate_index": 0.75,
        "distance_to_security_m": 600.0,
        "description": "Hostel 13 sector with verified OSM geographic coordinates."
    },
    "sports_complex": {
        "id": "sports_complex",
        "name": "Sports Complex",
        "latitude": 23.425020,
        "longitude": 85.434623,
        "radius": 200.0,
        "location_type": "open_recreational",
        "base_risk_score": 0.35,
        "lighting_level": 4.5,
        "isolation_index": 0.55,
        "crime_rate_index": 0.25,
        "distance_to_security_m": 250.0,
        "description": "Sports Complex & Gym Ground."
    },
    "main_gate": {
        "id": "main_gate",
        "name": "Main Gate",
        "latitude": 23.405500,
        "longitude": 85.435000,
        "radius": 150.0,
        "location_type": "security_checkpoint",
        "base_risk_score": 0.05,
        "lighting_level": 9.8,
        "isolation_index": 0.02,
        "crime_rate_index": 0.01,
        "distance_to_security_m": 0.0,
        "description": "24/7 guarded Main Gate (Security Checkpoint 1)."
    }
}

def get_all_landmarks_list() -> List[Dict[str, Any]]:
    """Returns all landmarks as a list of dictionaries."""
    return list(BIT_MESRA_LANDMARKS_DB.values())

def get_landmark_by_id(landmark_id: str) -> Dict[str, Any]:
    """Retrieves landmark configuration by landmark ID or returns default."""
    return BIT_MESRA_LANDMARKS_DB.get(landmark_id, BIT_MESRA_LANDMARKS_DB["main_building"])
