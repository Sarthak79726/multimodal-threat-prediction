"""
BIT Mesra Campus Landmarks Configuration & Coordinate Database.
Centralized location database with verified geographic coordinates, detection radii,
location categories, and prototype contextual risk parameters.
"""

from src.location.landmarks import (
    BIT_MESRA_CENTER,
    BIT_MESRA_LANDMARKS_DB,
    get_all_landmarks_list,
    get_landmark_by_id
)

__all__ = [
    "BIT_MESRA_CENTER",
    "BIT_MESRA_LANDMARKS_DB",
    "get_all_landmarks_list",
    "get_landmark_by_id"
]
