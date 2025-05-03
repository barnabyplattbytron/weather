#!/usr/bin/env python3
"""
Wind & Temperature Tile Server - Utilities
Utility functions for the tile generation process.
"""

import os
import math
import numpy as np
from pyproj import Transformer

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def calculate_wind_speed_direction(u, v):
    """Calculate wind speed and direction from U and V components."""
    speed = np.sqrt(u**2 + v**2)
    # Meteorological direction (270 - arctan2(v, u))
    direction = (270 - np.arctan2(v, u) * 180 / np.pi) % 360
    return speed, direction

def lat_lon_to_web_mercator(lat, lon):
    """Convert latitude/longitude to Web Mercator (EPSG:3857)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transformer.transform(lon, lat)

def find_matching_files(directory, wind_pattern, temp_pattern):
    """Find matching wind and temperature files."""
    pairs = []

    # Get all wind files
    wind_files = []
    for entry in os.listdir(directory):
        if entry.startswith("wind-") and entry.endswith(".GRIB"):
            wind_files.append(entry)

    # For each wind file, find corresponding temperature file
    for wind_file in wind_files:
        # Extract level from wind file (e.g., "006_012")
        level = wind_file.replace("wind-", "").replace(".GRIB", "")
        temp_file = f"temp-{level}.GRIB"

        wind_path = os.path.join(directory, wind_file)
        temp_path = os.path.join(directory, temp_file)

        # Check if temperature file exists
        if os.path.exists(temp_path):
            pairs.append((wind_path, temp_path))
        else:
            # If no matching temperature file, use wind file alone
            pairs.append((wind_path, None))

    return pairs