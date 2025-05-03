#!/usr/bin/env python3
"""
Wind & Temperature Tile Server - Configuration
Configuration settings for the tile generation process.
"""

import os
from multiprocessing import cpu_count

# Zoom level configuration
VECTOR_ZOOM_LEVELS = []  # No vector tiles - using only raster tiles
RASTER_ZOOM_LEVELS = [0, 1, 2, 3, 4, 5, 6, 7]  # Lower zoom levels use raster tiles

# File and directory settings
TILE_SIZE = 256  # Standard tile size in pixels
DATA_DIR = "data"
OUTPUT_DIR = "tiles"
WIND_FILE_PATTERN = "wind-*.GRIB"
TEMP_FILE_PATTERN = "temp-*.GRIB"

# Number of worker threads to use, defaults to the number of CPU cores
NUM_WORKERS = cpu_count()

# Fixed global temperature range in Celsius for consistent color normalization
MIN_TEMP_CELSIUS = -30.0
MAX_TEMP_CELSIUS = 40.0