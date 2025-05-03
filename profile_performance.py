#!/usr/bin/env python3
"""
Profile the performance of the weather tile server application.
This script will run the tile generation process with profiling enabled
to identify bottlenecks in the code.
"""

import cProfile
import pstats
import io
import os
import time
import sys
from pstats import SortKey
import shutil

# Import the main processing functions
from grib_processing import process_grib_files
from utils import find_matching_files, ensure_dir_exists
from config import VECTOR_ZOOM_LEVELS, RASTER_ZOOM_LEVELS, DATA_DIR, OUTPUT_DIR, NUM_WORKERS

def profile_tile_generation():
    """Run the tile generation process with profiling enabled."""
    # Ensure output directory exists
    ensure_dir_exists(OUTPUT_DIR)

    # Find matching wind and temperature files
    file_pairs = find_matching_files(DATA_DIR, "wind-*.GRIB", "temp-*.GRIB")

    if not file_pairs:
        print(f"No matching GRIB files found in {DATA_DIR}")
        return

    # Take only the first pair for profiling to keep it manageable
    wind_file, temp_file = file_pairs[0]

    print(f"Profiling with wind file: {wind_file}")
    print(f"Profiling with temperature file: {temp_file}")

    # Create a restricted zoom level set for faster profiling
    vector_zoom = VECTOR_ZOOM_LEVELS[:1]  # Just the first zoom level
    raster_zoom = [3]  # Profile only zoom level 3

    print(f"Using zoom levels: Vector {vector_zoom}, Raster {raster_zoom}")

    # Delete existing tiles for 250hPa level to force reprocessing
    test_level = "250hPa"
    wind_path = os.path.join(OUTPUT_DIR, "wind", test_level)
    temp_path = os.path.join(OUTPUT_DIR, "temp", test_level)

    if os.path.exists(wind_path):
        print(f"Removing existing wind tiles for {test_level} to force processing")
        shutil.rmtree(wind_path)

    if os.path.exists(temp_path):
        print(f"Removing existing temperature tiles for {test_level} to force processing")
        shutil.rmtree(temp_path)

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the actual processing
    process_grib_files(wind_file, temp_file, vector_zoom, raster_zoom, OUTPUT_DIR, NUM_WORKERS)

    # Stop profiling
    profiler.disable()

    # Output results sorted by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(30)  # Print top 30 functions by cumulative time
    print(s.getvalue())

    # Also output results sorted by total time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.TIME)
    ps.print_stats(30)  # Print top 30 functions by total time
    print("\n--- Sorted by total time ---")
    print(s.getvalue())

    # Save results to a file
    with open('profile_results.txt', 'w') as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.sort_stats(SortKey.CUMULATIVE)
        ps.print_stats()

    print(f"Full profile results saved to profile_results.txt")

if __name__ == "__main__":
    start_time = time.time()
    profile_tile_generation()
    elapsed = time.time() - start_time
    print(f"Total profiling time: {elapsed:.2f} seconds")