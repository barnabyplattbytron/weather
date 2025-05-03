#!/usr/bin/env python3
"""
Wind & Temperature Tile Server
Converts GRIB-format weather data (wind & temperature) into XYZ-based tiles.
Uses raster tiles for lower zoom levels and vector tiles for higher zoom levels.
"""

import os
import time
import argparse

# Import configuration
from config import (
    NUM_WORKERS, RASTER_ZOOM_LEVELS,
    DATA_DIR, OUTPUT_DIR, VECTOR_ZOOM_LEVELS
)

# Import utility functions
from utils import ensure_dir_exists, find_matching_files

# Import core processing functions
from grib_processing import process_grib_files

def main():
    """Main function to process GRIB files and generate tiles."""
    # Update globals from config module that might be changed by command line arguments
    global NUM_WORKERS, RASTER_ZOOM_LEVELS, DATA_DIR, OUTPUT_DIR

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate weather tiles from GRIB files')
    parser.add_argument('--threads', type=int, default=NUM_WORKERS,
                        help=f'Number of worker threads (default: {NUM_WORKERS})')
    parser.add_argument('--zoom', type=str, default=','.join(map(str, RASTER_ZOOM_LEVELS)),
                        help=f'Comma-separated list of zoom levels (default: {",".join(map(str, RASTER_ZOOM_LEVELS))})')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help=f'Directory containing GRIB files (default: {DATA_DIR})')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help=f'Directory to save tile files (default: {OUTPUT_DIR})')

    args = parser.parse_args()

    # Update configuration based on arguments
    NUM_WORKERS = args.threads
    RASTER_ZOOM_LEVELS = list(map(int, args.zoom.split(',')))
    DATA_DIR = args.data_dir
    OUTPUT_DIR = args.output_dir

    print("🌪️ Wind & Temperature Tile Server")
    print("=================================")
    print(f"Using {NUM_WORKERS} worker threads")
    print(f"Generating tiles for zoom levels: {RASTER_ZOOM_LEVELS}")

    # Start timing
    total_start_time = time.time()

    # Ensure output directory exists
    ensure_dir_exists(OUTPUT_DIR)

    # Find matching wind and temperature files
    file_pairs = find_matching_files(DATA_DIR, "wind-*.GRIB", "temp-*.GRIB")

    if not file_pairs:
        print(f"No matching GRIB files found in {DATA_DIR}")
        return

    # Process each pair of files
    for wind_file, temp_file in file_pairs:
        file_start_time = time.time()
        print(f"\nProcessing wind file: {wind_file}")
        if temp_file:
            print(f"Processing temperature file: {temp_file}")
        else:
            print("No matching temperature file found")

        process_grib_files(wind_file, temp_file, VECTOR_ZOOM_LEVELS, RASTER_ZOOM_LEVELS, OUTPUT_DIR, NUM_WORKERS)

        file_elapsed_time = time.time() - file_start_time
        print(f"File processing time: {file_elapsed_time:.2f} seconds")

    total_elapsed_time = time.time() - total_start_time
    print("\n✅ Tile generation complete!")
    print(f"Total processing time: {total_elapsed_time:.2f} seconds ({total_elapsed_time / 60:.2f} minutes)")

if __name__ == "__main__":
    main()