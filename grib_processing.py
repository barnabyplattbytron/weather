#!/usr/bin/env python3
"""
Wind & Temperature Tile Server - GRIB Processing
Main logic for processing GRIB files and generating tiles.
"""

import os
import time
import xarray as xr
import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from config import VECTOR_ZOOM_LEVELS, RASTER_ZOOM_LEVELS
from data_processing import (
    read_grib_file, get_wind_components, get_temperature,
    extract_flight_level, create_geojson_features,
    assign_features_to_tiles, save_geojson_tiles
)
from tile_generator import generate_raster_tiles
from utils import ensure_dir_exists

def process_grib_files(wind_file, temp_file, vector_zoom_levels, raster_zoom_levels, output_dir, num_workers, tile_type='all'):
    """
    Process GRIB files and create vector and raster tiles for each pressure level.

    Args:
        wind_file: Path to the wind GRIB file
        temp_file: Path to the temperature GRIB file (optional)
        vector_zoom_levels: List of zoom levels for vector tiles
        raster_zoom_levels: List of zoom levels for raster tiles
        output_dir: Directory to save output tiles
        num_workers: Number of workers for parallel processing
        tile_type: Type of tiles to generate ('wind', 'temp', 'streamline', or 'all')
    """
    # Read wind data
    print(f"\nProcessing wind file: {wind_file}")
    try:
        # First try to open the file directly with xarray - this gives us a single dataset
        wind_ds = read_grib_file(wind_file)
        if wind_ds is None:
            print(f"Unable to read wind file: {wind_file}")
            return

        # Check if there's a vertical level dimension in the dataset
        level_dim = None
        level_values = []

        # Check for common level dimension names
        for level_name in ['level', 'isobaricInhPa', 'pressure', 'isobaricInPa']:
            if level_name in wind_ds.dims:
                level_dim = level_name
                level_values = wind_ds[level_name].values
                break
            elif level_name in wind_ds.coords:
                level_dim = level_name
                level_values = wind_ds[level_name].values
                break

        # If no level dimension found, process as a single level
        if level_dim is None or len(level_values) <= 1:
            # Just process as a single level (could be surface data)
            print("Processing as a single level dataset")

            # Extract flight level
            flight_level = extract_flight_level(wind_ds)
            print(f"Processing data for flight level: {flight_level}")

            # Get U/V wind components
            try:
                u_data, v_data = get_wind_components(wind_ds)
            except ValueError as e:
                print(f"Error extracting wind components: {e}")
                return

            # Get latitude and longitude arrays
            try:
                lats = wind_ds.latitude.values
                lons = wind_ds.longitude.values
            except AttributeError:
                # Some GRIB files might use different names
                lat_vars = ['latitude', 'lat']
                lon_vars = ['longitude', 'lon']

                lat_var = next((var for var in lat_vars if hasattr(wind_ds, var)), None)
                lon_var = next((var for var in lon_vars if hasattr(wind_ds, var)), None)

                if lat_var is None or lon_var is None:
                    print("Could not find latitude/longitude coordinates in dataset")
                    return

                lats = getattr(wind_ds, lat_var).values
                lons = getattr(wind_ds, lon_var).values

            # Initialize temp_data to None
            temp_data = None

            # Read temperature data if available
            if temp_file and os.path.exists(temp_file):
                temp_ds = read_grib_file(temp_file)
                if temp_ds is not None:
                    try:
                        temp_data = get_temperature(temp_ds)
                    except ValueError as e:
                        print(f"Error extracting temperature: {e}")

            # Process the data for this level
            process_single_level(wind_ds, u_data, v_data, temp_data, lats, lons,
                               flight_level, vector_zoom_levels, raster_zoom_levels, output_dir, num_workers, tile_type)

        else:
            # Process each level separately in parallel
            print(f"Found {len(level_values)} levels: {level_values}")

            # Cache temperature data if available to avoid re-reading for each level
            temp_ds = None
            if temp_file and os.path.exists(temp_file):
                temp_ds = read_grib_file(temp_file)
                if temp_ds is None:
                    print(f"Warning: Unable to read temperature file: {temp_file}")

            # Prepare tasks for parallel processing
            tasks = []
            for level_val in level_values:
                flight_level = f"{int(level_val)}hPa"

                # Only skip if we're generating ALL tiles AND both wind and temp directories exist
                # If we're generating a specific tile type, always process the level
                if tile_type == 'all' and \
                   os.path.exists(os.path.join(output_dir, "wind", flight_level)) and \
                   os.path.exists(os.path.join(output_dir, "temp", flight_level)):
                    print(f"Skipping already processed level: {flight_level}")
                    continue

                # If generating a specific tile type, check if that type exists
                if tile_type == 'wind' and os.path.exists(os.path.join(output_dir, "wind", flight_level)):
                    print(f"Skipping level {flight_level} - wind tiles already exist")
                    continue

                if tile_type == 'temp' and os.path.exists(os.path.join(output_dir, "temp", flight_level)):
                    print(f"Skipping level {flight_level} - temperature tiles already exist")
                    continue

                if tile_type == 'streamline' and os.path.exists(os.path.join(output_dir, "streamline", flight_level)):
                    # Even if the directory exists, we're going to force regeneration for streamline tiles
                    # by not skipping it here
                    pass

                # Select data for this level
                level_wind_ds = wind_ds.sel({level_dim: level_val})

                # Get U/V wind components for this level
                try:
                    u_data, v_data = get_wind_components(level_wind_ds)
                except ValueError as e:
                    print(f"Error extracting wind components for level {flight_level}: {e}")
                    continue

                # Get latitude and longitude arrays
                try:
                    lats = level_wind_ds.latitude.values
                    lons = level_wind_ds.longitude.values
                except AttributeError:
                    # Some GRIB files might use different names
                    lat_vars = ['latitude', 'lat']
                    lon_vars = ['longitude', 'lon']

                    lat_var = next((var for var in lat_vars if hasattr(level_wind_ds, var)), None)
                    lon_var = next((var for var in lon_vars if hasattr(level_wind_ds, var)), None)

                    if lat_var is None or lon_var is None:
                        print(f"Could not find latitude/longitude coordinates for level {flight_level}")
                        continue

                    lats = getattr(level_wind_ds, lat_var).values
                    lons = getattr(level_wind_ds, lon_var).values

                # Initialize temp_data to None
                temp_data = None

                # Read temperature data if available
                if temp_ds is not None:
                    try:
                        # Try to select the same level from temperature data
                        if level_dim in temp_ds.dims or level_dim in temp_ds.coords:
                            level_temp_ds = temp_ds.sel({level_dim: level_val})
                            temp_data = get_temperature(level_temp_ds)
                        else:
                            temp_data = get_temperature(temp_ds)
                    except ValueError as e:
                        print(f"Error extracting temperature for level {flight_level}: {e}")

                # Add this level's processing task to the queue
                tasks.append((level_wind_ds, u_data, v_data, temp_data, lats, lons,
                             flight_level, vector_zoom_levels, raster_zoom_levels, output_dir, num_workers, tile_type))

            # Process levels in parallel
            if tasks:
                print(f"Processing {len(tasks)} flight levels in parallel with {min(len(tasks), num_workers)} processes")
                start_time = time.time()

                # Use ProcessPoolExecutor for CPU-bound tasks to bypass the GIL
                # and get true parallel processing on multiple cores
                with ProcessPoolExecutor(max_workers=min(len(tasks), num_workers)) as executor:
                    # Submit all tasks to the executor
                    # We're now using pickle to serialize the data between processes
                    futures = {executor.submit(process_single_level, *task): task[6] for task in tasks}

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        flight_level = futures[future]
                        try:
                            future.result()
                            print(f"✓ Completed processing for flight level {flight_level}")
                        except Exception as e:
                            print(f"✗ Error processing flight level {flight_level}: {e}")
                            import traceback
                            traceback.print_exc()

                elapsed_time = time.time() - start_time
                print(f"Processed {len(tasks)} flight levels in {elapsed_time:.1f} seconds ({len(tasks) / elapsed_time:.1f} levels/sec)")

    except Exception as e:
        print(f"Error processing file {wind_file}: {e}")
        import traceback
        traceback.print_exc()

def process_single_level(wind_ds, u_data, v_data, temp_data, lats, lons, flight_level,
                        vector_zoom_levels, raster_zoom_levels, output_dir, num_workers, tile_type='all'):
    """Process a single pressure level and generate tiles.

    Args:
        wind_ds: Wind dataset
        u_data: U component data
        v_data: V component data
        temp_data: Temperature data (optional)
        lats: Latitude values
        lons: Longitude values
        flight_level: Flight level string (e.g., '250hPa')
        vector_zoom_levels: List of zoom levels for vector tiles
        raster_zoom_levels: List of zoom levels for raster tiles
        output_dir: Directory to save output tiles
        num_workers: Number of worker processes/threads
        tile_type: Type of tiles to generate ('wind', 'temp', 'streamline', or 'all')
    """
    start_time = time.time()

    # Check if longitudes are in 0-360 range and convert to -180 to 180 if needed
    if np.min(lons) >= 0 and np.max(lons) > 180:
        print("Converting longitudes from 0-360 range to -180 to 180 if needed")
        # Create a copy of the longitude array
        lons_normalized = np.copy(lons)
        # Convert values > 180 to negative values
        lons_normalized[lons > 180] -= 360

        # We need to also reorder the data arrays to match the new longitude order
        if np.any(lons > 180):
            # Find where the split should occur (180 degrees)
            split_idx = np.searchsorted(lons, 180)

            # Reorder longitude array
            lons = np.concatenate([lons_normalized[split_idx:], lons_normalized[:split_idx]])

            # Reorder data arrays to match
            u_data_values = u_data.values
            v_data_values = v_data.values
            u_data_reordered = np.concatenate([u_data_values[:, split_idx:], u_data_values[:, :split_idx]], axis=1)
            v_data_reordered = np.concatenate([v_data_values[:, split_idx:], v_data_values[:, :split_idx]], axis=1)

            # Create new DataArray objects with reordered data
            u_data = xr.DataArray(u_data_reordered, dims=u_data.dims,
                                coords={u_data.dims[0]: lats, u_data.dims[1]: lons})
            v_data = xr.DataArray(v_data_reordered, dims=v_data.dims,
                                coords={v_data.dims[0]: lats, v_data.dims[1]: lons})

            # Also reorder temperature data if available
            if temp_data is not None:
                temp_data_values = temp_data.values
                temp_data_reordered = np.concatenate([temp_data_values[:, split_idx:], temp_data_values[:, :split_idx]], axis=1)
                temp_data = xr.DataArray(temp_data_reordered, dims=temp_data.dims,
                                      coords={temp_data.dims[0]: lats, temp_data.dims[1]: lons})
        else:
            # Just use the normalized longitudes without reordering
            lons = lons_normalized

    # Create required directories based on tile type
    if tile_type in ['wind', 'all']:
        ensure_dir_exists(os.path.join(output_dir, "wind", flight_level))
    if tile_type in ['temp', 'all'] and temp_data is not None:
        ensure_dir_exists(os.path.join(output_dir, "temp", flight_level))
    if tile_type in ['streamline', 'all']:
        ensure_dir_exists(os.path.join(output_dir, "streamline", flight_level))

    # ---- Generate Vector Tiles ----
    vector_start = time.time()
    if vector_zoom_levels and tile_type in ['wind', 'temp', 'all']:
        print(f"Generating vector tiles for zoom levels {vector_zoom_levels}...")
        # Create GeoJSON features
        features = create_geojson_features(lats, lons, u_data.values, v_data.values,
                                        temp_data.values if temp_data is not None else None)

        # Assign features to tiles
        tiles_features = assign_features_to_tiles(features, vector_zoom_levels)

        # Save GeoJSON tiles
        save_geojson_tiles(tiles_features, flight_level, output_dir)

        vector_time = time.time() - vector_start
        print(f"Generated vector tiles for flight level {flight_level} at zoom levels {vector_zoom_levels} in {vector_time:.2f}s")

    # ---- Generate Raster Tiles ----
    raster_start = time.time()
    if raster_zoom_levels and len(raster_zoom_levels) > 0:
        # Use multiple processes for tile generation
        generate_raster_tiles(lats, lons, u_data, v_data, temp_data,
                            raster_zoom_levels, flight_level, output_dir, num_workers, tile_type)

        raster_time = time.time() - raster_start
        print(f"Generated raster tiles for flight level {flight_level} at zoom levels {raster_zoom_levels} in {raster_time:.2f}s")

    # Clean up to free memory
    del u_data, v_data, temp_data

    total_time = time.time() - start_time
    print(f"Total time for flight level {flight_level}: {total_time:.2f}s")