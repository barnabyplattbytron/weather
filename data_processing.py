#!/usr/bin/env python3
"""
Wind & Temperature Tile Server - Data Processing
Functions for loading and processing GRIB data.
"""

import xarray as xr
import numpy as np
from geojson import Feature, FeatureCollection, Point
import mercantile
import json
import os

from utils import calculate_wind_speed_direction, ensure_dir_exists

def read_grib_file(file_path):
    """Read GRIB file using xarray and cfgrib."""
    try:
        ds = xr.open_dataset(file_path, engine='cfgrib')
        return ds
    except Exception as e:
        print(f"Error reading GRIB file {file_path}: {e}")
        return None

def get_wind_components(ds):
    """Extract U and V components from wind dataset."""
    # Different GRIB files might use different variable names
    u_vars = ['u', 'U', 'u10', 'U10', 'UGRD']
    v_vars = ['v', 'V', 'v10', 'V10', 'VGRD']

    u_var = next((var for var in u_vars if var in ds), None)
    v_var = next((var for var in v_vars if var in ds), None)

    if u_var is None or v_var is None:
        raise ValueError(f"Could not find U/V components in dataset: {list(ds.data_vars)}")

    # Get the variables
    u_data = ds[u_var]
    v_data = ds[v_var]

    # Handle multi-dimensional data - extract first time and level if present
    # This extracts the first element of all dimensions except lat/lon
    if len(u_data.dims) > 2:
        print(f"Found multi-dimensional data: {u_data.dims}")
        # Get the last two dimensions which should be lat/lon
        lat_dim = u_data.dims[-2]
        lon_dim = u_data.dims[-1]

        # Select first element of all other dimensions
        indexers = {dim: 0 for dim in u_data.dims if dim not in [lat_dim, lon_dim]}
        u_data = u_data.isel(**indexers)
        v_data = v_data.isel(**indexers)

    return u_data, v_data

def get_temperature(ds):
    """Extract temperature from temperature dataset."""
    # Different GRIB files might use different variable names
    temp_vars = ['t', 'T', 't2m', 'T2M', 'TMP']

    temp_var = next((var for var in temp_vars if var in ds), None)

    if temp_var is None:
        raise ValueError(f"Could not find temperature in dataset: {list(ds.data_vars)}")

    # Get the variable
    temp_data = ds[temp_var]

    # Handle multi-dimensional data - extract first time and level if present
    if len(temp_data.dims) > 2:
        # Get the last two dimensions which should be lat/lon
        lat_dim = temp_data.dims[-2]
        lon_dim = temp_data.dims[-1]

        # Select first element of all other dimensions
        indexers = {dim: 0 for dim in temp_data.dims if dim not in [lat_dim, lon_dim]}
        temp_data = temp_data.isel(**indexers)

    return temp_data

def extract_flight_level(dataset):
    """Extract flight level (pressure level) from the dataset in hPa."""
    try:
        # Check for common pressure level coordinate names
        level_vars = ['level', 'isobaricInhPa', 'pressure', 'isobaricInPa']

        # Find which level variable exists in the dataset
        level_var = next((var for var in level_vars if var in dataset.coords), None)

        if level_var:
            # Get the pressure level value
            level_value = dataset.coords[level_var].values

            # If it's an array, take the first value
            if hasattr(level_value, '__len__') and not isinstance(level_value, str):
                level_value = level_value[0]

            # Convert to string for directory name
            if level_var == 'isobaricInPa':
                # Convert Pa to hPa
                level_hpa = int(level_value / 100)
                return f"{level_hpa}hPa"
            else:
                return f"{int(level_value)}hPa"
        else:
            # If no pressure level is found, check for surface level indicators
            if any(var.startswith(('2m_', '10m_', 'surface')) for var in dataset.data_vars):
                return "surface"

            # Fall back to extracting from filename as before
            filename = dataset.encoding.get('source', '')
            parts = os.path.basename(filename).split('-')[1].split('.')[0]
            return parts
    except Exception as e:
        print(f"Error extracting flight level: {e}")
        return "unknown"

def create_geojson_features(lats, lons, u_data, v_data, temp_data=None):
    """Create GeoJSON features from data points."""
    features = []

    # Print shapes for debugging
    print(f"Shapes: lats {lats.shape}, lons {lons.shape}, u_data {u_data.shape}, v_data {v_data.shape}")
    if temp_data is not None:
        print(f"temp_data shape: {temp_data.shape}")

    # Process data - iterate over lat/lon points
    for i in range(len(lats)):
        lat = float(lats[i])

        # Skip polar regions (latitude too close to +/-90°) due to Web Mercator limitations
        if abs(lat) > 85.05:
            continue

        for j in range(len(lons)):
            lon = float(lons[j])

            # Get U/V values
            try:
                u_val = u_data[i, j].item()  # Use .item() to convert numpy scalar to Python scalar
                v_val = v_data[i, j].item()

                if np.isnan(u_val) or np.isnan(v_val):
                    continue

                # Calculate speed and direction
                speed, direction = calculate_wind_speed_direction(u_val, v_val)

                # Create properties dictionary
                properties = {
                    "u": u_val,
                    "v": v_val,
                    "speed": float(speed),
                    "direction": float(direction)
                }

                # Add temperature if available
                if temp_data is not None:
                    try:
                        temp_val = temp_data[i, j].item()
                        if not np.isnan(temp_val):
                            properties["temperature"] = float(temp_val)
                    except (IndexError, TypeError):
                        # Skip temperature if there's an issue
                        pass

                # Create GeoJSON feature
                feature = Feature(
                    geometry=Point((lon, lat)),
                    properties=properties
                )
                features.append(feature)
            except (IndexError, TypeError, ValueError) as e:
                # Skip this point if there's an issue
                continue

    print(f"Created {len(features)} GeoJSON features")
    return features

def assign_features_to_tiles(features, zoom_levels):
    """Assign features to tiles at different zoom levels."""
    tiles_features = {zoom: {} for zoom in zoom_levels}

    for feature in features:
        lon, lat = feature["geometry"]["coordinates"]

        for z in zoom_levels:
            # Get the tile coordinates at this zoom level
            try:
                tile = mercantile.tile(lon, lat, z)
                tile_key = f"{tile.z}_{tile.x}_{tile.y}"

                if tile_key not in tiles_features[z]:
                    tiles_features[z][tile_key] = []

                tiles_features[z][tile_key].append(feature)
            except Exception as e:
                print(f"Error processing tile for point ({lon}, {lat}) at zoom {z}: {e}")

    return tiles_features

def save_geojson_tiles(tiles_features, flight_level, output_dir):
    """Save tiles as GeoJSON files."""
    for z, tiles in tiles_features.items():
        for tile_key, features in tiles.items():
            z, x, y = map(int, tile_key.split("_"))

            # Create directory structure
            tile_dir = os.path.join(output_dir, flight_level, str(z), str(x))
            ensure_dir_exists(tile_dir)

            # Create feature collection
            feature_collection = FeatureCollection(features)

            # Save GeoJSON file
            file_path = os.path.join(tile_dir, f"{y}.geojson")
            with open(file_path, 'w') as f:
                json.dump(feature_collection, f)