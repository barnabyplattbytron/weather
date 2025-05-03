#!/usr/bin/env python3
"""
Simple HTTP server for serving the generated weather tiles.
This makes the tiles accessible to web mapping libraries like Leaflet or Mapbox.
"""

import os
import http.server
import socketserver
import argparse
from urllib.parse import unquote
import json
import re
import threading
import numpy as np
import xarray as xr
import mercantile
import time
import io
from PIL import Image

# Import necessary modules for on-the-fly tile generation
from tile_generator import create_wind_tile, create_temp_tile, create_streamline_tile
from grib_processing import process_single_level
from data_processing import read_grib_file, get_wind_components, get_temperature
from utils import ensure_dir_exists
from config import TILE_SIZE

# Global lock for thread safety during tile generation
tile_generation_lock = threading.Lock()

class DataCache:
    """Cache for loaded GRIB data to avoid repeated file reads."""
    def __init__(self, max_size=5):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.last_access = {}

    def get(self, key):
        """Get data from cache if it exists."""
        with self.lock:
            if key in self.cache:
                self.last_access[key] = time.time()
                return self.cache[key]
            return None

    def put(self, key, data):
        """Add data to cache, evicting least recently used item if needed."""
        with self.lock:
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.last_access.items(), key=lambda x: x[1])[0]
                del self.cache[oldest_key]
                del self.last_access[oldest_key]

            self.cache[key] = data
            self.last_access[key] = time.time()

    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.last_access.clear()

# Initialize global data cache
data_cache = DataCache()

def load_and_process_data(data_path, flight_level):
    """Load and process GRIB data for a specific flight level."""
    # Try to get from cache first
    cache_key = f"{data_path}_{flight_level}"
    cached_data = data_cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    try:
        # Load GRIB file
        print(f"Loading GRIB data from {data_path} for level {flight_level}")
        ds = read_grib_file(data_path)
        if ds is None:
            print(f"Failed to read GRIB file: {data_path}")
            return None

        # Check if this is a multi-level dataset
        level_dim = None
        level_value = None

        # Extract numeric value from flight level
        if isinstance(flight_level, str) and "hPa" in flight_level:
            level_num = int(flight_level.replace("hPa", ""))
        else:
            print(f"Invalid flight level format: {flight_level}")
            return None

        # Check for common level dimension names
        for level_name in ['level', 'isobaricInhPa', 'pressure', 'isobaricInPa']:
            if level_name in ds.dims:
                level_dim = level_name
                # Find closest level value
                closest_idx = np.abs(ds[level_name].values - level_num).argmin()
                level_value = ds[level_name].values[closest_idx]
                break
            elif level_name in ds.coords:
                level_dim = level_name
                closest_idx = np.abs(ds[level_name].values - level_num).argmin()
                level_value = ds[level_name].values[closest_idx]
                break

        # Select the correct level if it's a multi-level dataset
        if level_dim is not None and level_value is not None:
            ds = ds.sel({level_dim: level_value})

        # Get U/V components and temperature
        u_data, v_data = get_wind_components(ds)
        temp_data = get_temperature(ds)

        # Get latitude and longitude arrays
        try:
            lats = ds.latitude.values
            lons = ds.longitude.values
        except AttributeError:
            lat_vars = ['latitude', 'lat']
            lon_vars = ['longitude', 'lon']

            lat_var = next((var for var in lat_vars if hasattr(ds, var)), None)
            lon_var = next((var for var in lon_vars if hasattr(ds, var)), None)

            if lat_var is None or lon_var is None:
                print("Could not find latitude/longitude coordinates in dataset")
                return None

            lats = getattr(ds, lat_var).values
            lons = getattr(ds, lon_var).values

        # Check if longitudes are in 0-360 range and convert to -180 to 180 if needed
        if np.min(lons) >= 0 and np.max(lons) > 180:
            print("Converting longitudes from 0-360 range to -180 to 180")
            # Create a copy of the longitude array
            lons_normalized = np.copy(lons)
            # Convert values > 180 to negative values
            lons_normalized[lons > 180] -= 360

            # Reorder the data arrays if needed
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

                if temp_data is not None:
                    temp_data_values = temp_data.values
                    temp_data_reordered = np.concatenate([temp_data_values[:, split_idx:],
                                                        temp_data_values[:, :split_idx]], axis=1)
                    temp_data = xr.DataArray(temp_data_reordered, dims=temp_data.dims,
                                           coords={temp_data.dims[0]: lats, temp_data.dims[1]: lons})
            else:
                lons = lons_normalized

        # Cache the processed data
        result = {
            'u_data': u_data,
            'v_data': v_data,
            'temp_data': temp_data,
            'lats': lats,
            'lons': lons,
        }

        data_cache.put(cache_key, result)
        return result

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_blank_tile():
    """Create a blank transparent PNG tile."""
    # Create a blank transparent image
    img = Image.new('RGBA', (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))

    # Save to in-memory buffer
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)

    return buffer.getvalue()

def generate_tile_on_demand(tile_type, flight_level, z, x, y, format='png'):
    """Generate a requested tile on demand if it doesn't exist."""
    # Define path where this tile should be stored
    tile_dir = os.path.join("tiles", tile_type, flight_level, str(z), str(x))
    tile_path = os.path.join(tile_dir, f"{y}.{format}")

    # If the tile already exists, return the path
    if os.path.exists(tile_path):
        return tile_path

    # Ensure the directory exists
    ensure_dir_exists(tile_dir)

    # Use a lock to prevent multiple threads from generating the same tile
    with tile_generation_lock:
        # Check again in case another thread generated it while waiting
        if os.path.exists(tile_path):
            return tile_path

        try:
            # Find the appropriate data files
            # This is a simplified version - in production you'd have a more robust way to find the correct data file
            data_dir = "data"
            data_files = os.listdir(data_dir)

            wind_file = None
            temp_file = None

            for file in data_files:
                if file.lower().startswith("wind") and file.endswith(".GRIB"):
                    wind_file = os.path.join(data_dir, file)
                elif file.lower().startswith("temp") and file.endswith(".GRIB"):
                    temp_file = os.path.join(data_dir, file)

            if wind_file is None:
                print("No wind data file found")
                return None

            # Load and process the data
            data = load_and_process_data(wind_file, flight_level)
            if data is None:
                print(f"Failed to load data for {flight_level}")
                return None

            # Calculate tile bounds
            bounds = mercantile.bounds(x, y, z)
            min_lon, min_lat, max_lon, max_lat = bounds

            # Generate the tile based on type
            if tile_type == "wind":
                # Calculate wind speed for coloring
                u_array = np.array(data['u_data'].values)
                v_array = np.array(data['v_data'].values)
                wind_speed = np.sqrt(u_array**2 + v_array**2)
                max_speed = 50.0  # Maximum wind speed in m/s
                norm_wind_speed = np.clip(wind_speed / max_speed, 0, 1)

                # Create wind tile
                create_wind_tile(u_array, v_array, norm_wind_speed, data['lats'], data['lons'],
                               min_lat, max_lat, min_lon, max_lon, tile_path)

            elif tile_type == "temp" and data['temp_data'] is not None:
                # Convert temperature from Kelvin to Celsius
                temp_array = np.array(data['temp_data'].values)
                temp_celsius = temp_array - 273.15

                # Get appropriate temperature range for this flight level
                from colormap import get_temp_range_for_level
                min_temp, max_temp = get_temp_range_for_level(flight_level)

                # Normalize temperature for the colormap
                norm_temp = np.clip((temp_celsius - min_temp) / (max_temp - min_temp), 0, 1)

                # Create temperature tile
                create_temp_tile(norm_temp, data['lats'], data['lons'],
                               min_lat, max_lat, min_lon, max_lon, tile_path, flight_level)

            elif tile_type == "streamline":
                # Calculate wind speed for coloring
                u_array = np.array(data['u_data'].values)
                v_array = np.array(data['v_data'].values)
                wind_speed = np.sqrt(u_array**2 + v_array**2)
                max_speed = 50.0  # Maximum wind speed in m/s
                norm_wind_speed = np.clip(wind_speed / max_speed, 0, 1)

                # Create streamline tile
                create_streamline_tile(u_array, v_array, norm_wind_speed, data['lats'], data['lons'],
                                    min_lat, max_lat, min_lon, max_lon, tile_path)

            # Check if the tile was successfully created
            if os.path.exists(tile_path):
                print(f"Successfully generated tile on demand: {tile_path}")
                return tile_path
            else:
                print(f"Failed to generate tile: {tile_path}")
                return None

        except Exception as e:
            print(f"Error generating tile on demand: {e}")
            import traceback
            traceback.print_exc()
            return None

class TileHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for tile server."""

    def __init__(self, *args, **kwargs):
        # Set the directory to serve files from
        super().__init__(*args, directory=os.path.abspath("tiles"), **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        # Decode URL path
        path = unquote(self.path)

        # Serve API metadata for /api or /metadata
        if path in ["/api", "/metadata"]:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            api_info = {
                "app": "Weather Tile API",
                "version": "1.0.0",
                "endpoints": {
                    "times": "/times",
                    "levels": "/levels",
                    "levels_for_time": "/times/{time}/levels",
                    "tile_wind": "/wind/{level}/{z}/{x}/{y}.png",
                    "tile_temp": "/temp/{level}/{z}/{x}/{y}.png",
                    "tile_vector": "/tiles/vector/{level}/{z}/{x}/{y}.geojson",
                    "clear_cache": "/tiles/clear-cache",
                    "clear_data_cache": "/api/clear-data-cache"
                }
            }
            self.wfile.write(json.dumps(api_info).encode())
            return

        # --- API: /times ---
        if path == "/times":
            times = set()
            for root, dirs, files in os.walk(os.path.join("tiles", "wind")):
                for d in dirs:
                    if d.isdigit():
                        times.add(d)
            times = sorted(times)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"times": times}).encode())
            return

        # --- API: /levels ---
        if path == "/levels":
            levels = []
            wind_dir = os.path.join("tiles", "wind")
            if os.path.exists(wind_dir):
                levels = sorted([fl for fl in os.listdir(wind_dir) if not fl.startswith('.')])
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"levels": levels}).encode())
            return

        # --- API: /times/{time}/levels ---
        if path.startswith("/times/") and path.endswith("/levels"):
            try:
                time_val = path.split("/")[2]
                levels = []
                wind_dir = os.path.join("tiles", "wind")
                if os.path.exists(wind_dir):
                    for fl in os.listdir(wind_dir):
                        if not fl.startswith('.'):
                            level_time_dir = os.path.join(wind_dir, fl, time_val)
                            if os.path.exists(level_time_dir):
                                levels.append(fl)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"levels": levels}).encode())
                return
            except Exception as e:
                self.send_error(400, f"Invalid request: {e}")
                return

        # --- API: /tiles/clear-cache ---
        if path == "/tiles/clear-cache":
            import shutil
            try:
                cache_dir = os.path.join("tiles", "cache")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "cache cleared"}).encode())
            except Exception as e:
                self.send_error(500, f"Failed to clear cache: {e}")
            return

        # --- API: /api/clear-data-cache ---
        if path == "/api/clear-data-cache":
            try:
                data_cache.clear()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "data cache cleared"}).encode())
            except Exception as e:
                self.send_error(500, f"Failed to clear data cache: {e}")
            return

        # Serve index.html for root path
        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(self.generate_index_html().encode())
            return

        # --- Check for wind/temp tile request patterns ---
        # Match patterns like /wind/250hPa/6/12/24.png or /temp/250hPa/6/12/24.png or /streamline/250hPa/6/12/24.png
        tile_match = re.match(r'^/(wind|temp|streamline)/([^/]+)/(\d+)/(\d+)/(\d+)\.png$', path)
        if tile_match:
            tile_type = tile_match.group(1)  # wind, temp, or streamline
            flight_level = tile_match.group(2)  # e.g., 250hPa
            z = int(tile_match.group(3))  # zoom level
            x = int(tile_match.group(4))  # tile x coordinate
            y = int(tile_match.group(5))  # tile y coordinate

            # Standard file path based on current structure
            tile_path = os.path.join("tiles", tile_type, flight_level, str(z), str(x), f"{y}.png")

            # Check if the tile exists
            if not os.path.exists(tile_path):
                self.log_message(f"Generating missing tile on demand: {path}")

                # Try to generate the tile on demand
                generated_path = generate_tile_on_demand(tile_type, flight_level, z, x, y)

                # If generation failed, return a blank transparent tile instead of 404
                if generated_path is None:
                    self.log_message(f"Could not generate tile for {path}, returning blank tile")

                    # Create a directory for the blank tile if it doesn't exist
                    blank_tile_dir = os.path.join("tiles", tile_type, flight_level, str(z), str(x))
                    ensure_dir_exists(blank_tile_dir)

                    # Create a blank transparent tile in the expected location
                    blank_tile_path = os.path.join(blank_tile_dir, f"{y}.png")

                    # Save the blank tile to disk for future requests
                    with open(blank_tile_path, 'wb') as f:
                        f.write(create_blank_tile())

                    # Set tile_path to our new blank tile
                    tile_path = blank_tile_path
                else:
                    # We'll continue and serve the newly generated tile
                    tile_path = generated_path

            # Serve the tile
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Access-Control-Allow-Origin', '*')
                # Add cache control headers
                self.send_header('Cache-Control', 'public, max-age=86400')  # Cache for 24 hours
                self.end_headers()

                with open(tile_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            except Exception as e:
                # If all else fails, generate a blank tile on the fly
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(create_blank_tile())
                self.log_message(f"Error serving tile, returned blank tile: {e}")
                return

        # Check if this is a tile request
        try:
            file_path = os.path.join(os.path.abspath("tiles"), path.lstrip('/'))

            # Handle GeoJSON tiles
            if path.endswith('.geojson') and os.path.exists(file_path):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
                return

            # Handle PNG image tiles
            elif path.endswith('.png') and os.path.exists(file_path):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
                return

            # Log if the file wasn't found
            if not os.path.exists(file_path):
                self.log_message(f"File not found: {file_path}")
        except Exception as e:
            self.log_message(f"Error handling request: {e}")

        # Fall back to default behavior for other requests
        try:
            super().do_GET()
        except Exception as e:
            self.send_error(404, f"File not found: {self.path}")
            self.log_message(f"Error serving file: {e}")

    def generate_index_html(self):
        """Return a minimal HTML page indicating the API is running (frontend test harness removed)."""
        html = """<!DOCTYPE html>
<html><head><title>Weather Tile API</title></head><body>
<h2>Weather Tile API is running.</h2>
<p>This server provides weather tile data via API endpoints.</p>
<h3>Added Features:</h3>
<ul>
    <li><strong>On-the-fly Tile Generation:</strong> Tiles that don't exist will be generated dynamically when requested</li>
    <li><strong>Optimized Caching:</strong> Generated tiles are saved for future use</li>
    <li><strong>Memory-efficient Data Cache:</strong> Source data is cached in memory for faster tile generation</li>
</ul>
</body></html>"""
        return html

def main():
    """Start the tile server."""
    parser = argparse.ArgumentParser(description='Weather Tile Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    args = parser.parse_args()

    # Create server
    handler = TileHandler
    server = socketserver.TCPServer(("", args.port), handler)

    print(f"🌐 Serving weather tiles at http://localhost:{args.port}/")
    print("Press Ctrl+C to stop the server")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()