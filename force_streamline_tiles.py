#!/usr/bin/env python3
"""
Force generation of streamline tiles regardless of existing files.
This script bypasses the normal existence checks and forces regeneration.
"""

import os
import time
import sys
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import from our existing modules
from config import (
    RASTER_ZOOM_LEVELS, OUTPUT_DIR
)
from utils import ensure_dir_exists
from data_processing import read_grib_file, get_wind_components
from tile_generator import create_streamline_tile

def force_streamline_tiles():
    """Force generation of streamline tiles."""
    print("🌪️ Forcing Streamline Tile Generation")
    print("=====================================")

    # Start timing
    total_start_time = time.time()

    # Hard-code the wind file path - adjust if needed
    wind_file = "data/wind-006_012.GRIB"
    if not os.path.exists(wind_file):
        print(f"Error: Wind file {wind_file} not found")
        return

    # Load the wind data
    print(f"Loading data from {wind_file}")
    try:
        wind_ds = read_grib_file(wind_file)
        if wind_ds is None:
            print("Failed to read wind data")
            return

        # Find the level dimension
        level_dim = None
        level_values = []
        for level_name in ['level', 'isobaricInhPa', 'pressure', 'isobaricInPa']:
            if level_name in wind_ds.dims:
                level_dim = level_name
                level_values = wind_ds[level_name].values
                break
            elif level_name in wind_ds.coords:
                level_dim = level_name
                level_values = wind_ds[level_name].values
                break

        if level_dim is None:
            print("Could not find level dimension in the dataset")
            return

        print(f"Found {len(level_values)} levels: {level_values}")

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

        # Check if we need to rearrange longitudes
        lons_adjusted = np.copy(lons)
        data_needs_reordering = np.min(lons) >= 0 and np.max(lons) > 180

        if data_needs_reordering:
            print("Converting longitudes from 0-360 range to -180 to 180")
            # Convert values > 180 to negative values
            lons_adjusted[lons > 180] -= 360
            # Find the split index (where longitude > 180)
            split_idx = np.searchsorted(lons, 180)
            print(f"Longitude split index: {split_idx} (out of {len(lons)})")
    except Exception as e:
        print(f"Error loading wind data: {e}")
        return

    # Ensure the output directory exists
    ensure_dir_exists(OUTPUT_DIR)

    # Process each flight level
    print(f"Processing {len(level_values)} flight levels")
    for level_idx, level_val in enumerate(level_values):
        level_start_time = time.time()
        flight_level = f"{int(level_val)}hPa"
        print(f"Processing level {flight_level} ({level_idx+1}/{len(level_values)})")

        try:
            # Select data for this level
            level_wind_ds = wind_ds.sel({level_dim: level_val})

            # Get U/V wind components for this level
            u_data, v_data = get_wind_components(level_wind_ds)

            # Handle longitude reordering once per level
            if data_needs_reordering:
                # Reorder the data arrays to match new longitude order
                print(f"Reordering data arrays for level {flight_level}")
                split_idx = np.searchsorted(lons, 180)

                # First, get values from the arrays
                u_values = u_data.values
                v_values = v_data.values

                # Reorder into -180 to 180 configuration
                u_reordered = np.concatenate([u_values[:, split_idx:], u_values[:, :split_idx]], axis=1)
                v_reordered = np.concatenate([v_values[:, split_idx:], v_values[:, :split_idx]], axis=1)

                # Create reordered longitude array
                lons_reordered = np.concatenate([lons_adjusted[split_idx:], lons_adjusted[:split_idx]])
            else:
                # No reordering needed
                u_reordered = u_data.values
                v_reordered = v_data.values
                lons_reordered = lons

            # Calculate wind speed based on reordered values
            wind_speed = np.sqrt(u_reordered**2 + v_reordered**2)
            max_speed = 50.0
            norm_wind_speed = np.clip(wind_speed / max_speed, 0, 1)

            # Process each zoom level
            for z in RASTER_ZOOM_LEVELS:
                zoom_start_time = time.time()
                print(f"  Generating tiles for zoom level {z}")

                # Calculate tile grid size for this zoom level
                num_tiles = 2**z
                print(f"    Grid size: {num_tiles}x{num_tiles}")

                # Process each tile
                tiles_generated = 0
                for x in range(num_tiles):
                    # Create streamline directory for this zoom level and x coordinate
                    streamline_tile_dir = os.path.join(OUTPUT_DIR, "streamline", flight_level, str(z), str(x))
                    ensure_dir_exists(streamline_tile_dir)

                    for y in range(num_tiles):
                        # Calculate tile bounds
                        import mercantile
                        tile_bounds = mercantile.bounds(x, y, z)
                        min_lon, min_lat, max_lon, max_lat = tile_bounds

                        # Skip polar regions
                        if min_lat > 85.05 or max_lat < -85.05:
                            continue

                        streamline_tile_path = os.path.join(streamline_tile_dir, f"{y}.png")

                        # Generate the tile
                        try:
                            from scipy.interpolate import griddata

                            # Create a clean matplotlib figure
                            import matplotlib.pyplot as plt
                            plt.close('all')
                            fig = plt.figure(figsize=(256/100, 256/100), dpi=100)
                            ax = fig.add_axes([0, 0, 1, 1])

                            # Create a regular grid for the tile with buffer
                            buffer = 1.0  # Buffer in degrees to ensure smooth edges
                            grid_size = 30  # Resolution of the grid
                            x_grid = np.linspace(min_lon-buffer, max_lon+buffer, grid_size)
                            y_grid = np.linspace(min_lat-buffer, max_lat+buffer, grid_size)
                            xx, yy = np.meshgrid(x_grid, y_grid)

                            # Prepare points for interpolation
                            lons_mesh, lats_mesh = np.meshgrid(lons_reordered, lats)
                            points = np.column_stack((lons_mesh.flatten(), lats_mesh.flatten()))

                            # Interpolate u, v, and speed to the grid
                            u_grid = griddata(points, u_reordered.flatten(), (xx, yy), method='linear')
                            v_grid = griddata(points, v_reordered.flatten(), (xx, yy), method='linear')
                            speed_grid = griddata(points, norm_wind_speed.flatten(), (xx, yy), method='linear')

                            # Create the streamplot
                            ax.streamplot(xx, yy, u_grid, v_grid,
                                       density=1.5,
                                       color=speed_grid,
                                       cmap='viridis',
                                       linewidth=1.5 * speed_grid + 0.5,
                                       arrowsize=1.0,
                                       arrowstyle='->')

                            # Add a subtle background colored by wind speed
                            ax.contourf(xx, yy, speed_grid, levels=10, cmap='viridis', alpha=0.1)

                            # Set the plot limits to the tile boundaries
                            ax.set_xlim(min_lon, max_lon)
                            ax.set_ylim(min_lat, max_lat)
                            ax.axis('off')

                            # Save the streamline tile
                            fig.savefig(streamline_tile_path, format='png',
                                     transparent=True, bbox_inches=None, pad_inches=0, dpi=100)
                            plt.close(fig)

                            tiles_generated += 1

                            # Print progress every 50 tiles or at the start
                            if tiles_generated <= 3 or tiles_generated % 50 == 0:
                                print(f"    Generated {tiles_generated} tiles...")
                        except Exception as e:
                            print(f"Error generating tile z={z}, x={x}, y={y}: {e}")

                zoom_elapsed_time = time.time() - zoom_start_time
                print(f"  Completed zoom level {z}: {tiles_generated} tiles in {zoom_elapsed_time:.1f} seconds")

            level_elapsed_time = time.time() - level_start_time
            print(f"Completed level {flight_level} in {level_elapsed_time:.1f} seconds")

        except Exception as e:
            print(f"Error processing level {flight_level}: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed_time = time.time() - total_start_time
    print("\n✅ Streamline tile generation complete!")
    print(f"Total processing time: {total_elapsed_time:.2f} seconds ({total_elapsed_time / 60:.2f} minutes)")

if __name__ == "__main__":
    force_streamline_tiles()