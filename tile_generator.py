#!/usr/bin/env python3
"""
Wind & Temperature Tile Server - Tile Generator
Functions for generating wind and temperature tiles.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend to Agg (non-interactive) for thread safety
import matplotlib.pyplot as plt
from PIL import Image
import mercantile
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import platform

from config import TILE_SIZE
from colormap import get_temp_colormap_for_level, WIND_CMAP
from utils import ensure_dir_exists


# Define process_batch function at module level to be properly pickable
def process_batch(batch):
    """Process a batch of tile tasks."""
    results = []
    for task in batch:
        try:
            task_type = task[0]
            if task_type == 'wind':
                _, u, v, speed, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path, _ = task
                create_wind_tile(u, v, speed, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path)
            elif task_type == 'streamline':
                _, u, v, speed, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path, _ = task
                create_streamline_tile(u, v, speed, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path)
            else:  # temp
                _, _, _, temp, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path, flight_level = task
                create_temp_tile(temp, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path, flight_level)
            results.append(True)
        except Exception as e:
            print(f"Error processing tile: {e}")
            results.append(False)
    return results


def get_optimal_executor(use_processes=None):
    """Determine the optimal executor based on platform and environment."""
    if use_processes is not None:
        # Explicit choice overrides automatic selection
        return ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    # Check environment variable
    env_setting = os.environ.get('USE_PROCESSES', None)
    if env_setting is not None:
        return ProcessPoolExecutor if env_setting == '1' else ThreadPoolExecutor

    # Platform-specific defaults based on performance testing
    system = platform.system()
    if system == 'Darwin':  # macOS
        # On macOS, ThreadPoolExecutor often performs better due to lower overhead
        return ThreadPoolExecutor
    elif system == 'Linux':
        # On Linux, ProcessPoolExecutor often performs better especially on many-core systems
        return ProcessPoolExecutor
    elif system == 'Windows':
        # On Windows, ThreadPoolExecutor typically has less overhead
        return ThreadPoolExecutor
    else:
        # Default to ThreadPoolExecutor for unknown platforms
        return ThreadPoolExecutor


def generate_raster_tiles(lats, lons, u_data, v_data, temp_data, zoom_levels, flight_level, output_dir, num_workers, tile_type='all'):
    """Generate raster PNG tiles for wind and temperature data."""
    print(f"Generating raster tiles for zoom levels {zoom_levels}...")
    if tile_type != 'all':
        print(f"Generating only {tile_type} tiles")

    # Flag to force regeneration when a specific tile type is requested
    force_regeneration = tile_type != 'all'

    # Debug: Check if output_dir exists
    print(f"Output directory: {output_dir} exists: {os.path.exists(output_dir)}")

    # Make sure output directory exists
    ensure_dir_exists(output_dir)

    # Convert data to 2D numpy arrays
    u_array = np.array(u_data.values)
    v_array = np.array(v_data.values)

    # Calculate wind speed for coloring
    wind_speed = np.sqrt(u_array**2 + v_array**2)

    # Normalize wind speeds for colormap (adjust max_speed as needed for your data)
    max_speed = 50.0  # Maximum wind speed in m/s
    norm_wind_speed = np.clip(wind_speed / max_speed, 0, 1)

    # Process temperature data if available
    if temp_data is not None:
        temp_array = np.array(temp_data.values)
        # Convert Kelvin to Celsius
        temp_celsius = temp_array - 273.15

        # Get appropriate temperature range for this flight level
        from colormap import get_temp_range_for_level
        min_temp, max_temp = get_temp_range_for_level(flight_level)
        print(f"Using temperature range for {flight_level}: {min_temp}°C to {max_temp}°C")

        # Normalize temperature based on the flight-level appropriate range
        norm_temp = np.clip((temp_celsius - min_temp) / (max_temp - min_temp), 0, 1)
    else:
        norm_temp = None

    # Check if we should cache the processed data for reuse across zoom levels
    cache_dir = os.path.join("cache", f"{hash(tuple(lats)) & 0xFFFFFF:x}")
    os.makedirs(cache_dir, exist_ok=True)

    # Save processed data to cache for reuse
    u_cache_file = os.path.join(cache_dir, f"{flight_level}_u.npy")
    v_cache_file = os.path.join(cache_dir, f"{flight_level}_v.npy")
    temp_cache_file = os.path.join(cache_dir, f"{flight_level}_temp.npy")

    # Save normalized data to cache
    np.save(u_cache_file, u_array)
    np.save(v_cache_file, v_array)
    if norm_temp is not None:
        np.save(temp_cache_file, norm_temp)

    print(f"Cached processed data in {cache_dir}")

    # Ensure num_workers is an integer
    try:
        num_workers = int(num_workers)
    except (TypeError, ValueError):
        print(f"Warning: Invalid num_workers value: {num_workers}, defaulting to 4")
        num_workers = 4

    # Generate tiles for each zoom level
    for z in zoom_levels:
        tiles_processed = 0
        start_time = time.time()

        # Calculate bounds of the tile grid at this zoom level
        num_tiles = 2**z
        print(f"Processing zoom level {z}: {num_tiles}x{num_tiles} potential tiles")

        # If specifically generating streamline tiles, clean the directory structure first
        if tile_type == 'streamline':
            streamline_dir = os.path.join(output_dir, "streamline", flight_level, str(z))
            if os.path.exists(streamline_dir):
                print(f"Force regenerating streamline tiles for zoom level {z} - existing tiles will be replaced")

        # Prepare batch processing - group tiles into batches
        batch_size = max(1, min(100, (num_tiles * num_tiles) // (num_workers * 2)))  # Dynamically adjust batch size

        # Create all tile tasks
        tasks = []
        for x in range(num_tiles):
            for y in range(num_tiles):
                # Calculate the lat/lon bounds of this tile
                tile_bounds = mercantile.bounds(x, y, z)
                min_lon, min_lat, max_lon, max_lat = tile_bounds

                # Skip tiles that are outside the data bounds or in polar regions
                if min_lat > 85.05 or max_lat < -85.05:
                    continue

                # Create directory structure based on tile type
                if tile_type in ['wind', 'all']:
                    wind_tile_dir = os.path.join(output_dir, "wind", flight_level, str(z), str(x))
                    ensure_dir_exists(wind_tile_dir)
                    wind_tile_path = os.path.join(wind_tile_dir, f"{y}.png")
                    wind_exists = os.path.exists(wind_tile_path) and os.path.getsize(wind_tile_path) > 100
                    if force_regeneration and tile_type == 'wind':
                        wind_exists = False  # Force regeneration
                    if not wind_exists:
                        tasks.append(('wind', u_array, v_array, norm_wind_speed, lats, lons,
                                    min_lat, max_lat, min_lon, max_lon,
                                    wind_tile_path, None))

                if tile_type in ['streamline', 'all']:
                    streamline_tile_dir = os.path.join(output_dir, "streamline", flight_level, str(z), str(x))
                    ensure_dir_exists(streamline_tile_dir)
                    streamline_tile_path = os.path.join(streamline_tile_dir, f"{y}.png")

                    # Force regeneration for streamline tiles when specifically requested
                    streamline_exists = os.path.exists(streamline_tile_path) and os.path.getsize(streamline_tile_path) > 100
                    if tile_type == 'streamline':  # Always force regeneration when specifically asked for streamline tiles
                        streamline_exists = False
                        if os.path.exists(streamline_tile_path):
                            try:
                                os.remove(streamline_tile_path)
                            except Exception as e:
                                print(f"Error removing existing tile {streamline_tile_path}: {e}")

                    if not streamline_exists:
                        tasks.append(('streamline', u_array, v_array, norm_wind_speed, lats, lons,
                                    min_lat, max_lat, min_lon, max_lon,
                                    streamline_tile_path, None))

                if tile_type in ['temp', 'all'] and norm_temp is not None:
                    temp_tile_dir = os.path.join(output_dir, "temp", flight_level, str(z), str(x))
                    ensure_dir_exists(temp_tile_dir)
                    temp_tile_path = os.path.join(temp_tile_dir, f"{y}.png")

                    # Force regeneration for temp tiles when specifically requested
                    temp_exists = os.path.exists(temp_tile_path) and os.path.getsize(temp_tile_path) > 100
                    if force_regeneration and tile_type == 'temp':
                        temp_exists = False  # Force regeneration
                        if os.path.exists(temp_tile_path):
                            try:
                                os.remove(temp_tile_path)
                            except Exception as e:
                                print(f"Error removing tile {temp_tile_path}: {e}")

                    if not temp_exists:
                        tasks.append(('temp', None, None, norm_temp, lats, lons,
                                    min_lat, max_lat, min_lon, max_lon,
                                    temp_tile_path, flight_level))

        if not tasks:
            print(f"No new tiles to generate for zoom level {z}, all tiles exist")
            continue

        print(f"Processing {len(tasks)} new tiles for zoom level {z} with batch size {batch_size}")

        # Split tasks into batches
        task_batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

        # Choose the optimal executor based on platform and settings
        use_processes = os.environ.get('USE_PROCESSES', '0') == '1'
        executor_class = get_optimal_executor(use_processes)
        executor_name = executor_class.__name__.replace('Executor', '')

        print(f"Using {executor_name} with {num_workers} workers")

        # Process batches using selected executor
        with executor_class(max_workers=num_workers) as executor:
            futures = []
            for batch in task_batches:
                futures.append(executor.submit(process_batch, batch))

            # Wait for all batches to complete and collect results
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    successful = sum(1 for result in batch_results if result)
                    tiles_processed += successful
                    completed += 1
                    # Show progress every 5% or at least once every 10 batches
                    if completed % max(1, len(futures) // 20) == 0:
                        progress = (completed / len(futures)) * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / completed) * (len(futures) - completed) if completed > 0 else "unknown"
                        eta_str = f"{eta:.1f}s" if isinstance(eta, float) else eta
                        print(f"Progress: {progress:.1f}% ({completed}/{len(futures)} batches), ETA: {eta_str}")
                except Exception as e:
                    print(f"Error processing batch: {e}")

        elapsed_time = time.time() - start_time
        tile_rate = tiles_processed / elapsed_time if elapsed_time > 0 else 0
        print(f"Completed zoom level {z}: {tiles_processed} tiles in {elapsed_time:.1f} seconds ({tile_rate:.1f} tiles/sec)")

def create_wind_tile(u, v, speed, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path):
    """Create a raster tile for wind data."""
    # Close any existing figures to prevent memory leaks
    plt.close('all')

    # Create a figure with exact pixel dimensions
    fig = plt.figure(figsize=(TILE_SIZE/100, TILE_SIZE/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    # Filter points within this tile's bounds
    lat_mask = (lats >= min_lat) & (lats <= max_lat)
    lon_mask = (lons >= min_lon) & (lons <= max_lon)

    # Skip if no data points in this tile
    if not np.any(lat_mask) or not np.any(lon_mask):
        # Create an empty transparent tile with a small marker in the center to ensure it's not empty
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        # Add a small dot to ensure the image isn't completely empty
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
        ax.axis('off')

        # Explicitly set the figure size
        fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

        # Save with a higher quality and explicit format
        fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0,
                   format='png', dpi=100)
        plt.close(fig)
        return

    # Calculate a reasonable stride to avoid overcrowding
    lat_idxs = np.where(lat_mask)[0]
    lon_idxs = np.where(lon_mask)[0]

    if len(lat_idxs) == 0 or len(lon_idxs) == 0:
        # Create an empty transparent tile with a small marker
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        # Add a small dot to ensure the image isn't completely empty
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
        ax.axis('off')

        # Explicitly set the figure size
        fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

        # Save with a higher quality and explicit format
        fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0,
                   format='png', dpi=100)
        plt.close(fig)
        return

    stride_lat = max(1, len(lat_idxs) // 10)
    stride_lon = max(1, len(lon_idxs) // 10)

    # Get subset of lat/lon indices with stride
    lat_subset = lat_idxs[::stride_lat]
    lon_subset = lon_idxs[::stride_lon]

    # Create a meshgrid of coordinates
    lons_mesh, lats_mesh = np.meshgrid(lons[lon_subset], lats[lat_subset])

    # Get corresponding u, v components
    u_mesh = u[np.ix_(lat_subset, lon_subset)]
    v_mesh = v[np.ix_(lat_subset, lon_subset)]
    speed_mesh = speed[np.ix_(lat_subset, lon_subset)]

    # Map wind speed to colors - using discrete color levels
    colors = []
    for s in speed_mesh.flatten():
        if s < 0.2:  # Light winds
            colors.append('blue')
        elif s < 0.4:  # Moderate winds
            colors.append('green')
        elif s < 0.7:  # Strong winds
            colors.append('orange')
        else:  # Severe winds
            colors.append('red')

    # Create the wind barbs plot
    ax.barbs(lons_mesh.flatten(), lats_mesh.flatten(),
            u_mesh.flatten(), v_mesh.flatten(),
            # Using the color array instead of a colormap for individual colors
            color=colors,
            # Reduce barb size for clarity
            sizes=dict(emptybarb=0.05, spacing=0.15, height=0.3),
            linewidth=0.5,
            # Use appropriate units and scaling
            pivot='middle')

    # Set the plot limits to match the tile boundaries
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.axis('off')

    # Explicitly set the figure size
    fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

    # Save the figure as a PNG with transparency
    try:
        fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0,
                   format='png', dpi=100)

        # Verify the file exists and has content
        if os.path.getsize(output_path) < 100:  # Less than 100 bytes is suspicious
            print(f"Warning: Small file size for {output_path}")
            # Try a different approach - save with PIL
            canvas = fig.canvas
            canvas.draw()
            img = Image.frombytes('RGBA', canvas.get_width_height(), canvas.buffer_rgba())
            img.save(output_path, format='PNG')
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")

    plt.close(fig)

def create_streamline_tile(u, v, speed, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path):
    """Create a streamline visualization for wind data, showing flow lines colored by speed."""
    # Close any existing figures to prevent memory leaks
    plt.close('all')

    # Create a figure with exact pixel dimensions
    fig = plt.figure(figsize=(TILE_SIZE/100, TILE_SIZE/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    # Add a buffer to ensure smooth transitions across tile boundaries
    buffer = 1.0  # Buffer in degrees to capture more points outside the tile
    lat_mask = (lats >= min_lat - buffer) & (lats <= max_lat + buffer)
    lon_mask = (lons >= min_lon - buffer) & (lons <= max_lon + buffer)

    # Skip if no data points in this tile
    if not np.any(lat_mask) or not np.any(lon_mask):
        # Create an empty transparent tile with a small marker
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
        ax.axis('off')

        # Save with explicit format
        fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)
        fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
        plt.close(fig)
        return

    try:
        # Extract indices for our region of interest (with buffer)
        lat_idxs = np.where(lat_mask)[0]
        lon_idxs = np.where(lon_mask)[0]

        if len(lat_idxs) < 2 or len(lon_idxs) < 2:
            # Not enough data points - create a minimal tile
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
            ax.axis('off')
            fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
            plt.close(fig)
            return

        # Extract the relevant section of the data (with buffer)
        min_lat_idx = min(lat_idxs)
        max_lat_idx = max(lat_idxs)
        min_lon_idx = min(lon_idxs)
        max_lon_idx = max(lon_idxs)

        # Get the data subset including the buffer zone
        u_subset = u[min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
        v_subset = v[min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
        speed_subset = speed[min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
        lats_subset = lats[min_lat_idx:max_lat_idx+1]
        lons_subset = lons[min_lon_idx:max_lon_idx+1]

        # Check if longitude values are strictly increasing
        if not np.all(np.diff(lons_subset) > 0):
            # If we have longitude discontinuity (like crossing the antimeridian),
            # we need to create a regular grid instead
            print(f"Longitude discontinuity detected. Creating regular grid for tile.")

            # Create a regular grid for the streamplot that matches our tile boundaries
            grid_size = 30  # Use consistent grid size for streamlines
            x = np.linspace(min_lon, max_lon, grid_size)
            y = np.linspace(min_lat, max_lat, grid_size)

            # Create meshgrids for the regular grid
            xx, yy = np.meshgrid(x, y)

            # Now we need to interpolate the wind data to this regular grid
            from scipy.interpolate import griddata

            # Create source points for interpolation - reshape to 2D points
            lons_mesh, lats_mesh = np.meshgrid(lons_subset, lats_subset)
            points = np.column_stack((lons_mesh.flat, lats_mesh.flat))

            # Prepare flattened data values
            u_flat = u_subset.flat
            v_flat = v_subset.flat
            speed_flat = speed_subset.flat

            # Check that we have valid data
            valid_mask = ~np.isnan(u_flat) & ~np.isnan(v_flat) & ~np.isnan(speed_flat)
            if np.sum(valid_mask) < 3:
                # Not enough valid data points
                ax.set_xlim(min_lon, max_lon)
                ax.set_ylim(min_lat, max_lat)
                center_lon = (min_lon + max_lon) / 2
                center_lat = (min_lat + max_lat) / 2
                ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
                ax.axis('off')
                fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)
                fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
                plt.close(fig)
                return

            # Use only valid data points for interpolation
            valid_points = points[valid_mask]
            valid_u = u_flat[valid_mask]
            valid_v = v_flat[valid_mask]
            valid_speed = speed_flat[valid_mask]

            # Interpolate u and v components to the regular grid
            u_regular = griddata(valid_points, valid_u, (xx, yy), method='linear', fill_value=0)
            v_regular = griddata(valid_points, valid_v, (xx, yy), method='linear', fill_value=0)
            speed_regular = griddata(valid_points, valid_speed, (xx, yy), method='linear', fill_value=0)

            # Now create the streamplot with the regular grid
            strm = ax.streamplot(xx, yy, u_regular, v_regular,
                              density=1.5,  # Lower density for the regular grid
                              color=speed_regular,
                              cmap='viridis',
                              linewidth=1.5 * speed_regular + 0.5,
                              arrowsize=1.0,
                              arrowstyle='->',
                              zorder=1)

            # Add a very subtle background colored by speed
            try:
                # Only add contour if we have enough valid data points
                if np.sum(~np.isnan(speed_regular)) > 10:
                    contour = ax.contourf(xx, yy, speed_regular,
                                        levels=10,
                                        cmap='viridis',
                                        alpha=0.1,
                                        zorder=0)
            except Exception as ce:
                print(f"Skipping contour background: {ce}")
        else:
            # Longitude values are strictly increasing, we can use the original data
            # Create coordinate meshgrid for the data
            lons_mesh, lats_mesh = np.meshgrid(lons_subset, lats_subset)

            # Create a higher resolution grid for smoother streamlines
            density = 2  # Reduced density for better performance

            # Create the streamplot with color mapped to wind speed
            strm = ax.streamplot(lons_mesh, lats_mesh, u_subset, v_subset,
                              density=density,
                              color=speed_subset,
                              cmap='viridis',
                              linewidth=1.5 * speed_subset + 0.5,  # Vary line width with speed
                              arrowsize=1.0,
                              arrowstyle='->',
                              zorder=1)

            # Add a very subtle background colored by speed to enhance visibility of the flow field
            try:
                alpha_background = 0.1  # Very subtle background
                contour = ax.contourf(lons_mesh, lats_mesh, speed_subset,
                                   levels=10,
                                   cmap='viridis',
                                   alpha=alpha_background,
                                   zorder=0)
            except Exception as ce:
                print(f"Skipping contour background: {ce}")

        # Set the plot limits to match the tile boundaries exactly
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

        # Remove axes and padding
        ax.axis('off')

        # Explicitly set the figure size
        fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

        # Save the figure as a PNG with transparency
        try:
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)

            # Verify the file exists and has content
            if os.path.getsize(output_path) < 100:  # Less than 100 bytes is suspicious
                print(f"Warning: Small file size for {output_path}")
                # Try a different approach - save with PIL
                canvas = fig.canvas
                canvas.draw()
                img = Image.frombytes('RGBA', canvas.get_width_height(), canvas.buffer_rgba())
                img.save(output_path, format='PNG')
        except Exception as e:
            print(f"Error saving streamline file {output_path}: {e}")

    except Exception as e:
        # If anything fails, create an empty tile but log the error
        print(f"Error creating streamline tile: {e}")
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
        ax.axis('off')
        fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

        try:
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
        except Exception as e:
            print(f"Failed to save even the fallback streamline tile: {e}")

    plt.close(fig)

def create_temp_tile(temp, lats, lons, min_lat, max_lat, min_lon, max_lon, output_path, flight_level=None):
    """Create a raster tile for temperature data with improved boundary consistency."""
    # Close any existing figures to prevent memory leaks
    plt.close('all')

    # Create a figure with exact pixel dimensions and no padding
    fig = plt.figure(figsize=(TILE_SIZE/100, TILE_SIZE/100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    # Get the appropriate colormap based on flight level
    temp_cmap = get_temp_colormap_for_level(flight_level)

    # Use a larger buffer to ensure smooth transitions across tile boundaries
    buffer = 2.0  # Larger buffer in degrees to capture more points outside the tile
    lat_mask = (lats >= min_lat - buffer) & (lats <= max_lat + buffer)
    lon_mask = (lons >= min_lon - buffer) & (lons <= max_lon + buffer)

    # Skip if no data points in this tile
    if not np.any(lat_mask) or not np.any(lon_mask):
        # Create an empty transparent tile with a small marker to ensure it's not completely empty
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        # Add a small dot to ensure the image isn't completely empty
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
        ax.axis('off')

        # Explicitly set the figure size
        fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

        # Save with explicit format
        try:
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
        except Exception as e:
            print(f"Error saving empty tile: {e}")
        plt.close(fig)
        return

    try:
        # Extract indices for our region of interest (with buffer)
        lat_idxs = np.where(lat_mask)[0]
        lon_idxs = np.where(lon_mask)[0]

        if len(lat_idxs) < 2 or len(lon_idxs) < 2:
            # Not enough data points for interpolation - create a minimal tile
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            # Add a small dot to ensure the image isn't completely empty
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
            ax.axis('off')

            # Explicitly set the figure size
            fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

            # Save with explicit format
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
            plt.close(fig)
            return

        # Extract the relevant section of the temperature data (with buffer)
        min_lat_idx = min(lat_idxs)
        max_lat_idx = max(lat_idxs)
        min_lon_idx = min(lon_idxs)
        max_lon_idx = max(lon_idxs)

        # Get the data subset including the buffer zone
        temp_subset = temp[min_lat_idx:max_lat_idx+1, min_lon_idx:max_lon_idx+1]
        lats_subset = lats[min_lat_idx:max_lat_idx+1]
        lons_subset = lons[min_lon_idx:max_lon_idx+1]

        # Create higher resolution grid with higher density at edges
        grid_size = TILE_SIZE  # Use full tile resolution for smooth rendering

        # Create an evenly spaced grid covering the tile area plus a bit of buffer
        # to ensure the edges blend properly
        buffer_fraction = 0.1  # Small buffer for the plotting area
        plot_min_lon = min_lon - (max_lon - min_lon) * buffer_fraction
        plot_max_lon = max_lon + (max_lon - min_lon) * buffer_fraction
        plot_min_lat = min_lat - (max_lat - min_lat) * buffer_fraction
        plot_max_lat = max_lat + (max_lat - min_lat) * buffer_fraction

        lons_grid = np.linspace(plot_min_lon, plot_max_lon, grid_size)
        lats_grid = np.linspace(plot_min_lat, plot_max_lat, grid_size)
        lons_mesh, lats_mesh = np.meshgrid(lons_grid, lats_grid)

        # Combine nearest-neighbor and linear interpolation for better results
        from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

        # Prepare points for interpolation
        points = np.vstack((lons_mesh.flatten(), lats_mesh.flatten())).T

        # Create a mask of valid (non-NaN) values in the data
        valid_mask = ~np.isnan(temp_subset)
        if not np.any(valid_mask):
            # No valid data points - create a minimal tile
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            # Add a small dot to ensure the image isn't completely empty
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
            ax.axis('off')

            # Explicitly set the figure size
            fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

            # Save with explicit format
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
            plt.close(fig)
            return

        # Create coordinate grid for the data points
        lons_data_mesh, lats_data_mesh = np.meshgrid(lons_subset, lats_subset)

        # Prepare data points and values for interpolation
        data_points = np.vstack((lons_data_mesh.flatten(), lats_data_mesh.flatten())).T
        data_values = temp_subset.flatten()

        # Create the interpolator with only valid data points
        valid_points = data_points[~np.isnan(data_values)]
        valid_values = data_values[~np.isnan(data_values)]

        if len(valid_points) < 3:
            # Not enough valid points for interpolation - create a minimal tile
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            # Add a small dot to ensure the image isn't completely empty
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
            ax.axis('off')

            # Explicitly set the figure size
            fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

            # Save with explicit format
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
            plt.close(fig)
            return

        try:
            # First try linear interpolation for smooth results
            interp_linear = LinearNDInterpolator(valid_points, valid_values, fill_value=np.nan)
            temp_interp = interp_linear(points).reshape(lons_mesh.shape)

            # Fill in any NaN values with nearest neighbor interpolation
            if np.any(np.isnan(temp_interp)):
                interp_nearest = NearestNDInterpolator(valid_points, valid_values)
                nan_mask = np.isnan(temp_interp)
                temp_interp[nan_mask] = interp_nearest(points[nan_mask.flatten()]).reshape(np.sum(nan_mask))

            # Apply a slight Gaussian blur to smooth transitions further
            from scipy.ndimage import gaussian_filter
            temp_interp = gaussian_filter(temp_interp, sigma=1.0)

            # Create a smooth continuous surface with pcolormesh
            # Use the full global color scale for consistency
            im = ax.pcolormesh(lons_mesh, lats_mesh, temp_interp,
                            cmap=temp_cmap,  # Use the flight-level appropriate colormap
                            alpha=0.7,
                            shading='gouraud',  # For smoother interpolation between grid points
                            vmin=0.0, vmax=1.0)  # Use full range of normalized values

        except Exception as e:
            print(f"Advanced interpolation failed: {e}, falling back to simpler method")
            try:
                # Fallback to simpler contourf
                contour = ax.contourf(lons_data_mesh, lats_data_mesh, temp_subset,
                                    levels=np.linspace(0, 1, 20),
                                    cmap=temp_cmap,  # Use the flight-level appropriate colormap
                                    alpha=0.7,
                                    vmin=0.0, vmax=1.0,  # Use full range
                                    extend='both')
            except Exception as e2:
                print(f"Contour failed too: {e2}, falling back to basic rendering")
                # Last resort: imshow for reliable rendering
                try:
                    # Resample to a regular grid
                    from scipy.interpolate import griddata
                    xi = np.linspace(min_lon, max_lon, TILE_SIZE)
                    yi = np.linspace(min_lat, max_lat, TILE_SIZE)
                    xi_mesh, yi_mesh = np.meshgrid(xi, yi)

                    # Flatten the coordinate arrays
                    points = np.vstack((lons_data_mesh.flatten(), lats_data_mesh.flatten())).T
                    values = temp_subset.flatten()

                    # Remove NaN values
                    valid_mask = ~np.isnan(values)
                    points = points[valid_mask]
                    values = values[valid_mask]

                    # Interpolate to the regular grid
                    if len(points) > 0:
                        zi = griddata(points, values, (xi_mesh, yi_mesh), method='nearest')
                        ax.imshow(zi, origin='lower', extent=[min_lon, max_lon, min_lat, max_lat],
                                cmap=temp_cmap,  # Use the flight-level appropriate colormap
                                alpha=0.7, vmin=0.0, vmax=1.0)
                    else:
                        # If no valid points, add a minimal marker
                        center_lon = (min_lon + max_lon) / 2
                        center_lat = (min_lat + max_lat) / 2
                        ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
                except Exception as e3:
                    print(f"All interpolation methods failed: {e3}")
                    # Add a minimal marker as a last resort
                    center_lon = (min_lon + max_lon) / 2
                    center_lat = (min_lat + max_lat) / 2
                    ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)

        # Set the plot limits to match the tile boundaries exactly
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)

        # Remove axes and padding
        ax.axis('off')

        # Explicitly set the figure size
        fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

        # Save the figure as a PNG with transparency
        try:
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)

            # Verify the file exists and has content
            if os.path.getsize(output_path) < 100:  # Less than 100 bytes is suspicious
                print(f"Warning: Small file size for {output_path}")
                # Try a different approach - save with PIL
                canvas = fig.canvas
                canvas.draw()
                img = Image.frombytes('RGBA', canvas.get_width_height(), canvas.buffer_rgba())
                img.save(output_path, format='PNG')
        except Exception as e:
            print(f"Error saving file {output_path}: {e}")

        plt.close(fig)
    except Exception as e:
        # If anything fails, create an empty tile but log the error
        print(f"Error creating temperature tile: {e}")
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        # Add a small dot to ensure the image isn't completely empty
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        ax.plot(center_lon, center_lat, '.', color='gray', alpha=0.1, markersize=1)
        ax.axis('off')

        # Explicitly set the figure size
        fig.set_size_inches(TILE_SIZE/100, TILE_SIZE/100)

        # Save with explicit format
        try:
            fig.savefig(output_path, transparent=True, bbox_inches=None, pad_inches=0, format='png', dpi=100)
        except Exception as e:
            print(f"Failed to save even the fallback tile: {e}")

        plt.close(fig)