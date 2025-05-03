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

from config import TILE_SIZE
from colormap import get_temp_colormap_for_level, WIND_CMAP
from utils import ensure_dir_exists

def generate_raster_tiles(lats, lons, u_data, v_data, temp_data, zoom_levels, flight_level, output_dir, num_workers):
    """Generate raster PNG tiles for wind and temperature data."""
    print(f"Generating raster tiles for zoom levels {zoom_levels}...")

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

    # Generate separate wind and temperature tiles for each zoom level
    for z in zoom_levels:
        tiles_processed = 0

        # Calculate bounds of the tile grid at this zoom level
        num_tiles = 2**z

        # Process each tile at this zoom level
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for x in range(num_tiles):
                for y in range(num_tiles):
                    # Calculate the lat/lon bounds of this tile
                    tile_bounds = mercantile.bounds(x, y, z)
                    min_lon, min_lat, max_lon, max_lat = tile_bounds

                    # Skip tiles that are outside the data bounds or in polar regions
                    if min_lat > 85.05 or max_lat < -85.05:
                        continue

                    # Create directory structure
                    wind_tile_dir = os.path.join(output_dir, "wind", flight_level, str(z), str(x))
                    temp_tile_dir = os.path.join(output_dir, "temp", flight_level, str(z), str(x))
                    ensure_dir_exists(wind_tile_dir)
                    ensure_dir_exists(temp_tile_dir)

                    # Submit tasks to the thread pool
                    futures.append(executor.submit(create_wind_tile, u_array, v_array, norm_wind_speed, lats, lons,
                                                   min_lat, max_lat, min_lon, max_lon,
                                                   os.path.join(wind_tile_dir, f"{y}.png")))

                    if temp_data is not None:
                        futures.append(executor.submit(create_temp_tile, norm_temp, lats, lons,
                                                       min_lat, max_lat, min_lon, max_lon,
                                                       os.path.join(temp_tile_dir, f"{y}.png"), flight_level))

            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    tiles_processed += 1
                except Exception as e:
                    print(f"Error processing tile: {e}")

        print(f"Completed zoom level {z}: {tiles_processed} raster tiles")

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