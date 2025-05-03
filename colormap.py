#!/usr/bin/env python3
"""
Wind & Temperature Tile Server - Colormaps
Colormap definitions for visualizing temperature and wind data.
"""

from matplotlib.colors import LinearSegmentedColormap

# Default temperature colormap (for mid-levels)
TEMP_CMAP = LinearSegmentedColormap.from_list("temperature", [
    (0.0, '#00274D'),  # Deep navy blue (very cold)
    (0.2, '#005EA8'),  # Cool blue
    (0.4, '#00BFFF'),  # Sky blue
    (0.6, '#7FFF00'),  # Chartreuse (mild)
    (0.8, '#FFD700'),  # Gold (warm)
    (1.0, '#FF4500'),  # Orange-red (hot)
])

# High altitude temperature colormap (for pressure levels < 300hPa - colder temps)
HIGH_ALT_TEMP_CMAP = LinearSegmentedColormap.from_list("high_altitude_temperature", [
    (0.0, '#001F3F'),  # Almost black-blue (very cold)
    (0.2, '#004080'),  # Deep cold blue
    (0.4, '#0074D9'),  # Medium blue
    (0.6, '#7FDBFF'),  # Light sky blue
    (0.8, '#B0E0E6'),  # Pale blue (warmer)
    (1.0, '#CCCCFF'),  # Soft lavender (least cold)
])

# Low altitude temperature colormap (for pressure levels > 500hPa - warmer temps)
LOW_ALT_TEMP_CMAP = LinearSegmentedColormap.from_list("low_altitude_temperature", [
    (0.0, '#3366CC'),  # Cooler blue
    (0.2, '#66CCCC'),  # Aqua
    (0.4, '#99FF99'),  # Pale green
    (0.6, '#FFDB58'),  # Mustard yellow
    (0.8, '#FF8C00'),  # Dark orange
    (1.0, '#FF0000'),  # Bright red
])

# Wind speed colormap
WIND_CMAP = LinearSegmentedColormap.from_list("wind", [
    (0.0, '#0000FF'),  # Light winds (blue)
    (0.3, '#00FF00'),  # Moderate winds (green)
    (0.6, '#FFFF00'),  # Strong winds (yellow)
    (1.0, '#FF0000'),  # Severe winds (red)
])

def get_temp_colormap_for_level(flight_level):
    """Get appropriate temperature colormap for a specific flight level."""
    try:
        if isinstance(flight_level, str) and "hPa" in flight_level:
            pressure = int(flight_level.replace("hPa", ""))

            # High altitude (cold temperatures - blue to purple)
            if pressure <= 300:
                return HIGH_ALT_TEMP_CMAP
            # Low altitude (warmer temperatures - blue to red)
            elif pressure >= 500:
                return LOW_ALT_TEMP_CMAP
            # Mid altitudes - use default colormap
            else:
                return TEMP_CMAP
        else:
            # Default to standard colormap if we can't parse the flight level
            return TEMP_CMAP
    except (ValueError, TypeError):
        # Default colormap if there's an error
        return TEMP_CMAP

def get_temp_range_for_level(flight_level):
    """Get appropriate temperature range (in Celsius) for a specific flight level.
    Based on standard atmospheric model and typical temperature ranges at different pressure levels.
    """
    from config import MIN_TEMP_CELSIUS, MAX_TEMP_CELSIUS

    # Extract numeric value from flight level string (e.g. "500hPa" -> 500)
    try:
        if isinstance(flight_level, str) and "hPa" in flight_level:
            pressure = int(flight_level.replace("hPa", ""))
        else:
            # If we can't parse the flight level, use a default range
            return MIN_TEMP_CELSIUS, MAX_TEMP_CELSIUS

        # Define temperature ranges based on standard atmospheric model
        # Pressure (hPa) -> (min_temp_C, max_temp_C)
        pressure_temp_ranges = {
            200: (-75, -30),   # Upper troposphere / lower stratosphere
            250: (-70, -30),
            300: (-65, -25),
            400: (-60, -20),
            500: (-55, -15),   # Mid-troposphere
            600: (-45, -5),
            700: (-35, 10),    # Lower troposphere
            800: (-25, 20),
            850: (-20, 25),
            900: (-15, 30),
            950: (-10, 35),
            1000: (-10, 40)    # Surface
        }

        # Find the closest pressure level in our defined ranges
        closest_pressure = min(pressure_temp_ranges.keys(), key=lambda x: abs(x - pressure))
        return pressure_temp_ranges[closest_pressure]

    except (ValueError, TypeError):
        # Default range if we can't determine the appropriate range
        return MIN_TEMP_CELSIUS, MAX_TEMP_CELSIUS