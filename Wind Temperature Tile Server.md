
# 🌪️ Wind & Temperature Tile Server – Product Requirements Document (PRD)

## 📌 Overview

We are building a Python-based tile server that converts GRIB-format weather data (wind & temperature) into XYZ-based vector tiles. These tiles will be used in a web or mobile map application (e.g. React Native with Mapbox or Leaflet) to render live, interactive wind barbs and temperature overlays at all flight levels.

---

## 🎯 Goals

- Convert GRIB weather files into Z/X/Y tile format.
- Output GeoJSON vector tiles for frontend consumption.
- Include wind direction, speed, U/V components, and optionally temperature.
- Tiles should be generated for specified zoom levels and heigh levels.
- Support for web-accessible directory-based tile structure.
- Prepare data for client-side rendering of wind barbs.

---

## 📂 Input

- Folder: `data/`
- Files: GRIB files containing:
  - Wind: U (eastward) and V (northward) components
  - Optionally temperature
  - Files named like: `wind-006_012.GRIB`, `temp-006_012.GRIB`

---

## 📤 Output

- Folder: `tiles/{flightLevel}/{z}/{x}/{y}.geojson`
- Each tile is a GeoJSON FeatureCollection:
  ```json
  {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": { "type": "Point", "coordinates": [lon, lat] },
        "properties": {
          "u": float,
          "v": float,
          "speed": float,
          "direction": float,
          "temperature": float (optional)
        }
      }
    ]
  }
  ```

---

## ⚙️ Functionality

### 1. GRIB File Parsing

- Use `cfgrib` with `xarray` to extract data.
- Support both wind and temperature layers.
- Handle multi-time or multi-level GRIBs (start with first timestep/level).

### 2. Data Processing

- Reproject lat/lon to Web Mercator (EPSG:3857) using `pyproj`.
- Compute wind speed and direction:
  ```python
  speed = sqrt(u**2 + v**2)
  direction = (270 - atan2(v, u) * 180 / π) % 360
  ```

### 3. Tile Generation

- Use `mercantile` to compute which tile each point belongs to.
- Group features per Z/X/Y.
- Save each tile as a separate `.geojson` file.

### 4. Configurability

- Zoom levels configurable in script: e.g., `[0, 1, 2, 3, 4]`
- Output directory structure auto-created if not present.

---

## 📦 Libraries/Dependencies

Install with:

```bash
pip install xarray cfgrib pyproj mercantile shapely geojson numpy
```

Optional dev tools:
- `http.server` (for local tile serving)
- `watchdog` (for auto-regeneration if desired)

---

## 🔮 Future Enhancements (Stretch Goals)

- Add support for Mapbox Vector Tiles (MVT) instead of GeoJSON.
- Include timestamp + level filtering (multiple timesteps).
- Dockerize the pipeline for deployment.
- Generate PNG raster tiles using Metview or matplotlib.
- Add web interface to browse available tiles/times.

---

## 🧪 Testing

- Validate output tiles open in QGIS, Mapbox Studio, or Leaflet.
- Spot check tiles at different zoom levels.
- Compare barbs to source NetCDF/GRIB values for accuracy.

---

## 📁 Project Structure

```
project/
│
├── data/
│   └── wind-006_012.GRIB
│
├── tiles/
│   └── {z}/{x}/{y}.geojson
│
├── grib_to_tiles.py
└── requirements.txt
```

---

## ✅ Acceptance Criteria

- ✅ Can ingest a GRIB file and generate vector tiles for 3–5 zoom levels.
- ✅ Each tile contains U/V, speed, and direction.
- ✅ GeoJSON tiles can be loaded into Mapbox or Leaflet without errors.
- ✅ Runtime < 2 min for global file at moderate resolution (0.25°).
