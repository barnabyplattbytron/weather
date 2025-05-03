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
                    "tile_raster": "/tiles/raster/{level}/{time}/{z}/{x}/{y}.png",
                    "tile_vector": "/tiles/vector/{level}/{time}/{z}/{x}/{y}.pbf",
                    "tile_wind_map": "/tiles/wind-map/{level}/{time}/{overlay_type}/{z}/{x}/{y}.png",
                    "clear_cache": "/tiles/clear-cache"
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

        # Serve index.html for root path
        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(self.generate_index_html().encode())
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