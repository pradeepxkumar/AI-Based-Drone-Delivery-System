"""
============================================================
Step 5: Pathfinding Engine (Multi-Algorithm)
============================================================
The core navigation brain for the drone delivery system.

Features:
  - A* algorithm with Haversine heuristic (great-circle distance)
  - Best-First Search (Greedy) — fastest, may not find optimal path
  - Dijkstra's Algorithm — guaranteed shortest path, no heuristic
  - Grid-based navigation over the 20x20 km area
  - Collision detection against master map (buildings + zones)
  - Path smoothing (string-pulling) for straight flight vectors
  - Yellow zone permission handling (binary: open or blocked)
  - Distance, time, and compliance metrics

Input:  source (lat,lon), destination (lat,lon), permitted yellow zones
Output: optimized path coordinates + metrics
============================================================
"""

import os
import sys
import json
import math
import heapq
import random
from collections import deque
import time
import warnings
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, box
from shapely.prepared import prep

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MASTER_MAP_FILE = os.path.join(OUTPUT_DIR, "jaipur_master_map.geojson")

# Grid resolution (in degrees)
# ~50m per cell at Jaipur latitude
GRID_STEP_LON = 0.0005   # ~50m in longitude
GRID_STEP_LAT = 0.00045  # ~50m in latitude

# Operational area bounds
BOUNDS = {
    "min_lon": 75.7006,
    "max_lon": 75.9006,
    "min_lat": 26.7573,
    "max_lat": 26.9373,
}

# Drone parameters (defaults — can be overridden by dashboard)
DRONE_SPEED_KMH = 50    # km/h (default)
DRONE_CRUISE_ALT = 60   # meters (default)
DRONE_SAFETY_MARGIN = 10 # meters clearance above buildings
DRONE_BUILDING_BUFFER = 3 # meters horizontal gap around buildings
DRONE_WIDTH = 2          # meters (for collision buffer)

# Collision buffer in degrees (~5m buffer around obstacles)
COLLISION_BUFFER_DEG = 0.00005

# A* movement directions (8-directional)
DIRECTIONS = [
    (0, 1),   # North
    (0, -1),  # South
    (1, 0),   # East
    (-1, 0),  # West
    (1, 1),   # NE
    (1, -1),  # SE
    (-1, 1),  # NW
    (-1, -1), # SW
]

# Diagonal cost multiplier
DIAG_COST = math.sqrt(2)


# ──────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points in meters."""
    R = 6371000  # Earth's radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def coord_to_grid(lon, lat):
    """Convert (lon, lat) to grid cell (col, row)."""
    col = int((lon - BOUNDS["min_lon"]) / GRID_STEP_LON)
    row = int((lat - BOUNDS["min_lat"]) / GRID_STEP_LAT)
    return col, row


def grid_to_coord(col, row):
    """Convert grid cell (col, row) to (lon, lat) center."""
    lon = BOUNDS["min_lon"] + (col + 0.5) * GRID_STEP_LON
    lat = BOUNDS["min_lat"] + (row + 0.5) * GRID_STEP_LAT
    return lon, lat


def get_grid_dimensions():
    """Get grid width and height."""
    cols = int((BOUNDS["max_lon"] - BOUNDS["min_lon"]) / GRID_STEP_LON) + 1
    rows = int((BOUNDS["max_lat"] - BOUNDS["min_lat"]) / GRID_STEP_LAT) + 1
    return cols, rows


# ──────────────────────────────────────────────
# OBSTACLE MAP
# ──────────────────────────────────────────────

class ObstacleMap:
    """
    Pre-computed obstacle grid for fast collision checking.
    Converts all obstacles (buildings, zones) into a boolean grid.
    """

    def __init__(self, master_map_path, permitted_yellow_zones=None,
                 drone_altitude=None, safety_margin=None, building_buffer=None):
        """
        Args:
            master_map_path: Path to jaipur_master_map.geojson
            permitted_yellow_zones: List of yellow zone IDs that are OPEN
                                    e.g. ["Yellow-101", "Yellow-102"]
            drone_altitude: Drone flight altitude in meters (default: 60m)
            safety_margin: Safety clearance above buildings in meters (default: 10m)
            building_buffer: Horizontal buffer around buildings in meters (default: 3m)
        """
        self.permitted_yellow = set(permitted_yellow_zones or [])
        self.drone_altitude = drone_altitude or DRONE_CRUISE_ALT
        self.safety_margin = safety_margin or DRONE_SAFETY_MARGIN
        self.building_buffer = building_buffer if building_buffer is not None else DRONE_BUILDING_BUFFER
        # Convert building buffer from meters to degrees (approx at Jaipur latitude)
        self.buffer_deg = self.building_buffer / 99855.0  # 1 degree lon ~ 99855m
        self.min_obstacle_height = self.drone_altitude - self.safety_margin
        self.cols, self.rows = get_grid_dimensions()
        self.grid = np.zeros((self.rows, self.cols), dtype=bool)  # False = free, True = blocked
        self.zone_info = {}  # zone_id -> zone data for reporting

        print(f"  Grid dimensions: {self.cols} x {self.rows} = {self.cols * self.rows:,} cells")
        print(f"  Cell size: ~{GRID_STEP_LON * 99855:.0f}m x {GRID_STEP_LAT * 111320:.0f}m")
        print(f"  Drone altitude: {self.drone_altitude}m | Safety margin: {self.safety_margin}m")
        print(f"  Building buffer: {self.building_buffer}m (horizontal gap)")
        print(f"  Buildings blocked if height >= {self.min_obstacle_height}m")

        self._load_obstacles(master_map_path)

    def _load_obstacles(self, filepath):
        """Load obstacles and mark blocked cells in the grid."""
        print(f"  Loading master map: {os.path.basename(filepath)}")

        gdf = gpd.read_file(filepath)
        print(f"  Total features: {len(gdf)}")

        blocked_count = 0
        skipped_yellow = 0
        processed = 0

        for _, row in gdf.iterrows():
            zone_type = str(row.get("zone_type", "")).strip()
            zone_id = str(row.get("zone_id", "")).strip()
            height = float(row.get("height", 0))
            name = str(row.get("name", ""))

            # Store zone info for reporting
            if zone_type in ("red", "yellow"):
                self.zone_info[zone_id] = {
                    "type": zone_type,
                    "name": name,
                    "height": height,
                    "permitted": zone_id in self.permitted_yellow if zone_type == "yellow" else False,
                }

            # Skip boundary (height = 0, it's the outer area)
            if zone_type == "boundary":
                continue

            # Handle Yellow zones
            if zone_type == "yellow":
                if zone_id in self.permitted_yellow:
                    # Permission granted — don't block this zone
                    skipped_yellow += 1
                    continue
                # No permission — treat as blocked (height = 500m)

            # Skip non-obstacle features (height = 0)
            if height <= 0:
                continue

            # Block if: (1) it's a restricted zone, or (2) building is tall enough
            # to be a threat (height >= drone_altitude - safety_margin)
            if zone_type in ("red", "yellow") or height >= self.min_obstacle_height:
                # Mark grid cells that intersect with this obstacle
                geom = row.geometry
                if geom and geom.is_valid and not geom.is_empty:
                    # Apply horizontal buffer for buildings
                    if zone_type == "building" and self.buffer_deg > 0:
                        geom = geom.buffer(self.buffer_deg)
                    cells_marked = self._mark_geometry(geom)
                    blocked_count += cells_marked
                    processed += 1

        # Calculate stats
        total_cells = self.rows * self.cols
        blocked_cells = np.sum(self.grid)
        free_pct = (1 - blocked_cells / total_cells) * 100

        print(f"\n  Obstacle Processing:")
        print(f"    Features processed: {processed}")
        print(f"    Yellow zones permitted (skipped): {skipped_yellow}")
        print(f"    Grid cells blocked: {blocked_cells:,} / {total_cells:,} ({100-free_pct:.1f}%)")
        print(f"    Grid cells free: {total_cells - blocked_cells:,} ({free_pct:.1f}%)")

    def _mark_geometry(self, geom):
        """Mark grid cells that intersect with a geometry as blocked."""
        # Get bounding box of geometry
        minx, miny, maxx, maxy = geom.bounds

        # Convert to grid range
        col_start = max(0, int((minx - BOUNDS["min_lon"]) / GRID_STEP_LON) - 1)
        col_end = min(self.cols - 1, int((maxx - BOUNDS["min_lon"]) / GRID_STEP_LON) + 1)
        row_start = max(0, int((miny - BOUNDS["min_lat"]) / GRID_STEP_LAT) - 1)
        row_end = min(self.rows - 1, int((maxy - BOUNDS["min_lat"]) / GRID_STEP_LAT) + 1)

        # Prepare geometry for faster checks
        prepared_geom = prep(geom)
        count = 0

        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                if not self.grid[r, c]:  # Only check if not already blocked
                    lon, lat = grid_to_coord(c, r)
                    # Create a small box for the grid cell
                    cell = box(
                        lon - GRID_STEP_LON/2,
                        lat - GRID_STEP_LAT/2,
                        lon + GRID_STEP_LON/2,
                        lat + GRID_STEP_LAT/2
                    )
                    if prepared_geom.intersects(cell):
                        self.grid[r, c] = True
                        count += 1

        return count

    def is_blocked(self, col, row):
        """Check if a grid cell is blocked."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return True  # Out of bounds = blocked
        return self.grid[row, col]

    def is_coord_blocked(self, lon, lat):
        """Check if a coordinate is blocked."""
        col, row = coord_to_grid(lon, lat)
        return self.is_blocked(col, row)


# ──────────────────────────────────────────────
# A* PATHFINDER
# ──────────────────────────────────────────────

class AStarPathfinder:
    """A* pathfinding algorithm for drone navigation."""

    def __init__(self, obstacle_map):
        self.obs_map = obstacle_map
        self.cols = obstacle_map.cols
        self.rows = obstacle_map.rows

    def find_path(self, start_lon, start_lat, end_lon, end_lat):
        """
        Find optimal path from start to end using A*.

        Args:
            start_lon, start_lat: Start coordinates
            end_lon, end_lat: End coordinates

        Returns:
            dict with path, metrics, and status
        """
        # Convert to grid
        start_col, start_row = coord_to_grid(start_lon, start_lat)
        end_col, end_row = coord_to_grid(end_lon, end_lat)

        print(f"\n  A* Pathfinding:")
        print(f"    Start: ({start_lat:.6f}, {start_lon:.6f}) -> Grid({start_col}, {start_row})")
        print(f"    End:   ({end_lat:.6f}, {end_lon:.6f}) -> Grid({end_col}, {end_row})")

        # Validate start and end
        if self.obs_map.is_blocked(start_col, start_row):
            print(f"    [ERROR] Start position is inside an obstacle!")
            # Try to find nearest free cell
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return {"status": "BLOCKED", "error": "Start position is blocked", "path": []}
            print(f"    [FIX] Moved start to nearest free cell: Grid({start_col}, {start_row})")

        if self.obs_map.is_blocked(end_col, end_row):
            print(f"    [ERROR] End position is inside an obstacle!")
            end_col, end_row = self._find_nearest_free(end_col, end_row)
            if end_col is None:
                return {"status": "BLOCKED", "error": "End position is blocked", "path": []}
            print(f"    [FIX] Moved end to nearest free cell: Grid({end_col}, {end_row})")

        # Straight-line distance
        direct_dist = haversine(start_lat, start_lon, end_lat, end_lon)
        print(f"    Direct distance: {direct_dist:.0f}m ({direct_dist/1000:.2f}km)")

        # A* search
        start_time = time.time()
        path_grid = self._astar(start_col, start_row, end_col, end_row)
        elapsed = time.time() - start_time

        if path_grid is None:
            print(f"    [FAILED] No path found! ({elapsed:.2f}s)")
            return {"status": "NO_PATH", "error": "No path found between points", "path": []}

        print(f"    [OK] Raw path found: {len(path_grid)} waypoints ({elapsed:.2f}s)")

        # Convert grid path to coordinates
        path_coords = [(grid_to_coord(c, r)) for c, r in path_grid]

        # Smooth the path (string-pulling)
        smoothed = self._smooth_path(path_coords)
        print(f"    Smoothed path: {len(smoothed)} waypoints")

        # Calculate metrics (use instance speed if available)
        total_dist = self._calculate_path_distance(smoothed)
        _speed = getattr(self, '_drone_speed', DRONE_SPEED_KMH)
        travel_time = (total_dist / 1000) / _speed * 60  # minutes

        print(f"    Path distance: {total_dist:.0f}m ({total_dist/1000:.2f}km)")
        print(f"    Travel time: {travel_time:.1f} min (at {DRONE_SPEED_KMH} km/h)")
        print(f"    Detour ratio: {total_dist/direct_dist:.2f}x")

        return {
            "status": "SUCCESS",
            "path": smoothed,  # List of (lon, lat) tuples
            "raw_path": path_coords,
            "metrics": {
                "direct_distance_m": round(direct_dist, 1),
                "path_distance_m": round(total_dist, 1),
                "direct_distance_km": round(direct_dist / 1000, 3),
                "path_distance_km": round(total_dist / 1000, 3),
                "travel_time_min": round(travel_time, 1),
                "travel_time_sec": round(travel_time * 60, 0),
                "waypoints": len(smoothed),
                "raw_waypoints": len(path_coords),
                "detour_ratio": round(total_dist / direct_dist, 2),
                "drone_speed_kmh": _speed,
                "drone_altitude_m": getattr(self, '_drone_altitude', DRONE_CRUISE_ALT),
                "safety_margin_m": getattr(self, '_safety_margin', DRONE_SAFETY_MARGIN),
                "grid_resolution_m": round(GRID_STEP_LON * 99855, 0),
                "computation_time_s": round(elapsed, 3),
            },
        }

    def _astar(self, start_col, start_row, end_col, end_row):
        """Core A* algorithm."""
        # Priority queue: (f_score, counter, col, row)
        counter = 0
        open_set = [(0, counter, start_col, start_row)]
        came_from = {}
        g_score = {(start_col, start_row): 0}

        while open_set:
            f, _, curr_col, curr_row = heapq.heappop(open_set)

            # Goal reached
            if curr_col == end_col and curr_row == end_row:
                return self._reconstruct_path(came_from, end_col, end_row)

            # Check g_score is still valid (may have been superseded)
            curr_g = g_score.get((curr_col, curr_row), float('inf'))
            if f > curr_g + self._heuristic(curr_col, curr_row, end_col, end_row) + 0.001:
                continue

            # Explore neighbors
            for dx, dy in DIRECTIONS:
                next_col = curr_col + dx
                next_row = curr_row + dy

                # Check bounds and obstacles
                if self.obs_map.is_blocked(next_col, next_row):
                    continue

                # Calculate movement cost
                if dx != 0 and dy != 0:
                    move_cost = DIAG_COST
                else:
                    move_cost = 1.0

                new_g = curr_g + move_cost

                if new_g < g_score.get((next_col, next_row), float('inf')):
                    g_score[(next_col, next_row)] = new_g
                    h = self._heuristic(next_col, next_row, end_col, end_row)
                    f_score = new_g + h
                    came_from[(next_col, next_row)] = (curr_col, curr_row)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, next_col, next_row))

        return None  # No path found

    def _heuristic(self, col1, row1, col2, row2):
        """Octile distance heuristic (better than Euclidean for 8-dir grid)."""
        dx = abs(col1 - col2)
        dy = abs(row1 - row2)
        return max(dx, dy) + (DIAG_COST - 1) * min(dx, dy)

    def _reconstruct_path(self, came_from, end_col, end_row):
        """Reconstruct path from came_from map."""
        path = [(end_col, end_row)]
        current = (end_col, end_row)
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _find_nearest_free(self, col, row, max_radius=50):
        """Find the nearest free cell within max_radius."""
        for r in range(1, max_radius + 1):
            for dc in range(-r, r + 1):
                for dr in range(-r, r + 1):
                    if abs(dc) == r or abs(dr) == r:
                        nc, nr = col + dc, row + dr
                        if not self.obs_map.is_blocked(nc, nr):
                            return nc, nr
        return None, None

    def _smooth_path(self, path_coords):
        """
        String-pulling algorithm to smooth the path.
        Removes unnecessary waypoints by checking line-of-sight.
        """
        if len(path_coords) <= 2:
            return path_coords

        smoothed = [path_coords[0]]
        current = 0

        while current < len(path_coords) - 1:
            # Try to reach as far ahead as possible with a straight line
            farthest = current + 1

            for ahead in range(len(path_coords) - 1, current, -1):
                if self._line_of_sight(path_coords[current], path_coords[ahead]):
                    farthest = ahead
                    break

            smoothed.append(path_coords[farthest])
            current = farthest

        return smoothed

    def _line_of_sight(self, p1, p2):
        """
        Check if there is a clear line of sight between two points.
        Uses ray-marching along the line to check for obstacles.
        """
        lon1, lat1 = p1
        lon2, lat2 = p2

        # Number of sample points along the line
        dist = math.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)
        num_samples = max(int(dist / (GRID_STEP_LON * 0.5)), 5)

        for i in range(num_samples + 1):
            t = i / num_samples
            lon = lon1 + t * (lon2 - lon1)
            lat = lat1 + t * (lat2 - lat1)

            col, row = coord_to_grid(lon, lat)
            if self.obs_map.is_blocked(col, row):
                return False

        return True

    def _calculate_path_distance(self, path_coords):
        """Calculate total path distance in meters using Haversine."""
        total = 0
        for i in range(len(path_coords) - 1):
            lon1, lat1 = path_coords[i]
            lon2, lat2 = path_coords[i + 1]
            total += haversine(lat1, lon1, lat2, lon2)
        return total


# ──────────────────────────────────────────────
# BEST-FIRST SEARCH PATHFINDER
# ──────────────────────────────────────────────

class BestFirstPathfinder(AStarPathfinder):
    """
    Greedy Best-First Search pathfinder for drone navigation.

    Uses ONLY the heuristic (h-score) to guide search — ignores
    actual path cost (g-score). This makes it very fast but the
    resulting path may NOT be the shortest.
    """

    def find_path(self, start_lon, start_lat, end_lon, end_lat):
        """
        Find path using Greedy Best-First Search.
        Fast but may produce suboptimal routes.
        """
        start_col, start_row = coord_to_grid(start_lon, start_lat)
        end_col, end_row = coord_to_grid(end_lon, end_lat)

        print(f"\n  Best-First Search Pathfinding:")
        print(f"    Start: ({start_lat:.6f}, {start_lon:.6f}) -> Grid({start_col}, {start_row})")
        print(f"    End:   ({end_lat:.6f}, {end_lon:.6f}) -> Grid({end_col}, {end_row})")

        if self.obs_map.is_blocked(start_col, start_row):
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return {"status": "BLOCKED", "error": "Start position is blocked", "path": []}

        if self.obs_map.is_blocked(end_col, end_row):
            end_col, end_row = self._find_nearest_free(end_col, end_row)
            if end_col is None:
                return {"status": "BLOCKED", "error": "End position is blocked", "path": []}

        direct_dist = haversine(start_lat, start_lon, end_lat, end_lon)
        print(f"    Direct distance: {direct_dist:.0f}m ({direct_dist/1000:.2f}km)")

        start_time = time.time()
        path_grid = self._best_first(start_col, start_row, end_col, end_row)
        elapsed = time.time() - start_time

        if path_grid is None:
            print(f"    [FAILED] No path found! ({elapsed:.2f}s)")
            return {"status": "NO_PATH", "error": "No path found between points", "path": []}

        print(f"    [OK] Raw path found: {len(path_grid)} waypoints ({elapsed:.2f}s)")

        path_coords = [(grid_to_coord(c, r)) for c, r in path_grid]
        smoothed = self._smooth_path(path_coords)
        print(f"    Smoothed path: {len(smoothed)} waypoints")

        total_dist = self._calculate_path_distance(smoothed)
        _speed = getattr(self, '_drone_speed', DRONE_SPEED_KMH)
        travel_time = (total_dist / 1000) / _speed * 60

        print(f"    Path distance: {total_dist:.0f}m ({total_dist/1000:.2f}km)")
        print(f"    Travel time: {travel_time:.1f} min")
        print(f"    Detour ratio: {total_dist/direct_dist:.2f}x")

        return {
            "status": "SUCCESS",
            "path": smoothed,
            "raw_path": path_coords,
            "metrics": {
                "direct_distance_m": round(direct_dist, 1),
                "path_distance_m": round(total_dist, 1),
                "direct_distance_km": round(direct_dist / 1000, 3),
                "path_distance_km": round(total_dist / 1000, 3),
                "travel_time_min": round(travel_time, 1),
                "travel_time_sec": round(travel_time * 60, 0),
                "waypoints": len(smoothed),
                "raw_waypoints": len(path_coords),
                "detour_ratio": round(total_dist / direct_dist, 2),
                "drone_speed_kmh": _speed,
                "drone_altitude_m": getattr(self, '_drone_altitude', DRONE_CRUISE_ALT),
                "safety_margin_m": getattr(self, '_safety_margin', DRONE_SAFETY_MARGIN),
                "grid_resolution_m": round(GRID_STEP_LON * 99855, 0),
                "computation_time_s": round(elapsed, 3),
                "algorithm": "Best-First Search",
            },
        }

    def _best_first(self, start_col, start_row, end_col, end_row):
        """
        Core Greedy Best-First Search — priority = heuristic only (h).
        Does NOT track g-score: fast but may miss shortest path.
        """
        counter = 0
        # Priority queue: (h_score, counter, col, row)
        open_set = [(0, counter, start_col, start_row)]
        came_from = {}
        visited = set()

        while open_set:
            _, _, curr_col, curr_row = heapq.heappop(open_set)

            if (curr_col, curr_row) in visited:
                continue
            visited.add((curr_col, curr_row))

            if curr_col == end_col and curr_row == end_row:
                return self._reconstruct_path(came_from, end_col, end_row)

            for dx, dy in DIRECTIONS:
                next_col = curr_col + dx
                next_row = curr_row + dy

                if self.obs_map.is_blocked(next_col, next_row):
                    continue
                if (next_col, next_row) in visited:
                    continue

                if (next_col, next_row) not in came_from:
                    came_from[(next_col, next_row)] = (curr_col, curr_row)

                h = self._heuristic(next_col, next_row, end_col, end_row)
                counter += 1
                heapq.heappush(open_set, (h, counter, next_col, next_row))

        return None


# ──────────────────────────────────────────────
# DIJKSTRA'S PATHFINDER
# ──────────────────────────────────────────────

class DijkstraPathfinder(AStarPathfinder):
    """
    Dijkstra's Algorithm pathfinder for drone navigation.

    Uses NO heuristic — explores nodes based purely on actual
    accumulated cost (g-score). Guarantees the shortest path
    but is slower than A* because it explores more cells.
    """

    def find_path(self, start_lon, start_lat, end_lon, end_lat):
        """
        Find shortest path using Dijkstra's algorithm.
        Guaranteed optimal, but slower than A*.
        """
        start_col, start_row = coord_to_grid(start_lon, start_lat)
        end_col, end_row = coord_to_grid(end_lon, end_lat)

        print(f"\n  Dijkstra's Pathfinding:")
        print(f"    Start: ({start_lat:.6f}, {start_lon:.6f}) -> Grid({start_col}, {start_row})")
        print(f"    End:   ({end_lat:.6f}, {end_lon:.6f}) -> Grid({end_col}, {end_row})")

        if self.obs_map.is_blocked(start_col, start_row):
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return {"status": "BLOCKED", "error": "Start position is blocked", "path": []}

        if self.obs_map.is_blocked(end_col, end_row):
            end_col, end_row = self._find_nearest_free(end_col, end_row)
            if end_col is None:
                return {"status": "BLOCKED", "error": "End position is blocked", "path": []}

        direct_dist = haversine(start_lat, start_lon, end_lat, end_lon)
        print(f"    Direct distance: {direct_dist:.0f}m ({direct_dist/1000:.2f}km)")

        start_time = time.time()
        path_grid = self._dijkstra(start_col, start_row, end_col, end_row)
        elapsed = time.time() - start_time

        if path_grid is None:
            print(f"    [FAILED] No path found! ({elapsed:.2f}s)")
            return {"status": "NO_PATH", "error": "No path found between points", "path": []}

        print(f"    [OK] Raw path found: {len(path_grid)} waypoints ({elapsed:.2f}s)")

        path_coords = [(grid_to_coord(c, r)) for c, r in path_grid]
        smoothed = self._smooth_path(path_coords)
        print(f"    Smoothed path: {len(smoothed)} waypoints")

        total_dist = self._calculate_path_distance(smoothed)
        _speed = getattr(self, '_drone_speed', DRONE_SPEED_KMH)
        travel_time = (total_dist / 1000) / _speed * 60

        print(f"    Path distance: {total_dist:.0f}m ({total_dist/1000:.2f}km)")
        print(f"    Travel time: {travel_time:.1f} min")
        print(f"    Detour ratio: {total_dist/direct_dist:.2f}x")

        return {
            "status": "SUCCESS",
            "path": smoothed,
            "raw_path": path_coords,
            "metrics": {
                "direct_distance_m": round(direct_dist, 1),
                "path_distance_m": round(total_dist, 1),
                "direct_distance_km": round(direct_dist / 1000, 3),
                "path_distance_km": round(total_dist / 1000, 3),
                "travel_time_min": round(travel_time, 1),
                "travel_time_sec": round(travel_time * 60, 0),
                "waypoints": len(smoothed),
                "raw_waypoints": len(path_coords),
                "detour_ratio": round(total_dist / direct_dist, 2),
                "drone_speed_kmh": _speed,
                "drone_altitude_m": getattr(self, '_drone_altitude', DRONE_CRUISE_ALT),
                "safety_margin_m": getattr(self, '_safety_margin', DRONE_SAFETY_MARGIN),
                "grid_resolution_m": round(GRID_STEP_LON * 99855, 0),
                "computation_time_s": round(elapsed, 3),
                "algorithm": "Dijkstra's",
            },
        }

    def _dijkstra(self, start_col, start_row, end_col, end_row):
        """
        Core Dijkstra's algorithm — priority = actual cost only (g), no heuristic.
        Explores uniformly — guarantees shortest path.
        """
        counter = 0
        # Priority queue: (g_score, counter, col, row)
        open_set = [(0, counter, start_col, start_row)]
        came_from = {}
        g_score = {(start_col, start_row): 0}

        while open_set:
            curr_g, _, curr_col, curr_row = heapq.heappop(open_set)

            if curr_col == end_col and curr_row == end_row:
                return self._reconstruct_path(came_from, end_col, end_row)

            # Skip stale entries
            if curr_g > g_score.get((curr_col, curr_row), float('inf')):
                continue

            for dx, dy in DIRECTIONS:
                next_col = curr_col + dx
                next_row = curr_row + dy

                if self.obs_map.is_blocked(next_col, next_row):
                    continue

                move_cost = DIAG_COST if (dx != 0 and dy != 0) else 1.0
                new_g = curr_g + move_cost

                if new_g < g_score.get((next_col, next_row), float('inf')):
                    g_score[(next_col, next_row)] = new_g
                    came_from[(next_col, next_row)] = (curr_col, curr_row)
                    counter += 1
                    heapq.heappush(open_set, (new_g, counter, next_col, next_row))

        return None


# ──────────────────────────────────────────────
# BFS PATHFINDER
# ──────────────────────────────────────────────

class BFSPathfinder(AStarPathfinder):
    """
    Breadth-First Search (BFS) pathfinder for drone navigation.

    Explores all neighbours level by level (FIFO queue).
    Guarantees the path with the FEWEST STEPS (waypoints),
    but does NOT minimise total distance like Dijkstra's.
    Slower than Best-First but always complete.
    """

    def find_path(self, start_lon, start_lat, end_lon, end_lat):
        """Find path using BFS — fewest hops guaranteed."""
        start_col, start_row = coord_to_grid(start_lon, start_lat)
        end_col, end_row = coord_to_grid(end_lon, end_lat)

        print(f"\n  BFS Pathfinding:")
        print(f"    Start: ({start_lat:.6f}, {start_lon:.6f}) -> Grid({start_col}, {start_row})")
        print(f"    End:   ({end_lat:.6f}, {end_lon:.6f}) -> Grid({end_col}, {end_row})")

        if self.obs_map.is_blocked(start_col, start_row):
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return {"status": "BLOCKED", "error": "Start position is blocked", "path": []}

        if self.obs_map.is_blocked(end_col, end_row):
            end_col, end_row = self._find_nearest_free(end_col, end_row)
            if end_col is None:
                return {"status": "BLOCKED", "error": "End position is blocked", "path": []}

        direct_dist = haversine(start_lat, start_lon, end_lat, end_lon)
        print(f"    Direct distance: {direct_dist:.0f}m ({direct_dist/1000:.2f}km)")

        start_time = time.time()
        path_grid = self._bfs(start_col, start_row, end_col, end_row)
        elapsed = time.time() - start_time

        if path_grid is None:
            print(f"    [FAILED] No path found! ({elapsed:.2f}s)")
            return {"status": "NO_PATH", "error": "No path found between points", "path": []}

        print(f"    [OK] Raw path found: {len(path_grid)} waypoints ({elapsed:.2f}s)")

        path_coords = [grid_to_coord(c, r) for c, r in path_grid]
        smoothed = self._smooth_path(path_coords)
        print(f"    Smoothed path: {len(smoothed)} waypoints")

        total_dist = self._calculate_path_distance(smoothed)
        _speed = getattr(self, '_drone_speed', DRONE_SPEED_KMH)
        travel_time = (total_dist / 1000) / _speed * 60

        print(f"    Path distance: {total_dist:.0f}m ({total_dist/1000:.2f}km)")
        print(f"    Travel time: {travel_time:.1f} min")
        print(f"    Detour ratio: {total_dist/direct_dist:.2f}x")

        return {
            "status": "SUCCESS",
            "path": smoothed,
            "raw_path": path_coords,
            "metrics": {
                "direct_distance_m": round(direct_dist, 1),
                "path_distance_m": round(total_dist, 1),
                "direct_distance_km": round(direct_dist / 1000, 3),
                "path_distance_km": round(total_dist / 1000, 3),
                "travel_time_min": round(travel_time, 1),
                "travel_time_sec": round(travel_time * 60, 0),
                "waypoints": len(smoothed),
                "raw_waypoints": len(path_coords),
                "detour_ratio": round(total_dist / direct_dist, 2),
                "drone_speed_kmh": _speed,
                "drone_altitude_m": getattr(self, '_drone_altitude', DRONE_CRUISE_ALT),
                "safety_margin_m": getattr(self, '_safety_margin', DRONE_SAFETY_MARGIN),
                "grid_resolution_m": round(GRID_STEP_LON * 99855, 0),
                "computation_time_s": round(elapsed, 3),
                "algorithm": "BFS",
            },
        }

    def _bfs(self, start_col, start_row, end_col, end_row):
        """
        Core BFS — FIFO queue, explores level by level.
        Guaranteed fewest steps (hops) to goal.
        """
        queue = deque([(start_col, start_row)])
        visited = {(start_col, start_row)}   # separate visited set
        came_from = {}                        # start node NOT stored here

        while queue:
            curr_col, curr_row = queue.popleft()  # FIFO

            if curr_col == end_col and curr_row == end_row:
                return self._reconstruct_path(came_from, end_col, end_row)

            for dx, dy in DIRECTIONS:
                next_col = curr_col + dx
                next_row = curr_row + dy

                if self.obs_map.is_blocked(next_col, next_row):
                    continue
                if (next_col, next_row) in visited:
                    continue

                visited.add((next_col, next_row))
                came_from[(next_col, next_row)] = (curr_col, curr_row)
                queue.append((next_col, next_row))

        return None


# ──────────────────────────────────────────────
# DFS PATHFINDER
# ──────────────────────────────────────────────

class DFSPathfinder(AStarPathfinder):
    """
    Depth-First Search (DFS) pathfinder for drone navigation.

    Explores as deep as possible along each branch before
    backtracking (LIFO stack). Very fast discovery but often
    produces a long, winding path — NOT optimal.
    Useful mainly for comparison / educational purposes.
    """

    def find_path(self, start_lon, start_lat, end_lon, end_lat):
        """Find path using DFS — fast but winding route."""
        start_col, start_row = coord_to_grid(start_lon, start_lat)
        end_col, end_row = coord_to_grid(end_lon, end_lat)

        print(f"\n  DFS Pathfinding:")
        print(f"    Start: ({start_lat:.6f}, {start_lon:.6f}) -> Grid({start_col}, {start_row})")
        print(f"    End:   ({end_lat:.6f}, {end_lon:.6f}) -> Grid({end_col}, {end_row})")

        if self.obs_map.is_blocked(start_col, start_row):
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return {"status": "BLOCKED", "error": "Start position is blocked", "path": []}

        if self.obs_map.is_blocked(end_col, end_row):
            end_col, end_row = self._find_nearest_free(end_col, end_row)
            if end_col is None:
                return {"status": "BLOCKED", "error": "End position is blocked", "path": []}

        direct_dist = haversine(start_lat, start_lon, end_lat, end_lon)
        print(f"    Direct distance: {direct_dist:.0f}m ({direct_dist/1000:.2f}km)")

        start_time = time.time()
        path_grid = self._dfs(start_col, start_row, end_col, end_row)
        elapsed = time.time() - start_time

        if path_grid is None:
            print(f"    [FAILED] No path found! ({elapsed:.2f}s)")
            return {"status": "NO_PATH", "error": "No path found between points", "path": []}

        print(f"    [OK] Raw path found: {len(path_grid)} waypoints ({elapsed:.2f}s)")

        path_coords = [grid_to_coord(c, r) for c, r in path_grid]
        smoothed = self._smooth_path(path_coords)
        print(f"    Smoothed path: {len(smoothed)} waypoints")

        total_dist = self._calculate_path_distance(smoothed)
        _speed = getattr(self, '_drone_speed', DRONE_SPEED_KMH)
        travel_time = (total_dist / 1000) / _speed * 60

        print(f"    Path distance: {total_dist:.0f}m ({total_dist/1000:.2f}km)")
        print(f"    Travel time: {travel_time:.1f} min")
        print(f"    Detour ratio: {total_dist/direct_dist:.2f}x")

        return {
            "status": "SUCCESS",
            "path": smoothed,
            "raw_path": path_coords,
            "metrics": {
                "direct_distance_m": round(direct_dist, 1),
                "path_distance_m": round(total_dist, 1),
                "direct_distance_km": round(direct_dist / 1000, 3),
                "path_distance_km": round(total_dist / 1000, 3),
                "travel_time_min": round(travel_time, 1),
                "travel_time_sec": round(travel_time * 60, 0),
                "waypoints": len(smoothed),
                "raw_waypoints": len(path_coords),
                "detour_ratio": round(total_dist / direct_dist, 2),
                "drone_speed_kmh": _speed,
                "drone_altitude_m": getattr(self, '_drone_altitude', DRONE_CRUISE_ALT),
                "safety_margin_m": getattr(self, '_safety_margin', DRONE_SAFETY_MARGIN),
                "grid_resolution_m": round(GRID_STEP_LON * 99855, 0),
                "computation_time_s": round(elapsed, 3),
                "algorithm": "DFS",
            },
        }

    def _dfs(self, start_col, start_row, end_col, end_row):
        """
        Core iterative DFS — LIFO stack.
        """
        stack = [(start_col, start_row)]
        visited = {(start_col, start_row)}   # separate visited set
        came_from = {}                        # start node NOT stored here

        while stack:
            curr_col, curr_row = stack.pop()   # LIFO

            if curr_col == end_col and curr_row == end_row:
                return self._reconstruct_path(came_from, end_col, end_row)

            for dx, dy in reversed(DIRECTIONS):
                next_col = curr_col + dx
                next_row = curr_row + dy

                if self.obs_map.is_blocked(next_col, next_row):
                    continue
                if (next_col, next_row) in visited:
                    continue

                visited.add((next_col, next_row))
                came_from[(next_col, next_row)] = (curr_col, curr_row)
                stack.append((next_col, next_row))

        return None


# ──────────────────────────────────────────────
# RRT PATHFINDER (Rapidly-exploring Random Tree)
# ──────────────────────────────────────────────

class RRTPathfinder(AStarPathfinder):
    """
    RRT (Rapidly-exploring Random Tree) pathfinder for drone navigation.

    Sampling-based algorithm that builds a tree of random samples
    from start toward the goal. Works well in large open spaces
    but paths are NOT optimal — they follow the random tree structure.

    Parameters:
        max_iterations: Maximum tree nodes to explore (default: 15000)
        step_size: Grid cells to extend per step (default: 5)
        goal_bias: Probability of sampling toward goal (default: 0.15)
        goal_radius: Grid cells within which goal is considered reached (default: 3)
    """

    def __init__(self, obstacle_map, max_iterations=15000, step_size=5,
                 goal_bias=0.15, goal_radius=3):
        super().__init__(obstacle_map)
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_radius = goal_radius

    def find_path(self, start_lon, start_lat, end_lon, end_lat):
        """Find path using RRT — sampling-based exploration."""
        start_col, start_row = coord_to_grid(start_lon, start_lat)
        end_col, end_row = coord_to_grid(end_lon, end_lat)

        print(f"\n  RRT Pathfinding:")
        print(f"    Start: ({start_lat:.6f}, {start_lon:.6f}) -> Grid({start_col}, {start_row})")
        print(f"    End:   ({end_lat:.6f}, {end_lon:.6f}) -> Grid({end_col}, {end_row})")

        if self.obs_map.is_blocked(start_col, start_row):
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return {"status": "BLOCKED", "error": "Start position is blocked", "path": []}

        if self.obs_map.is_blocked(end_col, end_row):
            end_col, end_row = self._find_nearest_free(end_col, end_row)
            if end_col is None:
                return {"status": "BLOCKED", "error": "End position is blocked", "path": []}

        direct_dist = haversine(start_lat, start_lon, end_lat, end_lon)
        print(f"    Direct distance: {direct_dist:.0f}m ({direct_dist/1000:.2f}km)")

        start_time = time.time()
        path_grid, nodes_explored = self._rrt(start_col, start_row, end_col, end_row)
        elapsed = time.time() - start_time

        if path_grid is None:
            print(f"    [FAILED] No path found! ({elapsed:.2f}s, {nodes_explored} nodes)")
            return {"status": "NO_PATH", "error": "No path found between points", "path": []}

        print(f"    [OK] Raw path found: {len(path_grid)} waypoints ({elapsed:.2f}s, {nodes_explored} nodes)")

        path_coords = [grid_to_coord(c, r) for c, r in path_grid]
        smoothed = self._smooth_path(path_coords)
        print(f"    Smoothed path: {len(smoothed)} waypoints")

        total_dist = self._calculate_path_distance(smoothed)
        _speed = getattr(self, '_drone_speed', DRONE_SPEED_KMH)
        travel_time = (total_dist / 1000) / _speed * 60

        print(f"    Path distance: {total_dist:.0f}m ({total_dist/1000:.2f}km)")
        print(f"    Travel time: {travel_time:.1f} min")
        print(f"    Detour ratio: {total_dist/direct_dist:.2f}x")

        return {
            "status": "SUCCESS",
            "path": smoothed,
            "raw_path": path_coords,
            "metrics": {
                "direct_distance_m": round(direct_dist, 1),
                "path_distance_m": round(total_dist, 1),
                "direct_distance_km": round(direct_dist / 1000, 3),
                "path_distance_km": round(total_dist / 1000, 3),
                "travel_time_min": round(travel_time, 1),
                "travel_time_sec": round(travel_time * 60, 0),
                "waypoints": len(smoothed),
                "raw_waypoints": len(path_coords),
                "detour_ratio": round(total_dist / direct_dist, 2),
                "drone_speed_kmh": _speed,
                "drone_altitude_m": getattr(self, '_drone_altitude', DRONE_CRUISE_ALT),
                "safety_margin_m": getattr(self, '_safety_margin', DRONE_SAFETY_MARGIN),
                "grid_resolution_m": round(GRID_STEP_LON * 99855, 0),
                "computation_time_s": round(elapsed, 3),
                "nodes_explored": nodes_explored,
                "algorithm": "RRT",
            },
        }

    def _rrt(self, start_col, start_row, end_col, end_row):
        """
        Core RRT algorithm.
        Returns (path_grid_cells, nodes_explored) or (None, nodes_explored).
        """
        # Tree stored as: node_index -> (col, row), parent[i] -> parent_index
        tree_nodes = [(start_col, start_row)]
        parent = {0: -1}  # root has no parent

        # Spatial lookup: grid cell -> node index (for nearest-neighbour)
        # Using a simple list scan — sufficient for grids of this size
        nodes_explored = 0

        for iteration in range(self.max_iterations):
            # Sample a random point (with goal bias)
            if random.random() < self.goal_bias:
                sample_col, sample_row = end_col, end_row
            else:
                sample_col = random.randint(0, self.cols - 1)
                sample_row = random.randint(0, self.rows - 1)

            # Find nearest node in tree
            nearest_idx = self._nearest_node(tree_nodes, sample_col, sample_row)
            near_col, near_row = tree_nodes[nearest_idx]

            # Steer toward sample
            new_col, new_row = self._steer(near_col, near_row, sample_col, sample_row)

            # Check if the step is collision-free
            if self._collision_free_segment(near_col, near_row, new_col, new_row):
                new_idx = len(tree_nodes)
                tree_nodes.append((new_col, new_row))
                parent[new_idx] = nearest_idx
                nodes_explored += 1

                # Check if we reached the goal
                dist_to_goal = math.sqrt((new_col - end_col)**2 + (new_row - end_row)**2)
                if dist_to_goal <= self.goal_radius:
                    # Connect directly to goal if possible
                    if self._collision_free_segment(new_col, new_row, end_col, end_row):
                        goal_idx = len(tree_nodes)
                        tree_nodes.append((end_col, end_row))
                        parent[goal_idx] = new_idx
                        nodes_explored += 1
                        path = self._trace_rrt_path(tree_nodes, parent, goal_idx)
                        return path, nodes_explored

        return None, nodes_explored

    def _nearest_node(self, tree_nodes, col, row):
        """Find index of nearest node in tree (Euclidean)."""
        best_idx = 0
        best_dist = float('inf')
        for i, (nc, nr) in enumerate(tree_nodes):
            d = (nc - col)**2 + (nr - row)**2
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _steer(self, from_col, from_row, to_col, to_row):
        """Steer from one point toward another, limited by step_size."""
        dx = to_col - from_col
        dy = to_row - from_row
        dist = math.sqrt(dx**2 + dy**2)
        if dist <= self.step_size:
            return to_col, to_row
        ratio = self.step_size / dist
        new_col = int(from_col + dx * ratio)
        new_row = int(from_row + dy * ratio)
        # Clamp to grid bounds
        new_col = max(0, min(self.cols - 1, new_col))
        new_row = max(0, min(self.rows - 1, new_row))
        return new_col, new_row

    def _collision_free_segment(self, c1, r1, c2, r2):
        """Check if the straight line between two grid cells is obstacle-free."""
        dist = max(abs(c2 - c1), abs(r2 - r1))
        if dist == 0:
            return not self.obs_map.is_blocked(c1, r1)
        steps = max(dist, 2)
        for i in range(steps + 1):
            t = i / steps
            c = int(c1 + t * (c2 - c1))
            r = int(r1 + t * (r2 - r1))
            if self.obs_map.is_blocked(c, r):
                return False
        return True

    def _trace_rrt_path(self, tree_nodes, parent, goal_idx):
        """Trace path from goal back to start through parent pointers."""
        path = []
        idx = goal_idx
        while idx != -1:
            path.append(tree_nodes[idx])
            idx = parent[idx]
        path.reverse()
        return path


# ──────────────────────────────────────────────
# THETA* PATHFINDER (Any-Angle A*)
# ──────────────────────────────────────────────

class ThetaStarPathfinder(AStarPathfinder):
    """
    Theta* pathfinder for drone navigation.

    An any-angle variant of A* that checks line-of-sight between
    the PARENT of the current node and each neighbour. If there is
    clear line-of-sight, the neighbour is connected directly to the
    grandparent — skipping intermediate grid cells.

    This produces shorter, smoother paths than A* without needing
    a separate smoothing pass. Path quality is consistently superior
    to plain A* on grid-based maps.
    """

    def find_path(self, start_lon, start_lat, end_lon, end_lat):
        """Find path using Theta* — any-angle optimal pathfinding."""
        start_col, start_row = coord_to_grid(start_lon, start_lat)
        end_col, end_row = coord_to_grid(end_lon, end_lat)

        print(f"\n  Theta* Pathfinding:")
        print(f"    Start: ({start_lat:.6f}, {start_lon:.6f}) -> Grid({start_col}, {start_row})")
        print(f"    End:   ({end_lat:.6f}, {end_lon:.6f}) -> Grid({end_col}, {end_row})")

        if self.obs_map.is_blocked(start_col, start_row):
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return {"status": "BLOCKED", "error": "Start position is blocked", "path": []}

        if self.obs_map.is_blocked(end_col, end_row):
            end_col, end_row = self._find_nearest_free(end_col, end_row)
            if end_col is None:
                return {"status": "BLOCKED", "error": "End position is blocked", "path": []}

        direct_dist = haversine(start_lat, start_lon, end_lat, end_lon)
        print(f"    Direct distance: {direct_dist:.0f}m ({direct_dist/1000:.2f}km)")

        start_time = time.time()
        path_grid, nodes_explored = self._theta_star(start_col, start_row, end_col, end_row)
        elapsed = time.time() - start_time

        if path_grid is None:
            print(f"    [FAILED] No path found! ({elapsed:.2f}s, {nodes_explored} nodes)")
            return {"status": "NO_PATH", "error": "No path found between points", "path": []}

        print(f"    [OK] Raw path found: {len(path_grid)} waypoints ({elapsed:.2f}s, {nodes_explored} nodes)")

        # Theta* already produces any-angle paths; smoothing still helps a bit
        path_coords = [grid_to_coord(c, r) for c, r in path_grid]
        smoothed = self._smooth_path(path_coords)
        print(f"    Smoothed path: {len(smoothed)} waypoints")

        total_dist = self._calculate_path_distance(smoothed)
        _speed = getattr(self, '_drone_speed', DRONE_SPEED_KMH)
        travel_time = (total_dist / 1000) / _speed * 60

        print(f"    Path distance: {total_dist:.0f}m ({total_dist/1000:.2f}km)")
        print(f"    Travel time: {travel_time:.1f} min")
        print(f"    Detour ratio: {total_dist/direct_dist:.2f}x")

        return {
            "status": "SUCCESS",
            "path": smoothed,
            "raw_path": path_coords,
            "metrics": {
                "direct_distance_m": round(direct_dist, 1),
                "path_distance_m": round(total_dist, 1),
                "direct_distance_km": round(direct_dist / 1000, 3),
                "path_distance_km": round(total_dist / 1000, 3),
                "travel_time_min": round(travel_time, 1),
                "travel_time_sec": round(travel_time * 60, 0),
                "waypoints": len(smoothed),
                "raw_waypoints": len(path_coords),
                "detour_ratio": round(total_dist / direct_dist, 2),
                "drone_speed_kmh": _speed,
                "drone_altitude_m": getattr(self, '_drone_altitude', DRONE_CRUISE_ALT),
                "safety_margin_m": getattr(self, '_safety_margin', DRONE_SAFETY_MARGIN),
                "grid_resolution_m": round(GRID_STEP_LON * 99855, 0),
                "computation_time_s": round(elapsed, 3),
                "nodes_explored": nodes_explored,
                "algorithm": "Theta*",
            },
        }

    def _theta_star(self, start_col, start_row, end_col, end_row):
        """
        Core Theta* algorithm — A* with line-of-sight parent rewiring.
        Returns (path_grid_cells, nodes_explored) or (None, nodes_explored).
        """
        counter = 0
        nodes_explored = 0
        open_set = [(0, counter, start_col, start_row)]
        came_from = {(start_col, start_row): (start_col, start_row)}  # start is its own parent
        g_score = {(start_col, start_row): 0}

        while open_set:
            f, _, curr_col, curr_row = heapq.heappop(open_set)
            curr = (curr_col, curr_row)

            # Goal reached
            if curr_col == end_col and curr_row == end_row:
                path = self._reconstruct_theta_path(came_from, end_col, end_row)
                return path, nodes_explored

            curr_g = g_score.get(curr, float('inf'))
            if f > curr_g + self._heuristic(curr_col, curr_row, end_col, end_row) + 0.001:
                continue

            nodes_explored += 1

            for dx, dy in DIRECTIONS:
                next_col = curr_col + dx
                next_row = curr_row + dy
                nxt = (next_col, next_row)

                if self.obs_map.is_blocked(next_col, next_row):
                    continue

                # --- Theta* core: try to connect via parent (line-of-sight) ---
                parent = came_from.get(curr, curr)
                par_col, par_row = parent

                if self._grid_line_of_sight(par_col, par_row, next_col, next_row):
                    # Path 2: parent -> neighbour (any-angle shortcut)
                    dist_to_nb = math.sqrt((par_col - next_col)**2 + (par_row - next_row)**2)
                    new_g = g_score.get(parent, float('inf')) + dist_to_nb
                    if new_g < g_score.get(nxt, float('inf')):
                        g_score[nxt] = new_g
                        came_from[nxt] = parent
                        h = self._heuristic(next_col, next_row, end_col, end_row)
                        counter += 1
                        heapq.heappush(open_set, (new_g + h, counter, next_col, next_row))
                else:
                    # Path 1: standard A* step (current -> neighbour)
                    move_cost = DIAG_COST if (dx != 0 and dy != 0) else 1.0
                    new_g = curr_g + move_cost
                    if new_g < g_score.get(nxt, float('inf')):
                        g_score[nxt] = new_g
                        came_from[nxt] = curr
                        h = self._heuristic(next_col, next_row, end_col, end_row)
                        counter += 1
                        heapq.heappush(open_set, (new_g + h, counter, next_col, next_row))

        return None, nodes_explored

    def _grid_line_of_sight(self, c1, r1, c2, r2):
        """
        Bresenham-style line-of-sight check between two grid cells.
        Returns True if all cells along the line are free.
        """
        dc = abs(c2 - c1)
        dr = abs(r2 - r1)
        sc = 1 if c2 > c1 else -1
        sr = 1 if r2 > r1 else -1

        c, r = c1, r1

        if dc >= dr:
            err = dc // 2
            for _ in range(dc + 1):
                if self.obs_map.is_blocked(c, r):
                    return False
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
        else:
            err = dr // 2
            for _ in range(dr + 1):
                if self.obs_map.is_blocked(c, r):
                    return False
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr

        return True

    def _reconstruct_theta_path(self, came_from, end_col, end_row):
        """Reconstruct path for Theta* (parent may skip cells)."""
        path = [(end_col, end_row)]
        current = (end_col, end_row)
        while came_from.get(current) != current:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# ──────────────────────────────────────────────
# RRT* PATHFINDER (Optimal RRT)
# ──────────────────────────────────────────────

class RRTStarPathfinder(RRTPathfinder):
    """
    RRT* (Optimal RRT) pathfinder for drone navigation.

    Extends RRT with a rewiring step: after adding a new node,
    it checks whether nearby nodes can be reached more cheaply
    through the new node and rewires the tree accordingly.

    This makes RRT* asymptotically optimal — as iterations increase,
    the path converges to the true shortest path. Slower than RRT
    but produces significantly better paths.

    Parameters:
        max_iterations: Maximum tree nodes (default: 15000)
        step_size: Grid cells per step (default: 5)
        goal_bias: Probability of sampling toward goal (default: 0.15)
        goal_radius: Goal proximity threshold (default: 3)
        rewire_radius: Radius for rewiring neighbours (default: 15)
    """

    def __init__(self, obstacle_map, max_iterations=15000, step_size=5,
                 goal_bias=0.15, goal_radius=3, rewire_radius=15):
        super().__init__(obstacle_map, max_iterations, step_size,
                         goal_bias, goal_radius)
        self.rewire_radius = rewire_radius

    def find_path(self, start_lon, start_lat, end_lon, end_lat):
        """Find path using RRT* — asymptotically optimal sampling."""
        start_col, start_row = coord_to_grid(start_lon, start_lat)
        end_col, end_row = coord_to_grid(end_lon, end_lat)

        print(f"\n  RRT* Pathfinding:")
        print(f"    Start: ({start_lat:.6f}, {start_lon:.6f}) -> Grid({start_col}, {start_row})")
        print(f"    End:   ({end_lat:.6f}, {end_lon:.6f}) -> Grid({end_col}, {end_row})")

        if self.obs_map.is_blocked(start_col, start_row):
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return {"status": "BLOCKED", "error": "Start position is blocked", "path": []}

        if self.obs_map.is_blocked(end_col, end_row):
            end_col, end_row = self._find_nearest_free(end_col, end_row)
            if end_col is None:
                return {"status": "BLOCKED", "error": "End position is blocked", "path": []}

        direct_dist = haversine(start_lat, start_lon, end_lat, end_lon)
        print(f"    Direct distance: {direct_dist:.0f}m ({direct_dist/1000:.2f}km)")

        start_time = time.time()
        path_grid, nodes_explored = self._rrt_star(start_col, start_row, end_col, end_row)
        elapsed = time.time() - start_time

        if path_grid is None:
            print(f"    [FAILED] No path found! ({elapsed:.2f}s, {nodes_explored} nodes)")
            return {"status": "NO_PATH", "error": "No path found between points", "path": []}

        print(f"    [OK] Raw path found: {len(path_grid)} waypoints ({elapsed:.2f}s, {nodes_explored} nodes)")

        path_coords = [grid_to_coord(c, r) for c, r in path_grid]
        smoothed = self._smooth_path(path_coords)
        print(f"    Smoothed path: {len(smoothed)} waypoints")

        total_dist = self._calculate_path_distance(smoothed)
        _speed = getattr(self, '_drone_speed', DRONE_SPEED_KMH)
        travel_time = (total_dist / 1000) / _speed * 60

        print(f"    Path distance: {total_dist:.0f}m ({total_dist/1000:.2f}km)")
        print(f"    Travel time: {travel_time:.1f} min")
        print(f"    Detour ratio: {total_dist/direct_dist:.2f}x")

        return {
            "status": "SUCCESS",
            "path": smoothed,
            "raw_path": path_coords,
            "metrics": {
                "direct_distance_m": round(direct_dist, 1),
                "path_distance_m": round(total_dist, 1),
                "direct_distance_km": round(direct_dist / 1000, 3),
                "path_distance_km": round(total_dist / 1000, 3),
                "travel_time_min": round(travel_time, 1),
                "travel_time_sec": round(travel_time * 60, 0),
                "waypoints": len(smoothed),
                "raw_waypoints": len(path_coords),
                "detour_ratio": round(total_dist / direct_dist, 2),
                "drone_speed_kmh": _speed,
                "drone_altitude_m": getattr(self, '_drone_altitude', DRONE_CRUISE_ALT),
                "safety_margin_m": getattr(self, '_safety_margin', DRONE_SAFETY_MARGIN),
                "grid_resolution_m": round(GRID_STEP_LON * 99855, 0),
                "computation_time_s": round(elapsed, 3),
                "nodes_explored": nodes_explored,
                "algorithm": "RRT*",
            },
        }

    def _rrt_star(self, start_col, start_row, end_col, end_row):
        """
        Core RRT* algorithm with rewiring.
        Returns (path_grid_cells, nodes_explored) or (None, nodes_explored).
        """
        tree_nodes = [(start_col, start_row)]
        parent = {0: -1}
        cost = {0: 0.0}  # cost from start to each node
        nodes_explored = 0

        best_goal_idx = None
        best_goal_cost = float('inf')

        for iteration in range(self.max_iterations):
            # Sample with goal bias
            if random.random() < self.goal_bias:
                sample_col, sample_row = end_col, end_row
            else:
                sample_col = random.randint(0, self.cols - 1)
                sample_row = random.randint(0, self.rows - 1)

            # Find nearest node
            nearest_idx = self._nearest_node(tree_nodes, sample_col, sample_row)
            near_col, near_row = tree_nodes[nearest_idx]

            # Steer
            new_col, new_row = self._steer(near_col, near_row, sample_col, sample_row)

            if not self._collision_free_segment(near_col, near_row, new_col, new_row):
                continue

            new_idx = len(tree_nodes)
            tree_nodes.append((new_col, new_row))
            nodes_explored += 1

            # --- RRT* rewire: find best parent among nearby nodes ---
            near_indices = self._nodes_in_radius(tree_nodes, new_col, new_row, self.rewire_radius)

            # Choose best parent
            best_parent = nearest_idx
            best_cost = cost[nearest_idx] + math.sqrt(
                (near_col - new_col)**2 + (near_row - new_row)**2)

            for ni in near_indices:
                if ni == new_idx:
                    continue
                nc, nr = tree_nodes[ni]
                seg_cost = math.sqrt((nc - new_col)**2 + (nr - new_row)**2)
                candidate_cost = cost.get(ni, float('inf')) + seg_cost
                if candidate_cost < best_cost:
                    if self._collision_free_segment(nc, nr, new_col, new_row):
                        best_cost = candidate_cost
                        best_parent = ni

            parent[new_idx] = best_parent
            cost[new_idx] = best_cost

            # --- Rewire nearby nodes through new node ---
            for ni in near_indices:
                if ni == new_idx or ni == 0:
                    continue
                nc, nr = tree_nodes[ni]
                seg_cost = math.sqrt((nc - new_col)**2 + (nr - new_row)**2)
                new_candidate_cost = cost[new_idx] + seg_cost
                if new_candidate_cost < cost.get(ni, float('inf')):
                    if self._collision_free_segment(new_col, new_row, nc, nr):
                        parent[ni] = new_idx
                        cost[ni] = new_candidate_cost

            # Check goal proximity
            dist_to_goal = math.sqrt((new_col - end_col)**2 + (new_row - end_row)**2)
            if dist_to_goal <= self.goal_radius:
                if self._collision_free_segment(new_col, new_row, end_col, end_row):
                    goal_seg = math.sqrt((new_col - end_col)**2 + (new_row - end_row)**2)
                    total_goal_cost = cost[new_idx] + goal_seg
                    if total_goal_cost < best_goal_cost:
                        # Add or update goal node
                        if best_goal_idx is None:
                            best_goal_idx = len(tree_nodes)
                            tree_nodes.append((end_col, end_row))
                        parent[best_goal_idx] = new_idx
                        cost[best_goal_idx] = total_goal_cost
                        best_goal_cost = total_goal_cost

        if best_goal_idx is not None:
            path = self._trace_rrt_path(tree_nodes, parent, best_goal_idx)
            return path, nodes_explored

        return None, nodes_explored

    def _nodes_in_radius(self, tree_nodes, col, row, radius):
        """Find all node indices within a given radius."""
        radius_sq = radius ** 2
        result = []
        for i, (nc, nr) in enumerate(tree_nodes):
            if (nc - col)**2 + (nr - row)**2 <= radius_sq:
                result.append(i)
        return result


# ──────────────────────────────────────────────
# ZONE COMPLIANCE CHECK
# ──────────────────────────────────────────────

def check_zone_compliance(path_coords, master_map_path, permitted_yellow):
    """
    Check which zones the path passes through.
    Returns compliance report.
    """
    gdf = gpd.read_file(master_map_path)
    path_line = LineString(path_coords)

    zones_crossed = []
    zones_avoided = []

    for _, row in gdf.iterrows():
        zone_type = str(row.get("zone_type", "")).strip()
        zone_id = str(row.get("zone_id", "")).strip()
        name = str(row.get("name", ""))

        if zone_type not in ("red", "yellow"):
            continue

        geom = row.geometry
        if geom and geom.is_valid:
            if path_line.intersects(geom):
                zones_crossed.append({
                    "zone_id": zone_id,
                    "type": zone_type,
                    "name": name,
                    "permitted": zone_id in permitted_yellow if zone_type == "yellow" else False,
                })
            else:
                zones_avoided.append({
                    "zone_id": zone_id,
                    "type": zone_type,
                    "name": name,
                })

    # Check for violations
    violations = [z for z in zones_crossed
                  if z["type"] == "red" or (z["type"] == "yellow" and not z["permitted"])]

    return {
        "compliant": len(violations) == 0,
        "zones_crossed": zones_crossed,
        "zones_avoided": zones_avoided,
        "violations": violations,
        "yellow_zones_used": [z for z in zones_crossed if z["type"] == "yellow" and z["permitted"]],
    }


# ──────────────────────────────────────────────
# SAVE PATH
# ──────────────────────────────────────────────

def save_path_csv(path_coords, output_path, metrics=None):
    """Save path to CSV file."""
    import csv
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["waypoint", "longitude", "latitude"])
        for i, (lon, lat) in enumerate(path_coords):
            writer.writerow([i, round(lon, 8), round(lat, 8)])

    print(f"  Path saved to: {output_path}")


def save_path_geojson(path_coords, output_path):
    """Save path as GeoJSON LineString for visualization."""
    feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"type": "flight_path"},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lon, lat in path_coords]
                }
            },
            {
                "type": "Feature",
                "properties": {"type": "start_point", "label": "PICKUP"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [path_coords[0][0], path_coords[0][1]]
                }
            },
            {
                "type": "Feature",
                "properties": {"type": "end_point", "label": "DROP"},
                "geometry": {
                    "type": "Point",
                    "coordinates": [path_coords[-1][0], path_coords[-1][1]]
                }
            }
        ]
    }

    with open(output_path, "w") as f:
        json.dump(feature, f, indent=2)
    print(f"  Path GeoJSON saved to: {output_path}")


# ──────────────────────────────────────────────
# PUBLIC API (used by dashboard)
# ──────────────────────────────────────────────

# Algorithm name -> class mapping
ALGORITHM_MAP = {
    "A*": AStarPathfinder,
    "Best-First Search": BestFirstPathfinder,
    "BFS": BFSPathfinder,
    "Theta*": ThetaStarPathfinder,
    "RRT*": RRTStarPathfinder,
}


def compute_path(start_lat, start_lon, end_lat, end_lon,
                 permitted_yellow_zones=None,
                 master_map_path=None,
                 drone_altitude=None,
                 drone_speed=None,
                 safety_margin=None,
                 building_buffer=None,
                 algorithm="A*"):
    """
    Main API function — compute optimal drone path.

    Args:
        start_lat, start_lon: Pickup coordinates
        end_lat, end_lon: Drop coordinates
        permitted_yellow_zones: List of yellow zone IDs with permission
        master_map_path: Path to master map GeoJSON
        drone_altitude: Flight altitude in meters (default: 60m)
        drone_speed: Speed in km/h (default: 50 km/h)
        safety_margin: Safety clearance in meters (default: 10m)
        building_buffer: Horizontal buffer around buildings in meters (default: 3m)

    Returns:
        dict with path, metrics, compliance info
    """
    # Validate algorithm choice
    if algorithm not in ALGORITHM_MAP:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(ALGORITHM_MAP.keys())}")

    if master_map_path is None:
        master_map_path = MASTER_MAP_FILE

    permitted = permitted_yellow_zones or []
    altitude = drone_altitude or DRONE_CRUISE_ALT
    speed = drone_speed or DRONE_SPEED_KMH
    margin = safety_margin or DRONE_SAFETY_MARGIN
    buffer = building_buffer if building_buffer is not None else DRONE_BUILDING_BUFFER

    print(f"\n{'='*60}")
    print(f"  DRONE PATH COMPUTATION")
    print(f"{'='*60}")
    print(f"  Pickup:  ({start_lat:.6f}, {start_lon:.6f})")
    print(f"  Drop:    ({end_lat:.6f}, {end_lon:.6f})")
    print(f"  Algorithm:      {algorithm}")
    print(f"  Drone Altitude: {altitude}m | Speed: {speed} km/h | Safety: {margin}m | Buffer: {buffer}m")
    print(f"  Permitted Yellow Zones: {len(permitted)}")
    if permitted:
        print(f"    {', '.join(permitted[:10])}")
        if len(permitted) > 10:
            print(f"    ... and {len(permitted)-10} more")

    # 1. Build obstacle map
    print(f"\n  [1/4] Building obstacle map...")
    obs_map = ObstacleMap(master_map_path, permitted,
                          drone_altitude=altitude, safety_margin=margin,
                          building_buffer=buffer)

    # 2. Run selected pathfinder
    PathfinderClass = ALGORITHM_MAP[algorithm]
    print(f"\n  [2/4] Running {algorithm} pathfinder...")
    pathfinder = PathfinderClass(obs_map)
    pathfinder._drone_speed = speed
    pathfinder._drone_altitude = altitude
    pathfinder._safety_margin = margin
    result = pathfinder.find_path(start_lon, start_lat, end_lon, end_lat)

    # Tag the result with algorithm name
    if result.get("metrics"):
        result["metrics"].setdefault("algorithm", algorithm)

    if result["status"] != "SUCCESS":
        print(f"\n  [FAILED] {result.get('error', 'Unknown error')}")
        return result

    # 3. Check zone compliance
    print(f"\n  [3/4] Checking zone compliance...")
    compliance = check_zone_compliance(result["path"], master_map_path, set(permitted))
    result["compliance"] = compliance

    if compliance["compliant"]:
        print(f"    [PASS] Route is fully compliant!")
    else:
        print(f"    [WARN] Route has {len(compliance['violations'])} violations!")

    if compliance["yellow_zones_used"]:
        print(f"    Yellow zones used (with permission):")
        for z in compliance["yellow_zones_used"]:
            print(f"      {z['zone_id']}: {z['name']}")

    # 4. Save outputs
    print(f"\n  [4/4] Saving results...")
    csv_path = os.path.join(OUTPUT_DIR, "mission_path.csv")
    geojson_path = os.path.join(OUTPUT_DIR, "mission_path.geojson")
    save_path_csv(result["path"], csv_path)
    save_path_geojson(result["path"], geojson_path)

    # Final summary
    m = result["metrics"]
    print(f"\n  {'='*55}")
    print(f"  MISSION SUMMARY")
    print(f"  {'='*55}")
    print(f"    Status:          {result['status']}")
    print(f"    Direct distance: {m['direct_distance_km']:.3f} km")
    print(f"    Path distance:   {m['path_distance_km']:.3f} km")
    print(f"    Detour ratio:    {m['detour_ratio']}x")
    print(f"    Travel time:     {m['travel_time_min']:.1f} min")
    print(f"    Waypoints:       {m['waypoints']}")
    print(f"    Computation:     {m['computation_time_s']:.3f}s")
    print(f"    Compliant:       {'YES' if compliance['compliant'] else 'NO'}")
    print(f"  {'='*55}")

    return result


# ──────────────────────────────────────────────
# DEMO / TEST
# ──────────────────────────────────────────────

def main():
    """Run a test mission to verify the pathfinder works."""
    print("=" * 60)
    print("  STEP 5: A* Pathfinder Test Run")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MASTER_MAP_FILE):
        print(f"\n  [ERROR] Master map not found: {MASTER_MAP_FILE}")
        print(f"  Run Step 4 first!")
        sys.exit(1)

    # Test Mission: From north Jaipur to south (crossing multiple zones)
    # Pickup: Near Nahargarh (north)
    # Drop: Near Jawahar Circle (south)
    start_lat, start_lon = 26.920, 75.780
    end_lat, end_lon = 26.850, 75.810

    # Test with some yellow zones permitted
    permitted = ["Yellow-101", "Yellow-107", "Yellow-122"]

    result = compute_path(
        start_lat, start_lon,
        end_lat, end_lon,
        permitted_yellow_zones=permitted
    )

    if result["status"] == "SUCCESS":
        print(f"\n  [OK] Pathfinder test PASSED!")
        print(f"  Output files:")
        print(f"    - mission_path.csv")
        print(f"    - mission_path.geojson")
        print(f"\n  Ready for Step 6 (Streamlit Dashboard).")
    else:
        print(f"\n  [WARN] Test returned: {result['status']}")
        print(f"  The pathfinder works but this specific route may be blocked.")

    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
