"""
============================================================
Step 3: Procedurally Generate 1000+ Clustered Buildings
============================================================
This script:
  1. Loads 89 real buildings from buildings_raw.geojson
  2. Creates 30 neighborhood cluster centers across the 20km area
  3. Generates ~33 buildings per cluster with random offsets
  4. Ensures minimum gap between buildings so drone can navigate
  5. Heights: 30-60m (at drone cruise altitude)
  6. Uses Shapely for proper polygon footprints
  7. Saves to output/buildings_simulated.geojson
============================================================
"""

import os
import sys
import json
import random
import warnings
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Operational area bounds (from map.geojson)
BOUNDS = {
    "min_lon": 75.7006,
    "max_lon": 75.9006,
    "min_lat": 26.7573,
    "max_lat": 26.9373,
}

# Simulation parameters
NUM_CLUSTERS = 30          # Number of neighborhood hubs
BUILDINGS_PER_CLUSTER = 33 # Buildings around each hub
MIN_GAP_METERS = 18        # Minimum gap between buildings (drone width + safety)
MIN_HEIGHT = 30.0          # Minimum building height (meters)
MAX_HEIGHT = 60.0          # Maximum building height (meters)

# Building footprint size (in degrees, approximate)
# At Jaipur latitude: 1 degree lon ~ 99,855m, 1 degree lat ~ 111,320m
# Building size: 15-40m -> 0.00015 to 0.0004 degrees
BUILDING_SIZE_MIN = 0.00015  # ~15m
BUILDING_SIZE_MAX = 0.00040  # ~40m

# Cluster spread (how far buildings spread from cluster center)
CLUSTER_SPREAD = 0.003  # ~300m radius

# Min gap in degrees (18m)
MIN_GAP_DEG = MIN_GAP_METERS / 99855.0  # Convert meters to degrees

# Red zone buffer - don't place buildings inside red zones
RED_ZONE_IDS = ["Red-101", "Red-102", "Red-103", "Red-104", "Red-105",
                "Red-106", "Red-107", "Red-108", "Red-109", "Red-110",
                "Red-111", "Red-112"]

# Random seed
random.seed(42)
np.random.seed(42)


def load_real_buildings(filepath):
    """Load the real buildings from Step 2."""
    if not os.path.exists(filepath):
        print(f"  [WARNING] No real buildings file found: {filepath}")
        return gpd.GeoDataFrame()
    gdf = gpd.read_file(filepath)
    print(f"  Loaded {len(gdf)} real buildings from buildings_raw.geojson")
    return gdf


def load_red_zones(map_path):
    """Load red zones from map.geojson to avoid placing buildings in them."""
    if not os.path.exists(map_path):
        print(f"  [WARNING] map.geojson not found. Skipping zone avoidance.")
        return []

    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    red_polygons = []
    for feature in raw["features"]:
        props = feature.get("properties", {})
        zone_id = props.get("zone_id") or props.get("zone-id") or ""
        zone_type = (props.get("type") or props.get("type ") or "").strip().lower()

        if zone_type == "red" or "Red" in str(zone_id):
            try:
                coords = feature["geometry"]["coordinates"][0]
                poly = Polygon(coords)
                if poly.is_valid:
                    red_polygons.append(poly)
            except Exception:
                pass

    print(f"  Loaded {len(red_polygons)} red zone polygons for avoidance")
    return red_polygons


def create_random_polygon(center_lon, center_lat):
    """
    Create a random rectangular building footprint around a center point.
    Returns a Shapely Polygon.
    """
    width = random.uniform(BUILDING_SIZE_MIN, BUILDING_SIZE_MAX)
    height = random.uniform(BUILDING_SIZE_MIN, BUILDING_SIZE_MAX)

    # Add slight rotation for realism
    angle = random.uniform(0, 15)  # degrees
    angle_rad = np.radians(angle)

    # Create basic rectangle
    half_w = width / 2
    half_h = height / 2

    # Corner points before rotation
    corners = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]

    # Apply rotation
    rotated = []
    for x, y in corners:
        rx = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        ry = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        rotated.append((center_lon + rx, center_lat + ry))

    return Polygon(rotated)


def check_overlap(new_poly, existing_polygons, min_gap):
    """Check if new polygon overlaps with or is too close to existing ones."""
    buffered = new_poly.buffer(min_gap)
    for existing in existing_polygons:
        if buffered.intersects(existing):
            return True
    return False


def is_inside_red_zone(polygon, red_zones):
    """Check if polygon is inside any red zone."""
    for rz in red_zones:
        if polygon.intersects(rz):
            return True
    return False


def is_inside_bounds(lon, lat):
    """Check if point is within operational bounds."""
    return (BOUNDS["min_lon"] <= lon <= BOUNDS["max_lon"] and
            BOUNDS["min_lat"] <= lat <= BOUNDS["max_lat"])


def generate_cluster_centers(num_clusters, red_zones):
    """Generate cluster center points, avoiding red zones."""
    centers = []
    attempts = 0
    max_attempts = num_clusters * 10

    while len(centers) < num_clusters and attempts < max_attempts:
        lon = random.uniform(BOUNDS["min_lon"] + 0.01, BOUNDS["max_lon"] - 0.01)
        lat = random.uniform(BOUNDS["min_lat"] + 0.01, BOUNDS["max_lat"] - 0.01)

        # Check if center is in a red zone
        from shapely.geometry import Point
        pt = Point(lon, lat)
        in_red = any(rz.contains(pt) for rz in red_zones)

        if not in_red:
            centers.append((lon, lat))

        attempts += 1

    print(f"  Generated {len(centers)} cluster centers (avoiding red zones)")
    return centers


def simulate_buildings(real_buildings_gdf, red_zones):
    """Generate simulated buildings in clusters."""
    print(f"\n  Generating {NUM_CLUSTERS} clusters x {BUILDINGS_PER_CLUSTER} buildings...")
    print(f"  Min gap between buildings: {MIN_GAP_METERS}m")
    print(f"  Height range: {MIN_HEIGHT}m - {MAX_HEIGHT}m")

    # Collect existing building polygons (from real buildings)
    existing_polys = []
    if len(real_buildings_gdf) > 0:
        for _, row in real_buildings_gdf.iterrows():
            if row.geometry and row.geometry.is_valid:
                existing_polys.append(row.geometry)
        print(f"  Existing real buildings to avoid: {len(existing_polys)}")

    # Generate cluster centers
    centers = generate_cluster_centers(NUM_CLUSTERS, red_zones)

    # Generate buildings
    buildings = []
    total_attempts = 0
    total_rejected = 0

    for ci, (cx, cy) in enumerate(centers):
        cluster_buildings = 0
        cluster_attempts = 0
        max_cluster_attempts = BUILDINGS_PER_CLUSTER * 5

        while cluster_buildings < BUILDINGS_PER_CLUSTER and cluster_attempts < max_cluster_attempts:
            # Random offset from cluster center (Gaussian distribution)
            offset_lon = random.gauss(0, CLUSTER_SPREAD / 2)
            offset_lat = random.gauss(0, CLUSTER_SPREAD / 2)

            blon = cx + offset_lon
            blat = cy + offset_lat

            # Check bounds
            if not is_inside_bounds(blon, blat):
                cluster_attempts += 1
                total_attempts += 1
                continue

            # Create building polygon
            poly = create_random_polygon(blon, blat)

            # Check red zone avoidance
            if is_inside_red_zone(poly, red_zones):
                cluster_attempts += 1
                total_attempts += 1
                total_rejected += 1
                continue

            # Check overlap with existing buildings
            if check_overlap(poly, existing_polys, MIN_GAP_DEG):
                cluster_attempts += 1
                total_attempts += 1
                total_rejected += 1
                continue

            # Building passed all checks - add it
            height = round(random.uniform(MIN_HEIGHT, MAX_HEIGHT), 1)
            buildings.append({
                "geometry": poly,
                "height": height,
                "building_name": f"SIM-C{ci+1:02d}-B{cluster_buildings+1:02d}",
                "height_method": "simulated",
                "type": "building",
                "source": "simulated",
                "cluster_id": ci + 1,
            })
            existing_polys.append(poly)
            cluster_buildings += 1
            cluster_attempts += 1
            total_attempts += 1

        if (ci + 1) % 10 == 0:
            print(f"    Cluster {ci+1}/{NUM_CLUSTERS}: {cluster_buildings} buildings placed")

    print(f"\n  Generation complete:")
    print(f"    Total buildings placed: {len(buildings)}")
    print(f"    Total attempts: {total_attempts}")
    print(f"    Rejected (overlap/red zone): {total_rejected}")

    return buildings


def save_simulated_buildings(buildings, real_gdf, output_path):
    """Save simulated + real buildings to GeoJSON."""
    # Create GeoDataFrame from simulated buildings
    sim_gdf = gpd.GeoDataFrame(buildings, geometry="geometry")
    sim_gdf.crs = "EPSG:4326"

    # Combine with real buildings
    if len(real_gdf) > 0:
        real_subset = real_gdf[["geometry", "height", "building_name",
                                "height_method", "type", "source"]].copy()
        real_subset["cluster_id"] = 0  # Real buildings have cluster_id 0
        combined = gpd.GeoDataFrame(
            __import__("pandas").concat([real_subset, sim_gdf], ignore_index=True),
            geometry="geometry"
        )
    else:
        combined = sim_gdf

    combined.crs = "EPSG:4326"

    # Save
    combined.to_file(output_path, driver="GeoJSON")
    file_size = os.path.getsize(output_path) / 1024
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Total buildings: {len(combined)}")
    print(f"    Real: {len(real_gdf)}")
    print(f"    Simulated: {len(buildings)}")

    return combined


def print_cluster_stats(buildings):
    """Print statistics about building clusters."""
    cluster_counts = {}
    cluster_heights = {}

    for b in buildings:
        cid = b["cluster_id"]
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        if cid not in cluster_heights:
            cluster_heights[cid] = []
        cluster_heights[cid].append(b["height"])

    print(f"\n  {'='*55}")
    print(f"  CLUSTER STATISTICS:")
    print(f"  {'='*55}")
    print(f"  {'Cluster':>8s} | {'Buildings':>9s} | {'Avg Height':>10s} | {'Min-Max'}")
    print(f"  {'─'*8}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*15}")

    for cid in sorted(cluster_counts.keys()):
        count = cluster_counts[cid]
        heights = cluster_heights[cid]
        avg_h = np.mean(heights)
        min_h = min(heights)
        max_h = max(heights)
        print(f"  {f'C{cid:02d}':>8s} | {count:>9d} | {avg_h:>8.1f}m | {min_h:.1f}m - {max_h:.1f}m")


def main():
    print("=" * 60)
    print("  STEP 3: Simulate 1000+ Clustered Buildings")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load real buildings
    print("\n  [1/4] Loading real buildings...")
    real_path = os.path.join(OUTPUT_DIR, "buildings_raw.geojson")
    real_gdf = load_real_buildings(real_path)

    # 2. Load red zones (for avoidance)
    print("\n  [2/4] Loading red zones...")
    map_path = os.path.join(DATA_DIR, "map.geojson")
    red_zones = load_red_zones(map_path)

    # 3. Generate simulated buildings
    print("\n  [3/4] Generating simulated buildings...")
    buildings = simulate_buildings(real_gdf, red_zones)

    # 4. Save combined output
    print("\n  [4/4] Saving combined buildings...")
    output_path = os.path.join(OUTPUT_DIR, "buildings_simulated.geojson")
    combined = save_simulated_buildings(buildings, real_gdf, output_path)

    # Print cluster stats
    print_cluster_stats(buildings)

    # Height distribution
    all_heights = combined["height"].values
    print(f"\n  {'='*55}")
    print(f"  HEIGHT DISTRIBUTION (Combined):")
    print(f"  {'='*55}")
    print(f"    Min:    {all_heights.min():.1f}m")
    print(f"    Max:    {all_heights.max():.1f}m")
    print(f"    Mean:   {all_heights.mean():.1f}m")
    print(f"    Median: {np.median(all_heights):.1f}m")

    # Summary
    print(f"\n  {'='*55}")
    print(f"  SUMMARY")
    print(f"  {'='*55}")
    print(f"    Real buildings:      {len(real_gdf)}")
    print(f"    Simulated buildings: {len(buildings)}")
    print(f"    Total combined:      {len(combined)}")
    print(f"    Clusters:            {NUM_CLUSTERS}")
    print(f"    Min gap:             {MIN_GAP_METERS}m")
    print(f"    Output: buildings_simulated.geojson")
    print(f"\n  [OK] Step 3 Complete! Ready for Step 4 (Merge Master Map).")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
