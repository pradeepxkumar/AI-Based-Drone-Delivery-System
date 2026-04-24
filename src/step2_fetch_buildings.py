"""
============================================================
Step 2: Fetch Real Buildings from OpenStreetMap (OSMnx)
============================================================
This script:
  1. Downloads building footprints within 10km of Jawahar Circle, Jaipur
  2. Assigns heights using smart modeling:
     - Manual overrides for famous buildings
     - Floor-to-meter conversion (H = floors × 3.5)
     - Stochastic estimation by building type
  3. Filters buildings >= 30m (drone cruise at 60m)
  4. Saves to output/buildings_raw.geojson
============================================================
"""

import os
import sys
import json
import random
import warnings
import numpy as np
import geopandas as gpd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Jawahar Circle, Jaipur — center point for building search
CENTER_POINT = (26.8514, 75.8064)  # (lat, lon)
SEARCH_RADIUS = 10000  # 10 km radius

# Drone config
MIN_HEIGHT_FILTER = 30  # meters — ignore buildings shorter than this
DRONE_CRUISE_ALT = 60   # meters

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────
# MANUAL HEIGHT OVERRIDES (Famous Jaipur Buildings)
# ──────────────────────────────────────────────
MANUAL_HEIGHTS = {
    "world trade park": 45,
    "wtp": 45,
    "gaurav tower": 40,
    "gt central": 40,
    "crystal palm": 35,
    "crystal court": 35,
    "triton mall": 38,
    "marriott": 50,
    "jw marriott": 50,
    "holiday inn": 42,
    "radisson": 45,
    "hilton": 48,
    "clarks amer": 35,
    "hotel clarks amer": 35,
    "rajmandir cinema": 30,
    "pink square mall": 32,
    "mgf metropolitan": 35,
    "elante mall": 36,
    "birla mandir": 35,
    "hawa mahal": 15,
    "city palace": 20,
    "nahargarh fort": 25,
    "amber fort": 30,
    "jaipur airport": 15,
    "sms hospital": 30,
    "fortis hospital": 35,
    "manipal hospital": 40,
    "mahatma gandhi hospital": 32,
}

# ──────────────────────────────────────────────
# HEIGHT ESTIMATION BY BUILDING TYPE
# ──────────────────────────────────────────────
HEIGHT_RANGES = {
    "commercial": (25, 50),
    "retail": (15, 35),
    "office": (20, 45),
    "hotel": (25, 55),
    "hospital": (20, 40),
    "industrial": (12, 30),
    "warehouse": (10, 25),
    "residential": (8, 25),
    "apartments": (15, 40),
    "house": (5, 12),
    "school": (10, 20),
    "university": (12, 25),
    "religious": (10, 35),
    "temple": (10, 30),
    "mosque": (8, 25),
    "church": (10, 30),
    "civic": (10, 25),
    "government": (12, 30),
    "public": (10, 25),
    "default": (8, 20),
}


def fetch_buildings():
    """Fetch building footprints from OpenStreetMap using OSMnx."""
    try:
        import osmnx as ox
    except ImportError:
        print("  [ERROR] osmnx not installed. Run: pip install osmnx")
        sys.exit(1)

    print(f"  Fetching buildings within {SEARCH_RADIUS/1000:.0f}km of Jawahar Circle...")
    print(f"  Center: ({CENTER_POINT[0]}, {CENTER_POINT[1]})")
    print(f"  This may take 1-2 minutes...\n")

    try:
        # Fetch building footprints
        tags = {"building": True}
        gdf = ox.features_from_point(CENTER_POINT, tags=tags, dist=SEARCH_RADIUS)

        # Keep only Polygon and MultiPolygon geometries
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

        print(f"  Downloaded {len(gdf)} building footprints from OSM")
        return gdf

    except Exception as e:
        print(f"  [ERROR] Failed to fetch buildings: {e}")
        print(f"  Make sure you have internet connection.")
        sys.exit(1)


def get_manual_height(name):
    """Check if building name matches any manual override."""
    if not name or not isinstance(name, str):
        return None
    name_lower = name.lower().strip()
    for key, height in MANUAL_HEIGHTS.items():
        if key in name_lower:
            return height
    return None


def estimate_height(row):
    """
    Estimate building height using a priority system:
    1. Manual override (famous buildings)
    2. Explicit height tag from OSM
    3. Floor-to-meter conversion (H = levels x 3.5)
    4. Stochastic estimation by building type
    """
    # 1. Check manual overrides
    name = row.get("name", "")
    manual_h = get_manual_height(name)
    if manual_h is not None:
        return manual_h, "manual"

    # 2. Check OSM height tag
    osm_height = row.get("height")
    if osm_height is not None:
        try:
            h = float(str(osm_height).replace("m", "").strip())
            if h > 0:
                return h, "osm_height"
        except (ValueError, TypeError):
            pass

    # 3. Floor-to-meter conversion
    levels = row.get("building:levels")
    if levels is not None:
        try:
            h = float(str(levels)) * 3.5
            if h > 0:
                return h, "floors"
        except (ValueError, TypeError):
            pass

    # 4. Stochastic estimation by building type
    building_type = str(row.get("building", "default")).lower()

    # Map specific types to our categories
    type_mapping = {
        "yes": "default",
        "true": "default",
        "commercial": "commercial",
        "retail": "retail",
        "office": "office",
        "hotel": "hotel",
        "hospital": "hospital",
        "industrial": "industrial",
        "warehouse": "warehouse",
        "residential": "residential",
        "apartments": "apartments",
        "house": "house",
        "detached": "house",
        "semidetached_house": "house",
        "terrace": "house",
        "school": "school",
        "university": "university",
        "college": "university",
        "religious": "religious",
        "temple": "temple",
        "mosque": "mosque",
        "church": "church",
        "civic": "civic",
        "government": "government",
        "public": "public",
    }

    category = type_mapping.get(building_type, "default")
    low, high = HEIGHT_RANGES.get(category, HEIGHT_RANGES["default"])
    h = random.uniform(low, high)
    return round(h, 1), "estimated"


def process_buildings(gdf):
    """Process buildings: assign heights and filter."""
    print(f"  Assigning heights to {len(gdf)} buildings...")

    heights = []
    methods = []
    names = []

    for idx, row in gdf.iterrows():
        h, method = estimate_height(row)
        heights.append(h)
        methods.append(method)
        names.append(row.get("name", ""))

    gdf = gdf.copy()
    gdf["height"] = heights
    gdf["height_method"] = methods
    gdf["building_name"] = names

    # Stats before filtering
    method_counts = {}
    for m in methods:
        method_counts[m] = method_counts.get(m, 0) + 1

    print(f"\n  Height Assignment Methods:")
    for m, c in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"    {m:>12s}: {c:>5d} buildings")

    print(f"\n  Height Distribution (before filter):")
    h_array = np.array(heights)
    print(f"    Min:    {h_array.min():.1f}m")
    print(f"    Max:    {h_array.max():.1f}m")
    print(f"    Mean:   {h_array.mean():.1f}m")
    print(f"    Median: {np.median(h_array):.1f}m")

    # Filter: keep only buildings >= MIN_HEIGHT_FILTER
    before_count = len(gdf)
    gdf = gdf[gdf["height"] >= MIN_HEIGHT_FILTER]
    after_count = len(gdf)

    print(f"\n  Filtering buildings >= {MIN_HEIGHT_FILTER}m (drone cruise: {DRONE_CRUISE_ALT}m):")
    print(f"    Before: {before_count}")
    print(f"    After:  {after_count}")
    print(f"    Removed: {before_count - after_count} short buildings")

    return gdf


def save_buildings(gdf, output_path):
    """Save processed buildings to GeoJSON with clean columns."""
    # Keep only essential columns
    keep_cols = ["geometry", "height", "height_method", "building_name"]
    available_cols = [c for c in keep_cols if c in gdf.columns]
    gdf_clean = gdf[available_cols].copy()

    # Add metadata
    gdf_clean["type"] = "building"
    gdf_clean["source"] = "osm"

    # Reset index for clean output
    gdf_clean = gdf_clean.reset_index(drop=True)

    # Save
    gdf_clean.to_file(output_path, driver="GeoJSON")
    file_size = os.path.getsize(output_path) / 1024
    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Buildings: {len(gdf_clean)}")

    return gdf_clean


def print_top_buildings(gdf, n=15):
    """Print the tallest buildings found."""
    gdf_sorted = gdf.sort_values("height", ascending=False).head(n)
    print(f"\n  {'='*55}")
    print(f"  TOP {n} TALLEST BUILDINGS:")
    print(f"  {'='*55}")
    print(f"  {'#':>3s} | {'Name':<30s} | {'Height':>7s} | {'Method'}")
    print(f"  {'─'*3}─┼─{'─'*30}─┼─{'─'*7}─┼─{'─'*12}")

    for i, (idx, row) in enumerate(gdf_sorted.iterrows(), 1):
        name = str(row.get("building_name", ""))[:30] or "Unnamed"
        height = f"{row['height']:.1f}m"
        method = row.get("height_method", "?")
        print(f"  {i:>3d} | {name:<30s} | {height:>7s} | {method}")


def main():
    print("=" * 60)
    print("  STEP 2: Fetch Real Buildings (OSMnx)")
    print("=" * 60)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Fetch buildings from OSM
    print("\n  [1/3] Downloading building footprints from OpenStreetMap...")
    gdf = fetch_buildings()

    # 2. Process: assign heights + filter
    print(f"\n  [2/3] Processing buildings...")
    gdf = process_buildings(gdf)

    # 3. Save to GeoJSON
    output_path = os.path.join(OUTPUT_DIR, "buildings_raw.geojson")
    print(f"\n  [3/3] Saving processed buildings...")
    gdf_clean = save_buildings(gdf, output_path)

    # Print top buildings
    print_top_buildings(gdf_clean)

    # Summary
    print(f"\n  {'='*55}")
    print(f"  SUMMARY")
    print(f"  {'='*55}")
    print(f"    Buildings fetched from OSM:   {len(gdf)}")
    print(f"    After height filter (>={MIN_HEIGHT_FILTER}m): {len(gdf_clean)}")
    print(f"    Output file: buildings_raw.geojson")
    print(f"\n  [OK] Step 2 Complete! Ready for Step 3 (Simulate Buildings).")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
