"""
============================================================
Step 4: Merge Zones + Buildings into Master Map
============================================================
This script:
  1. Loads map.geojson (Red/Yellow zones)
  2. Loads buildings_simulated.geojson (real + simulated)
  3. Assigns heights to zones:
     - Working Area boundary: 0m (ground, free to fly)
     - Red Zones: 500m (impenetrable wall)
     - Yellow Zones: 500m by default (blocked if no permission)
       → Set to 0m when permission is granted (done at runtime)
  4. Standardizes all columns
  5. Saves to output/jaipur_master_map.geojson
============================================================
"""

import os
import sys
import json
import warnings
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

MAP_FILE = os.path.join(DATA_DIR, "map.geojson")
BUILDINGS_FILE = os.path.join(OUTPUT_DIR, "buildings_simulated.geojson")
MASTER_OUTPUT = os.path.join(OUTPUT_DIR, "jaipur_master_map.geojson")

# Zone height assignments
RED_ZONE_HEIGHT = 500    # meters — impenetrable wall
YELLOW_ZONE_HEIGHT = 500  # meters — blocked by default (no permission)
YELLOW_PERMITTED_HEIGHT = 0  # meters — open when permission granted
BOUNDARY_HEIGHT = 0      # meters — ground level, free to fly


def load_zones(filepath):
    """Load and process zone data from map.geojson."""
    print(f"  Loading zones from: {os.path.basename(filepath)}")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    zones = []
    skipped = 0

    for i, feature in enumerate(raw["features"]):
        props = feature.get("properties", {})

        # Get zone identifiers (handle inconsistent keys)
        zone_id = props.get("zone_id") or props.get("zone-id") or ""
        zone_type = (props.get("type") or props.get("type ") or "").strip().lower()
        name = props.get("name") or props.get("nam") or props.get("name ") or ""

        # Parse geometry
        try:
            geom = shape(feature["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)  # Fix invalid geometries
            if geom.is_empty or geom.area == 0:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        # Classify and assign height
        if name == "Jaipur_Wide_Drone_World" or props.get("description", "").startswith("20x20"):
            # Operational boundary
            zone_entry = {
                "geometry": geom,
                "zone_id": "BOUNDARY",
                "zone_type": "boundary",
                "name": "Operational Area",
                "height": BOUNDARY_HEIGHT,
                "source": "zone_map",
                "description": props.get("description", "20x20 km Working Area"),
            }
        elif zone_type == "red" or "Red" in str(zone_id):
            # Red zone — impenetrable
            zone_entry = {
                "geometry": geom,
                "zone_id": zone_id if zone_id else f"Red-{i}",
                "zone_type": "red",
                "name": name or "Unnamed Red Zone",
                "height": RED_ZONE_HEIGHT,
                "source": "zone_map",
                "description": f"No-Fly Zone: {name}",
            }
        elif zone_type == "yellow" or "Yellow" in str(zone_id):
            # Yellow zone — blocked by default
            zone_entry = {
                "geometry": geom,
                "zone_id": zone_id if zone_id else f"Yellow-{i}",
                "zone_type": "yellow",
                "name": name or "Unnamed Yellow Zone",
                "height": YELLOW_ZONE_HEIGHT,  # Blocked by default
                "source": "zone_map",
                "description": f"Restricted Zone: {name}",
            }
        elif not zone_id and not zone_type:
            # Empty/junk feature — skip
            skipped += 1
            continue
        else:
            # Unknown zone — treat as obstacle
            zone_entry = {
                "geometry": geom,
                "zone_id": zone_id or f"Unknown-{i}",
                "zone_type": "unknown",
                "name": name or "Unknown",
                "height": BOUNDARY_HEIGHT,
                "source": "zone_map",
                "description": "",
            }

        zones.append(zone_entry)

    print(f"    Processed: {len(zones)} zones")
    print(f"    Skipped (empty/invalid): {skipped}")

    # Count by type
    type_counts = {}
    for z in zones:
        t = z["zone_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, c in sorted(type_counts.items()):
        icon = {"red": "[RED]", "yellow": "[YLW]", "boundary": "[BND]"}.get(t, "[???]")
        print(f"    {icon} {t}: {c}")

    return zones


def load_buildings(filepath):
    """Load building data from buildings_simulated.geojson."""
    print(f"\n  Loading buildings from: {os.path.basename(filepath)}")

    if not os.path.exists(filepath):
        print(f"    [WARNING] File not found: {filepath}")
        return []

    gdf = gpd.read_file(filepath)
    print(f"    Loaded {len(gdf)} buildings")

    buildings = []
    for _, row in gdf.iterrows():
        buildings.append({
            "geometry": row.geometry,
            "zone_id": row.get("building_name", ""),
            "zone_type": "building",
            "name": row.get("building_name", ""),
            "height": float(row.get("height", 30)),
            "source": row.get("source", "building"),
            "description": f"Building: {row.get('building_name', '')} ({row.get('height', 0):.0f}m)",
        })

    # Stats
    heights = [b["height"] for b in buildings]
    real_count = sum(1 for b in buildings if b.get("source") == "osm")
    sim_count = len(buildings) - real_count
    print(f"    Real (OSM): {real_count}")
    print(f"    Simulated:  {sim_count}")
    print(f"    Height range: {min(heights):.1f}m - {max(heights):.1f}m")

    return buildings


def merge_and_save(zones, buildings, output_path):
    """Merge zones + buildings into a single master GeoJSON."""
    print(f"\n  Merging {len(zones)} zones + {len(buildings)} buildings...")

    # Combine all entries
    all_entries = zones + buildings

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(all_entries, geometry="geometry")
    gdf.crs = "EPSG:4326"

    # Ensure clean column order
    columns = ["geometry", "zone_id", "zone_type", "name", "height", "source", "description"]
    for col in columns:
        if col not in gdf.columns:
            gdf[col] = ""
    gdf = gdf[columns]

    # Save
    gdf.to_file(output_path, driver="GeoJSON")
    file_size = os.path.getsize(output_path) / 1024

    print(f"\n  Saved to: {output_path}")
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Total features: {len(gdf)}")

    return gdf


def print_master_summary(gdf):
    """Print a detailed summary of the master map."""
    print(f"\n  {'='*55}")
    print(f"  MASTER MAP SUMMARY")
    print(f"  {'='*55}")

    # Count by type
    type_counts = gdf["zone_type"].value_counts()
    print(f"\n  Feature Counts:")
    type_icons = {
        "red": "  [RED]",
        "yellow": "  [YLW]",
        "boundary": "  [BND]",
        "building": "  [BLD]",
        "unknown": "  [???]",
    }
    for t, c in type_counts.items():
        icon = type_icons.get(t, "  [---]")
        print(f"    {icon} {t:>10s}: {c:>5d}")

    # Height stats by type
    print(f"\n  Height Statistics by Type:")
    print(f"  {'Type':>10s} | {'Count':>6s} | {'Min':>7s} | {'Max':>7s} | {'Mean':>7s}")
    print(f"  {'─'*10}─┼─{'─'*6}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")

    for t in ["red", "yellow", "building", "boundary"]:
        subset = gdf[gdf["zone_type"] == t]
        if len(subset) > 0:
            h = subset["height"].astype(float)
            print(f"  {t:>10s} | {len(subset):>6d} | {h.min():>5.1f}m | {h.max():>5.1f}m | {h.mean():>5.1f}m")

    # Zone compliance info
    red_count = len(gdf[gdf["zone_type"] == "red"])
    yellow_count = len(gdf[gdf["zone_type"] == "yellow"])
    building_count = len(gdf[gdf["zone_type"] == "building"])

    print(f"\n  Drone Navigation Notes:")
    print(f"    - {red_count} Red zones at 500m height (ALWAYS blocked)")
    print(f"    - {yellow_count} Yellow zones at 500m (blocked by default)")
    print(f"      -> Set to 0m at runtime when permission is granted")
    print(f"    - {building_count} buildings at 30-60m height (physical obstacles)")
    print(f"    - Drone cruise altitude: 60m")


def main():
    print("=" * 60)
    print("  STEP 4: Merge into Master Map")
    print("=" * 60)

    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load zones
    print("\n  [1/3] Loading zone data...")
    if not os.path.exists(MAP_FILE):
        print(f"  [ERROR] map.geojson not found at: {MAP_FILE}")
        sys.exit(1)
    zones = load_zones(MAP_FILE)

    # 2. Load buildings
    print("\n  [2/3] Loading building data...")
    buildings = load_buildings(BUILDINGS_FILE)

    # 3. Merge and save
    print("\n  [3/3] Merging and saving master map...")
    gdf = merge_and_save(zones, buildings, MASTER_OUTPUT)

    # Summary
    print_master_summary(gdf)

    print(f"\n  {'='*55}")
    print(f"  [OK] Step 4 Complete!")
    print(f"    Master map: jaipur_master_map.geojson")
    print(f"    Ready for Step 5 (A* Pathfinder).")
    print(f"  {'='*55}")


if __name__ == "__main__":
    main()
