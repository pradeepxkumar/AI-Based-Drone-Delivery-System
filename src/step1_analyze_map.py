"""
============================================================
Step 1: Analyze & Validate map.geojson
============================================================
This script reads the zone data from map.geojson and produces
a comprehensive analysis report including:
  - Zone counts (Red vs Yellow)
  - Data quality issues (missing names, inconsistent keys)
  - Zone area calculations
  - Bounding box of the operational area
  - Visual map of all zones
============================================================
"""

import json
import os
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import shape

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MAP_FILE = os.path.join(DATA_DIR, "map.geojson")


def load_geojson(filepath):
    """Load GeoJSON file and return raw dict + GeoDataFrame."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    gdf = gpd.read_file(filepath)
    return raw, gdf


def classify_zones(raw_data):
    """
    Classify each feature into Red, Yellow, Boundary, or Unknown.
    Handles inconsistent property keys (zone_id vs zone-id, type with trailing space).
    """
    zones = {
        "red": [],
        "yellow": [],
        "boundary": [],
        "empty": [],
        "unknown": []
    }
    issues = []

    for i, feature in enumerate(raw_data["features"]):
        props = feature.get("properties", {})

        # Get zone_id (handle both 'zone_id' and 'zone-id')
        zone_id = props.get("zone_id") or props.get("zone-id")

        # Get type (handle trailing space: 'type ' vs 'type')
        zone_type = props.get("type") or props.get("type ")
        if zone_type:
            zone_type = zone_type.strip().lower()

        name = props.get("name") or props.get("nam") or props.get("name ")

        # ── Check for data issues ──
        if props.get("zone-id") and not props.get("zone_id"):
            issues.append(f"  ⚠️  Feature {i}: Uses 'zone-id' instead of 'zone_id' → {zone_id}")
        if props.get("type "):
            issues.append(f"  ⚠️  Feature {i}: 'type ' has trailing space → {zone_id}")
        if props.get("nam"):
            issues.append(f"  ⚠️  Feature {i}: Uses 'nam' instead of 'name' → {zone_id}")
        if zone_id and not name:
            issues.append(f"  ⚠️  Feature {i}: Missing name for zone → {zone_id}")

        # ── Classify ──
        if not props or len(props) == 0:
            zones["empty"].append({"index": i, "zone_id": None, "name": None})
        elif props.get("name") == "Jaipur_Wide_Drone_World":
            zones["boundary"].append({"index": i, "zone_id": None, "name": "Operational Boundary"})
        elif zone_type == "red" or (zone_id and "Red" in str(zone_id)):
            zones["red"].append({"index": i, "zone_id": zone_id, "name": name or "Unnamed"})
        elif zone_type == "yellow" or (zone_id and "Yellow" in str(zone_id)):
            zones["yellow"].append({"index": i, "zone_id": zone_id, "name": name or "Unnamed"})
        elif zone_id:
            zones["unknown"].append({"index": i, "zone_id": zone_id, "name": name})
        else:
            zones["empty"].append({"index": i, "zone_id": None, "name": None})

    return zones, issues


def calculate_areas(raw_data):
    """Calculate area of each zone in square meters (approximate)."""
    zone_areas = []
    for feature in raw_data["features"]:
        props = feature.get("properties", {})
        zone_id = props.get("zone_id") or props.get("zone-id")
        if not zone_id:
            continue
        try:
            geom = shape(feature["geometry"])
            # Convert degrees to approximate meters (at Jaipur latitude ~26.85°N)
            # 1 degree lat ≈ 111,320 m, 1 degree lon ≈ 99,855 m at this latitude
            area_deg2 = geom.area
            area_m2 = area_deg2 * 111320 * 99855  # Rough approximation
            zone_areas.append({
                "zone_id": zone_id,
                "name": props.get("name") or props.get("nam") or "?",
                "area_m2": area_m2,
                "area_km2": area_m2 / 1e6
            })
        except Exception:
            pass
    return zone_areas


def get_bounding_box(raw_data):
    """Get the operational area bounding box."""
    for feature in raw_data["features"]:
        props = feature.get("properties", {})
        if props.get("name") == "Jaipur_Wide_Drone_World":
            coords = feature["geometry"]["coordinates"][0]
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            return {
                "min_lon": min(lons),
                "max_lon": max(lons),
                "min_lat": min(lats),
                "max_lat": max(lats),
                "width_km": (max(lons) - min(lons)) * 99.855,
                "height_km": (max(lats) - min(lats)) * 111.32
            }
    return None


def plot_zones(raw_data, zones, output_dir):
    """Create a visual map of all zones."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0d0d1a")

    # Draw boundary
    for feature in raw_data["features"]:
        props = feature.get("properties", {})
        if props.get("name") == "Jaipur_Wide_Drone_World":
            geom = shape(feature["geometry"])
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.1, color="#00ff88", linewidth=2, edgecolor="#00ff88")
            ax.plot(x, y, color="#00ff88", linewidth=2, linestyle="--", alpha=0.6)

    # Draw Red zones
    for z in zones["red"]:
        feature = raw_data["features"][z["index"]]
        geom = shape(feature["geometry"])
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=0.5, color="#ff3333", edgecolor="#ff0000", linewidth=1.5)
        cx, cy = geom.centroid.coords[0]
        ax.annotate(z["zone_id"], (cx, cy), fontsize=5, ha="center",
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ff0000", alpha=0.7))

    # Draw Yellow zones
    for z in zones["yellow"]:
        feature = raw_data["features"][z["index"]]
        geom = shape(feature["geometry"])
        try:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.4, color="#ffcc00", edgecolor="#ff9900", linewidth=1)
            cx, cy = geom.centroid.coords[0]
            ax.annotate(z["zone_id"], (cx, cy), fontsize=4, ha="center",
                        color="black", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#ffcc00", alpha=0.6))
        except Exception:
            pass

    # Legend
    red_patch = mpatches.Patch(color="#ff3333", alpha=0.6, label=f'🔴 Red Zones ({len(zones["red"])})')
    yellow_patch = mpatches.Patch(color="#ffcc00", alpha=0.6, label=f'🟡 Yellow Zones ({len(zones["yellow"])})')
    green_patch = mpatches.Patch(color="#00ff88", alpha=0.3, label="🟢 Operational Area")
    ax.legend(handles=[red_patch, yellow_patch, green_patch], loc="upper left",
              fontsize=10, facecolor="#1a1a2e", edgecolor="#00ff88",
              labelcolor="white")

    ax.set_xlabel("Longitude", color="white", fontsize=12)
    ax.set_ylabel("Latitude", color="white", fontsize=12)
    ax.set_title("🛸 Jaipur Drone Airspace Map — Zone Analysis", color="#00ff88",
                 fontsize=16, fontweight="bold", pad=20)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#00ff88")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "zone_analysis_map.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  📊 Map saved to: {save_path}")


def main():
    print("=" * 60)
    print("  🛸 DRONE AIRSPACE MAP ANALYZER — Step 1")
    print("=" * 60)

    # ── Load data ──
    if not os.path.exists(MAP_FILE):
        print(f"\n  ❌ ERROR: map.geojson not found at: {MAP_FILE}")
        sys.exit(1)

    print(f"\n  📂 Loading: {MAP_FILE}")
    raw_data, gdf = load_geojson(MAP_FILE)
    print(f"  ✅ Loaded {len(raw_data['features'])} features")

    # ── Bounding Box ──
    bbox = get_bounding_box(raw_data)
    if bbox:
        print(f"\n  📐 Operational Area:")
        print(f"     Longitude: {bbox['min_lon']:.4f} → {bbox['max_lon']:.4f}")
        print(f"     Latitude:  {bbox['min_lat']:.4f} → {bbox['max_lat']:.4f}")
        print(f"     Size:      {bbox['width_km']:.1f} km × {bbox['height_km']:.1f} km")

    # ── Classify zones ──
    zones, issues = classify_zones(raw_data)
    print(f"\n  📊 Zone Classification:")
    print(f"     🔴 Red Zones:    {len(zones['red'])}")
    print(f"     🟡 Yellow Zones: {len(zones['yellow'])}")
    print(f"     🟢 Boundary:     {len(zones['boundary'])}")
    print(f"     ❓ Empty/Unknown: {len(zones['empty']) + len(zones['unknown'])}")

    # ── List all zones ──
    print(f"\n  {'─' * 50}")
    print(f"  🔴 RED ZONES (No-Fly):")
    print(f"  {'─' * 50}")
    for z in zones["red"]:
        print(f"     {z['zone_id']:>10s} │ {z['name']}")

    print(f"\n  {'─' * 50}")
    print(f"  🟡 YELLOW ZONES (Permission Required):")
    print(f"  {'─' * 50}")
    for z in zones["yellow"]:
        print(f"     {z['zone_id']:>12s} │ {z['name']}")

    # ── Data Quality Issues ──
    if issues:
        print(f"\n  {'─' * 50}")
        print(f"  ⚠️  DATA QUALITY ISSUES ({len(issues)} found):")
        print(f"  {'─' * 50}")
        for issue in issues:
            print(issue)
    else:
        print(f"\n  ✅ No data quality issues found!")

    # ── Area calculations ──
    zone_areas = calculate_areas(raw_data)
    zone_areas.sort(key=lambda x: x["area_m2"], reverse=True)
    print(f"\n  {'─' * 50}")
    print(f"  📐 TOP 10 LARGEST ZONES BY AREA:")
    print(f"  {'─' * 50}")
    print(f"  {'Zone ID':>14s} │ {'Name':<30s} │ {'Area':>10s}")
    print(f"  {'─' * 14}─┼─{'─' * 30}─┼─{'─' * 10}")
    for z in zone_areas[:10]:
        if z["area_km2"] >= 0.01:
            area_str = f"{z['area_km2']:.3f} km²"
        else:
            area_str = f"{z['area_m2']:.0f} m²"
        print(f"  {z['zone_id']:>14s} │ {z['name']:<30s} │ {area_str:>10s}")

    # ── Plot zones ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  🎨 Generating zone visualization map...")
    plot_zones(raw_data, zones, OUTPUT_DIR)

    # ── Summary ──
    total_red_area = sum(z["area_km2"] for z in zone_areas if "Red" in z["zone_id"])
    total_yellow_area = sum(z["area_km2"] for z in zone_areas if "Yellow" in z["zone_id"])
    total_area = bbox["width_km"] * bbox["height_km"] if bbox else 0
    green_area = total_area - total_red_area - total_yellow_area

    print(f"\n  {'═' * 50}")
    print(f"  📋 SUMMARY")
    print(f"  {'═' * 50}")
    print(f"     Total operational area:  {total_area:.1f} km²")
    print(f"     Red zone coverage:       {total_red_area:.3f} km² ({total_red_area/total_area*100:.1f}%)")
    print(f"     Yellow zone coverage:    {total_yellow_area:.3f} km² ({total_yellow_area/total_area*100:.1f}%)")
    print(f"     Green (open) area:       {green_area:.1f} km² ({green_area/total_area*100:.1f}%)")
    print(f"\n  ✅ Step 1 Complete! Ready for Step 2 (Building Extraction).")
    print(f"  {'═' * 50}")


if __name__ == "__main__":
    main()
