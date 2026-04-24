"""
============================================================
Step 6: Interactive Drone Route Planning Dashboard
============================================================
Streamlit dashboard for warehouse operators to plan safe
and optimized drone delivery routes in Jaipur.

Features:
  - Manual + click-to-select pickup/drop locations (2D map)
  - Yellow zone permission manager
  - A* pathfinding with zone compliance
  - 3D interactive visualization (PyDeck)
  - Distance, time, and compliance metrics

Run: streamlit run src/step6_dashboard.py
============================================================
"""

import os
import sys
import json
import time
import warnings
import io
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import shape, Polygon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from step5_pathfinder import compute_path, ALGORITHM_MAP

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MAP_FILE = os.path.join(DATA_DIR, "map.geojson")
MASTER_MAP_FILE = os.path.join(OUTPUT_DIR, "jaipur_master_map.geojson")
BUILDINGS_FILE = os.path.join(OUTPUT_DIR, "buildings_simulated.geojson")

# ──────────────────────────────────────────────
# MAP CENTER (Jaipur)
# ──────────────────────────────────────────────
JAIPUR_CENTER = [26.8514, 75.8064]
BOUNDS = {
    "min_lon": 75.7006, "max_lon": 75.9006,
    "min_lat": 26.7573, "max_lat": 26.9373,
}


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
# Only set page config when running standalone (not imported by app.py)
import inspect as _inspect
_caller = _inspect.stack()
_is_imported = any("app.py" in str(frame.filename) for frame in _caller)
if not _is_imported:
    st.set_page_config(
        page_title="Drone Route Planner - Jaipur",
        page_icon="🛸",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main app */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a3e 50%, #0a2a4a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 255, 136, 0.2);
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.1);
    }
    .main-header h1 {
        color: #00ff88 !important;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #8892b0;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(0, 255, 136, 0.15);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .metric-card .metric-value {
        color: #00ff88;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-card .metric-label {
        color: #8892b0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Status badges */
    .status-success {
        background: rgba(0, 255, 136, 0.15);
        color: #00ff88;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    .status-fail {
        background: rgba(255, 51, 51, 0.15);
        color: #ff3333;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(255, 51, 51, 0.3);
    }

    /* Zone table */
    .zone-table {
        font-size: 0.85rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #00ff88 !important;
    }

    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,255,136,0.3), transparent);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD ZONE DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_zone_data():
    """Load and classify zones from map.geojson."""
    with open(MAP_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    red_zones = []
    yellow_zones = []

    for feature in raw["features"]:
        props = feature.get("properties", {})
        zone_id = props.get("zone_id") or props.get("zone-id") or ""
        zone_type = (props.get("type") or props.get("type ") or "").strip().lower()
        name = props.get("name") or props.get("nam") or props.get("name ") or ""

        if zone_type == "red" or "Red" in str(zone_id):
            red_zones.append({
                "zone_id": zone_id,
                "name": name or "Unnamed",
                "geometry": feature["geometry"],
            })
        elif zone_type == "yellow" or "Yellow" in str(zone_id):
            yellow_zones.append({
                "zone_id": zone_id,
                "name": name or "Unnamed",
                "geometry": feature["geometry"],
            })

    return red_zones, yellow_zones


@st.cache_data
def load_buildings_data():
    """Load building data for 3D visualization."""
    if not os.path.exists(BUILDINGS_FILE):
        return []

    gdf = gpd.read_file(BUILDINGS_FILE)
    buildings = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom and geom.is_valid:
            # Get centroid for simplified rendering
            centroid = geom.centroid
            # Get bounding box for polygon rendering
            minx, miny, maxx, maxy = geom.bounds
            buildings.append({
                "lon": centroid.x,
                "lat": centroid.y,
                "height": float(row.get("height", 30)),
                "polygon": list(geom.exterior.coords) if geom.geom_type == "Polygon" else [],
                "name": str(row.get("building_name", "")),
            })

    return buildings


# ──────────────────────────────────────────────
# 2D FOLIUM MAP
# ──────────────────────────────────────────────
def create_2d_map(red_zones, yellow_zones, buildings=None, pickup=None, drop=None, path=None):
    """Create interactive 2D Folium map for point selection."""
    m = folium.Map(
        location=JAIPUR_CENTER,
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    # Add operational boundary
    folium.Rectangle(
        bounds=[[BOUNDS["min_lat"], BOUNDS["min_lon"]],
                [BOUNDS["max_lat"], BOUNDS["max_lon"]]],
        color="#00ff88",
        weight=2,
        fill=False,
        dash_array="10 5",
        popup="Operational Area (20x20 km)",
    ).add_to(m)

    # Draw Red zones
    for zone in red_zones:
        coords = zone["geometry"]["coordinates"][0]
        folium_coords = [[c[1], c[0]] for c in coords]  # flip to lat,lon
        folium.Polygon(
            locations=folium_coords,
            color="#ff3333",
            fill=True,
            fill_color="#ff3333",
            fill_opacity=0.4,
            weight=2,
            popup=f"RED: {zone['zone_id']} - {zone['name']}",
            tooltip=f"🔴 {zone['zone_id']}: {zone['name']}",
        ).add_to(m)

    # Draw Yellow zones
    for zone in yellow_zones:
        coords = zone["geometry"]["coordinates"][0]
        folium_coords = [[c[1], c[0]] for c in coords]
        folium.Polygon(
            locations=folium_coords,
            color="#ffcc00",
            fill=True,
            fill_color="#ffcc00",
            fill_opacity=0.3,
            weight=1,
            popup=f"YELLOW: {zone['zone_id']} - {zone['name']}",
            tooltip=f"🟡 {zone['zone_id']}: {zone['name']}",
        ).add_to(m)

    # Draw Buildings (only if enabled, limit for performance)
    if buildings:
        for b in buildings[:200]:  # limit to 200 for performance
            if b["polygon"] and len(b["polygon"]) > 2:
                folium_coords = [[c[1], c[0]] for c in b["polygon"]]
                height = b.get("height", 0)
                name = b.get("name", "Building")
                folium.Polygon(
                    locations=folium_coords,
                    color="#aabbcc",
                    fill=True,
                    fill_color="#667788",
                    fill_opacity=0.5,
                    weight=1,
                    tooltip=f"🏢 {name} ({height:.0f}m)",
                ).add_to(m)

    # Draw path if available
    if path and len(path) > 1:
        path_coords = [[lat, lon] for lon, lat in path]
        folium.PolyLine(
            locations=path_coords,
            color="#00ff88",
            weight=4,
            opacity=0.9,
            popup="Drone Flight Path",
        ).add_to(m)

    # Add pickup marker
    if pickup:
        folium.Marker(
            location=[pickup[0], pickup[1]],
            popup=f"PICKUP ({pickup[0]:.6f}, {pickup[1]:.6f})",
            tooltip="📦 Pickup Point",
            icon=folium.Icon(color="green", icon="arrow-up", prefix="fa"),
        ).add_to(m)

    # Add drop marker
    if drop:
        folium.Marker(
            location=[drop[0], drop[1]],
            popup=f"DROP ({drop[0]:.6f}, {drop[1]:.6f})",
            tooltip="📍 Drop Point",
            icon=folium.Icon(color="blue", icon="arrow-down", prefix="fa"),
        ).add_to(m)

    # Click instruction
    folium.LatLngPopup().add_to(m)

    return m


# ──────────────────────────────────────────────
# 3D PYDECK MAP
# ──────────────────────────────────────────────
def create_3d_map(red_zones, yellow_zones, buildings, path=None,
                  pickup=None, drop=None, permitted_ids=None, **kwargs):
    """Create 3D PyDeck visualization."""
    layers = []
    permitted_ids = set(permitted_ids or [])

    # -- Building Layer (3D extrusions) --
    if buildings:
        building_polys = []
        for b in buildings:
            if b["polygon"]:
                building_polys.append({
                    "polygon": b["polygon"],
                    "height": b["height"],
                    "name": b["name"],
                })

        if building_polys:
            layers.append(pdk.Layer(
                "PolygonLayer",
                data=building_polys,
                get_polygon="polygon",
                get_elevation="height",
                get_fill_color=[200, 200, 200, 160],
                get_line_color=[255, 255, 255, 80],
                line_width_min_pixels=1,
                extruded=True,
                wireframe=True,
                pickable=True,
                auto_highlight=True,
            ))

    # -- Red Zone Layer --
    red_polys = []
    for zone in red_zones:
        coords = zone["geometry"]["coordinates"][0]
        red_polys.append({
            "polygon": coords,
            "height": 100,
            "name": f"RED: {zone['zone_id']} - {zone['name']}",
        })
    if red_polys:
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=red_polys,
            get_polygon="polygon",
            get_elevation="height",
            get_fill_color=[255, 50, 50, 100],
            get_line_color=[255, 0, 0, 200],
            line_width_min_pixels=2,
            extruded=True,
            pickable=True,
        ))

    # -- Yellow Zone Layer --
    yellow_polys = []
    for zone in yellow_zones:
        coords = zone["geometry"]["coordinates"][0]
        is_permitted = zone["zone_id"] in permitted_ids
        yellow_polys.append({
            "polygon": coords,
            "height": 10 if is_permitted else 80,
            "name": f"{'✅' if is_permitted else '🔒'} {zone['zone_id']} - {zone['name']}",
            "permitted": is_permitted,
        })

    if yellow_polys:
        # Permitted yellow zones (green tint)
        permitted_polys = [p for p in yellow_polys if p["permitted"]]
        blocked_polys = [p for p in yellow_polys if not p["permitted"]]

        if permitted_polys:
            layers.append(pdk.Layer(
                "PolygonLayer",
                data=permitted_polys,
                get_polygon="polygon",
                get_elevation="height",
                get_fill_color=[0, 200, 100, 60],
                get_line_color=[0, 255, 136, 150],
                line_width_min_pixels=1,
                extruded=True,
                pickable=True,
            ))
        if blocked_polys:
            layers.append(pdk.Layer(
                "PolygonLayer",
                data=blocked_polys,
                get_polygon="polygon",
                get_elevation="height",
                get_fill_color=[255, 204, 0, 80],
                get_line_color=[255, 170, 0, 180],
                line_width_min_pixels=1,
                extruded=True,
                pickable=True,
            ))

    # -- Flight Path Layer --
    drone_alt = kwargs.get('drone_altitude', 60) if kwargs else 60
    if path and len(path) > 1:
        path_data = [{"path": [[lon, lat, drone_alt] for lon, lat in path]}]
        layers.append(pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color=[0, 255, 136, 240],
            width_min_pixels=4,
            get_width=5,
            pickable=True,
        ))

        # Waypoint markers
        waypoints = [{"position": [lon, lat, 70], "label": f"WP{i}"}
                     for i, (lon, lat) in enumerate(path)]
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=waypoints,
            get_position="position",
            get_radius=30,
            get_fill_color=[0, 255, 200, 200],
            pickable=True,
        ))

    # -- Start/End markers --
    markers = []
    if pickup:
        markers.append({
            "position": [pickup[1], pickup[0], 80],
            "color": [0, 255, 100, 255],
            "label": "PICKUP"
        })
    if drop:
        markers.append({
            "position": [drop[1], drop[0], 80],
            "color": [50, 150, 255, 255],
            "label": "DROP"
        })

    if markers:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=markers,
            get_position="position",
            get_radius=60,
            get_fill_color="color",
            pickable=True,
        ))

    # View state
    view = pdk.ViewState(
        longitude=JAIPUR_CENTER[1],
        latitude=JAIPUR_CENTER[0],
        zoom=12,
        pitch=45,
        bearing=0,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{name}\nHeight: {height}m"},
    )


# ──────────────────────────────────────────────
# STATIC PNG GENERATOR
# ──────────────────────────────────────────────

def generate_path_image(red_zones, yellow_zones, path_coords=None,
                        pickup=None, drop=None, permitted_ids=None,
                        algorithm="A*", metrics=None):
    """
    Generate a high-quality static PNG of the drone route map using matplotlib.
    Returns bytes of the PNG image.
    """
    permitted_ids = set(permitted_ids or [])

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # Operational area boundary
    from matplotlib.patches import FancyBboxPatch
    boundary = plt.Rectangle(
        (BOUNDS["min_lon"], BOUNDS["min_lat"]),
        BOUNDS["max_lon"] - BOUNDS["min_lon"],
        BOUNDS["max_lat"] - BOUNDS["min_lat"],
        linewidth=1.5, edgecolor="#00ff88", facecolor="none",
        linestyle="--", alpha=0.6
    )
    ax.add_patch(boundary)

    # Draw Red zones
    for zone in red_zones:
        try:
            coords = zone["geometry"]["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.fill(xs, ys, color="#ff3333", alpha=0.45, zorder=2)
            ax.plot(xs, ys, color="#ff3333", linewidth=1.2, alpha=0.85, zorder=2)
        except Exception:
            pass

    # Draw Yellow zones
    for zone in yellow_zones:
        try:
            coords = zone["geometry"]["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            is_permitted = zone["zone_id"] in permitted_ids
            color = "#00c864" if is_permitted else "#ffcc00"
            ax.fill(xs, ys, color=color, alpha=0.30, zorder=2)
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.7, zorder=2)
        except Exception:
            pass

    # Draw flight path
    if path_coords and len(path_coords) > 1:
        lons = [p[0] for p in path_coords]
        lats = [p[1] for p in path_coords]
        ax.plot(lons, lats, color="#00ff88", linewidth=3.0,
                zorder=5, label="Flight Path", solid_capstyle="round")
        # Waypoint dots
        ax.scatter(lons, lats, color="#00ffcc", s=18, zorder=6, alpha=0.85)

    # Pickup marker
    if pickup:
        ax.scatter(pickup[1], pickup[0], color="#00ff88", s=180,
                   marker="^", zorder=7, label="Pickup")
        ax.annotate("PICKUP", xy=(pickup[1], pickup[0]),
                    xytext=(6, 6), textcoords="offset points",
                    color="#00ff88", fontsize=9, fontweight="bold")

    # Drop marker
    if drop:
        ax.scatter(drop[1], drop[0], color="#3399ff", s=180,
                   marker="v", zorder=7, label="Drop")
        ax.annotate("DROP", xy=(drop[1], drop[0]),
                    xytext=(6, 6), textcoords="offset points",
                    color="#3399ff", fontsize=9, fontweight="bold")

    # Axes styling
    ax.set_xlim(BOUNDS["min_lon"] - 0.005, BOUNDS["max_lon"] + 0.005)
    ax.set_ylim(BOUNDS["min_lat"] - 0.004, BOUNDS["max_lat"] + 0.004)
    ax.set_xlabel("Longitude", color="#8892b0", fontsize=10)
    ax.set_ylabel("Latitude", color="#8892b0", fontsize=10)
    ax.tick_params(colors="#8892b0", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")

    # Grid
    ax.grid(True, color="#2a2a4a", linewidth=0.5, linestyle="--", alpha=0.5)

    # Title
    algo_label = algorithm or "A*"
    title = f"🛸 Drone Route Plan — Jaipur Airspace ({algo_label})"
    ax.set_title(title, color="#00ff88", fontsize=13, fontweight="bold", pad=14)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#ff3333", alpha=0.7, label="Red Zone (No-Fly)"),
        mpatches.Patch(color="#ffcc00", alpha=0.6, label="Yellow Zone (Blocked)"),
        mpatches.Patch(color="#00c864", alpha=0.6, label="Yellow Zone (Permitted)"),
        Line2D([0], [0], color="#00ff88", linewidth=2.5, label=f"Flight Path ({algo_label})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#00ff88",
               markersize=10, label="Pickup", linestyle="None"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#3399ff",
               markersize=10, label="Drop", linestyle="None"),
    ]
    legend = ax.legend(
        handles=legend_elements, loc="lower right",
        facecolor="#1a1a2e", edgecolor="#00ff88",
        labelcolor="#ccddff", fontsize=8.5, framealpha=0.9
    )

    # Metrics annotation (bottom-left)
    if metrics:
        info_lines = [
            f"Algorithm : {metrics.get('algorithm', algo_label)}",
            f"Path Dist : {metrics.get('path_distance_km', 0):.3f} km",
            f"Travel Time: {metrics.get('travel_time_min', 0):.1f} min",
            f"Waypoints : {metrics.get('waypoints', 0)}",
            f"Detour    : {metrics.get('detour_ratio', 0)}x",
        ]
        info_text = "\n".join(info_lines)
        ax.text(
            BOUNDS["min_lon"] + 0.002, BOUNDS["min_lat"] + 0.002,
            info_text,
            fontsize=8, color="#ccddff",
            bbox=dict(facecolor="#1a1a2e", edgecolor="#00ff88", alpha=0.85,
                      boxstyle="round,pad=0.5"),
            verticalalignment="bottom", zorder=8,
            fontfamily="monospace"
        )

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────
# MAIN DASHBOARD
# ──────────────────────────────────────────────
def main():
    # Load data
    red_zones, yellow_zones = load_zone_data()
    buildings = load_buildings_data()

    # Initialize session state
    if "pickup" not in st.session_state:
        st.session_state.pickup = None
    if "drop" not in st.session_state:
        st.session_state.drop = None
    if "path_result" not in st.session_state:
        st.session_state.path_result = None
    if "selecting" not in st.session_state:
        st.session_state.selecting = "pickup"

    # ═══════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════
    with st.sidebar:
        st.markdown("# 🛸 Drone Navigation")
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Input Method ──
        st.markdown("### 📍 Delivery Points")
        input_method = st.radio(
            "Select input method:",
            ["Manual Coordinates", "Click on Map"],
            horizontal=True,
        )

        if input_method == "Manual Coordinates":
            st.markdown("**Pickup Location:**")
            col1, col2 = st.columns(2)
            with col1:
                pickup_lat = st.number_input("Lat", value=26.920, format="%.6f",
                                             key="pickup_lat", min_value=26.7573, max_value=26.9373)
            with col2:
                pickup_lon = st.number_input("Lon", value=75.780, format="%.6f",
                                             key="pickup_lon", min_value=75.7006, max_value=75.9006)

            st.markdown("**Drop Location:**")
            col3, col4 = st.columns(2)
            with col3:
                drop_lat = st.number_input("Lat", value=26.850, format="%.6f",
                                           key="drop_lat", min_value=26.7573, max_value=26.9373)
            with col4:
                drop_lon = st.number_input("Lon", value=75.810, format="%.6f",
                                           key="drop_lon", min_value=75.7006, max_value=75.9006)

            st.session_state.pickup = (pickup_lat, pickup_lon)
            st.session_state.drop = (drop_lat, drop_lon)

        else:  # Click on Map
            st.info("👆 Click on the 2D map to set points")
            st.session_state.selecting = st.radio(
                "Currently selecting:",
                ["pickup", "drop"],
                format_func=lambda x: "📦 Pickup" if x == "pickup" else "📍 Drop",
                horizontal=True,
            )

            if st.session_state.pickup:
                p = st.session_state.pickup
                st.success(f"📦 Pickup: ({p[0]:.6f}, {p[1]:.6f})")
            if st.session_state.drop:
                d = st.session_state.drop
                st.success(f"📍 Drop: ({d[0]:.6f}, {d[1]:.6f})")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Drone Configuration ──
        st.markdown("### 🛸 Drone Configuration")

        drone_altitude = st.slider(
            "Flight Altitude (m)",
            min_value=20, max_value=120, value=60, step=5,
            help="Drone cruise altitude. Buildings taller than (altitude - safety margin) will be avoided."
        )

        drone_speed = st.slider(
            "Flight Speed (km/h)",
            min_value=10, max_value=100, value=50, step=5,
            help="Drone cruise speed for travel time estimation."
        )

        safety_margin = st.slider(
            "Vertical Safety Margin (m)",
            min_value=0, max_value=30, value=10, step=5,
            help="Clearance above buildings. Buildings >= (altitude - margin) are obstacles."
        )

        building_buffer = st.slider(
            "Horizontal Building Buffer (m)",
            min_value=0, max_value=20, value=3, step=1,
            help="Minimum gap around buildings. The drone won't pass between buildings closer than this."
        )

        min_obstacle = drone_altitude - safety_margin
        st.caption(f"Blocks buildings >= **{min_obstacle}m** | Keeps **{building_buffer}m** gap from walls")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Yellow Zone Permissions ──
        st.markdown("### 🟡 Yellow Zone Permissions")

        # Create options with names
        yellow_options = {f"{z['zone_id']}: {z['name']}": z['zone_id']
                          for z in yellow_zones}

        # Quick select buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select All", width="stretch"):
                st.session_state.yellow_selected = list(yellow_options.keys())
        with col_b:
            if st.button("Clear All", width="stretch"):
                st.session_state.yellow_selected = []

        selected_display = st.multiselect(
            "Permitted zones:",
            options=list(yellow_options.keys()),
            default=st.session_state.get("yellow_selected", []),
            key="yellow_multi",
            placeholder="Select permitted yellow zones...",
        )

        permitted_ids = [yellow_options[s] for s in selected_display]
        st.caption(f"✅ {len(permitted_ids)} / {len(yellow_zones)} zones permitted")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Performance ──
        st.markdown("### ⚡ Performance")
        show_buildings_2d = st.checkbox("Show Buildings on 2D Map", value=False,
            help="Disable for faster map loading (buildings always visible in 3D)")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Algorithm Selection ──
        st.markdown("### 🧠 Pathfinding Algorithm")

        ALGO_INFO = {
            "A*": "⭐ Best balance of speed & optimality (recommended)",
            "Best-First Search": "⚡ Fastest — may not find shortest path",
            "BFS": "🔵 Fewest hops (FIFO queue) — complete but slow on large grids",
            "Theta*": "🔷 Any-angle A* — shorter, smoother paths than A*",
            "RRT*": "🌟 Optimal RRT with rewiring — converges to shortest path",
        }

        selected_algo = st.selectbox(
            "Algorithm:",
            options=list(ALGORITHM_MAP.keys()),
            index=0,
            key="selected_algorithm",
            help="Select which pathfinding algorithm to use."
        )
        st.caption(ALGO_INFO.get(selected_algo, ""))

        st.markdown('\u003chr class="custom-divider"\u003e', unsafe_allow_html=True)

        # ── Compute Path Button ──
        st.markdown("### 🚀 Path Computation")

        can_compute = (st.session_state.pickup is not None and
                       st.session_state.drop is not None)

        if st.button("🚀 Compute Optimal Path", type="primary",
                     width="stretch", disabled=not can_compute):
            with st.spinner("Computing optimal path..."):
                pickup = st.session_state.pickup
                drop = st.session_state.drop

                result = compute_path(
                    start_lat=pickup[0],
                    start_lon=pickup[1],
                    end_lat=drop[0],
                    end_lon=drop[1],
                    permitted_yellow_zones=permitted_ids,
                    master_map_path=MASTER_MAP_FILE,
                    drone_altitude=drone_altitude,
                    drone_speed=drone_speed,
                    safety_margin=safety_margin,
                    building_buffer=building_buffer,
                    algorithm=selected_algo,
                )
                st.session_state.path_result = result

            if result["status"] == "SUCCESS":
                st.success("✅ Path computed successfully!")
            else:
                st.error(f"❌ {result.get('error', 'Failed to find path')}")

        if not can_compute:
            st.caption("⚠️ Set both pickup and drop points first")

        # ── Zone Summary ──
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown("### 📊 Zone Summary")
        st.markdown(f"🔴 **Red Zones:** {len(red_zones)} (always blocked)")
        st.markdown(f"🟡 **Yellow Zones:** {len(yellow_zones)} total")
        st.markdown(f"✅ **Permitted:** {len(permitted_ids)}")
        st.markdown(f"🔒 **Blocked:** {len(yellow_zones) - len(permitted_ids)}")

    # ═══════════════════════════════════════════
    # MAIN CONTENT AREA
    # ═══════════════════════════════════════════

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛸 Drone Route Planner — Jaipur Airspace</h1>
        <p>Interactive route planning with real-time zone compliance and 3D visualization</p>
    </div>
    """, unsafe_allow_html=True)

    # Get path data
    path_result = st.session_state.path_result
    path_coords = None
    if path_result and path_result["status"] == "SUCCESS":
        path_coords = path_result["path"]

    # ── Metrics Row ──
    if path_result and path_result["status"] == "SUCCESS":
        m = path_result["metrics"]
        comp = path_result.get("compliance", {})

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Path Distance</div>
                <div class="metric-value">{m['path_distance_km']:.2f} km</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Travel Time</div>
                <div class="metric-value">{m['travel_time_min']:.1f} min</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Waypoints</div>
                <div class="metric-value">{m['waypoints']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Detour Ratio</div>
                <div class="metric-value">{m['detour_ratio']}x</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            badge = "status-success" if comp.get("compliant") else "status-fail"
            label = "COMPLIANT" if comp.get("compliant") else "VIOLATION"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Zone Compliance</div>
                <div style="margin-top:0.5rem"><span class="{badge}">{label}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            algo_used = m.get('algorithm', 'A*')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Algorithm</div>
                <div class="metric-value" style="font-size:1rem;">{algo_used}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

    # ── Map Tabs ──
    tab_2d, tab_3d, tab_compliance = st.tabs([
        "🗺️ 2D Map (Select Points)",
        "🌐 3D Visualization",
        "📋 Compliance Report"
    ])

    # ── TAB 1: 2D Map ──
    with tab_2d:
        st.markdown("**Click on the map to select pickup/drop points** (when using 'Click on Map' mode)")

        folium_map = create_2d_map(
            red_zones, yellow_zones,
            buildings=buildings if show_buildings_2d else None,
            pickup=st.session_state.pickup,
            drop=st.session_state.drop,
            path=path_coords,
        )

        map_data = st_folium(
            folium_map, width=None, height=550, key="main_map",
            returned_objects=["last_clicked"],
        )

        # Handle map clicks
        if map_data and map_data.get("last_clicked") and input_method == "Click on Map":
            clicked = map_data["last_clicked"]
            lat = clicked["lat"]
            lng = clicked["lng"]

            # Check bounds
            if (BOUNDS["min_lat"] <= lat <= BOUNDS["max_lat"] and
                    BOUNDS["min_lon"] <= lng <= BOUNDS["max_lon"]):
                if st.session_state.selecting == "pickup":
                    st.session_state.pickup = (lat, lng)
                    st.toast(f"📦 Pickup set: ({lat:.6f}, {lng:.6f})")
                else:
                    st.session_state.drop = (lat, lng)
                    st.toast(f"📍 Drop set: ({lat:.6f}, {lng:.6f})")
            else:
                st.warning("⚠️ Clicked point is outside the operational area!")

        # ── Download Buttons (2D Tab) ──
        st.markdown('\n---')
        st.markdown("#### ⬇️ Download Map")
        dl_col1, dl_col2 = st.columns(2)

        with dl_col1:
            # HTML interactive map
            map_html_bytes = folium_map.get_root().render().encode("utf-8")
            st.download_button(
                label="🗺️ Download Interactive Map (HTML)",
                data=map_html_bytes,
                file_name="drone_route_map.html",
                mime="text/html",
                width="stretch",
                help="Opens in any browser as a fully interactive map",
            )

        with dl_col2:
            # Static PNG
            algo_name = path_result["metrics"].get("algorithm", "A*") if (path_result and path_result.get("status") == "SUCCESS") else "A*"
            png_bytes = generate_path_image(
                red_zones, yellow_zones,
                path_coords=path_coords,
                pickup=st.session_state.pickup,
                drop=st.session_state.drop,
                permitted_ids=permitted_ids,
                algorithm=algo_name,
                metrics=path_result["metrics"] if (path_result and path_result.get("status") == "SUCCESS") else None,
            )
            st.download_button(
                label="📸 Download Route Image (PNG)",
                data=png_bytes,
                file_name="drone_route_map.png",
                mime="image/png",
                width="stretch",
                help="High-quality static map image (150 DPI)",
            )

    # ── TAB 2: 3D Visualization ──
    with tab_3d:
        st.markdown("**Rotate, zoom, and tilt** to explore the 3D airspace")

        deck = create_3d_map(
            red_zones, yellow_zones, buildings,
            path=path_coords,
            pickup=st.session_state.pickup,
            drop=st.session_state.drop,
            permitted_ids=permitted_ids,
            drone_altitude=drone_altitude,
        )
        st.pydeck_chart(deck, height=600, width="stretch")

        # Legend
        st.markdown("""
        <div style="display:flex; gap:1.5rem; justify-content:center; margin-top:0.5rem; flex-wrap:wrap;">
            <span>⬜ Buildings</span>
            <span style="color:#ff3333;">🟥 Red Zones (No-Fly)</span>
            <span style="color:#ffcc00;">🟨 Yellow (Blocked)</span>
            <span style="color:#00c864;">🟩 Yellow (Permitted)</span>
            <span style="color:#00ff88;">━━ Flight Path</span>
            <span style="color:#00ff88;">● Waypoints</span>
        </div>
        """, unsafe_allow_html=True)

        # ── PNG Download (3D Tab) ──
        if path_coords:
            st.markdown("---")
            algo_name_3d = path_result["metrics"].get("algorithm", "A*") if (path_result and path_result.get("status") == "SUCCESS") else "A*"
            png_bytes_3d = generate_path_image(
                red_zones, yellow_zones,
                path_coords=path_coords,
                pickup=st.session_state.pickup,
                drop=st.session_state.drop,
                permitted_ids=permitted_ids,
                algorithm=algo_name_3d,
                metrics=path_result["metrics"] if (path_result and path_result.get("status") == "SUCCESS") else None,
            )
            st.download_button(
                label="📸 Download Route Image (PNG)",
                data=png_bytes_3d,
                file_name="drone_route_3d_view.png",
                mime="image/png",
                width="stretch",
                help="Static route map with all zones and path information",
                key="dl_png_3d",
            )

    # ── TAB 3: Compliance Report ──
    with tab_compliance:
        if path_result and path_result["status"] == "SUCCESS":
            comp = path_result.get("compliance", {})
            m = path_result["metrics"]

            # Status header
            if comp.get("compliant"):
                st.success("✅ Route is fully compliant with all airspace regulations!")
            else:
                st.error(f"❌ Route has {len(comp.get('violations', []))} violations!")
                for v in comp.get("violations", []):
                    st.warning(f"⚠️ Violation: {v['zone_id']} ({v['type'].upper()}) - {v['name']}")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 📏 Path Metrics")
                metrics_df = pd.DataFrame([
                    {"Metric": "Algorithm Used", "Value": m.get('algorithm', 'A*')},
                    {"Metric": "Direct Distance", "Value": f"{m['direct_distance_km']:.3f} km"},
                    {"Metric": "Path Distance", "Value": f"{m['path_distance_km']:.3f} km"},
                    {"Metric": "Detour Ratio", "Value": f"{m['detour_ratio']}x"},
                    {"Metric": "Travel Time", "Value": f"{m['travel_time_min']:.1f} min"},
                    {"Metric": "Waypoints", "Value": str(m['waypoints'])},
                    {"Metric": "Drone Speed", "Value": f"{m['drone_speed_kmh']} km/h"},
                    {"Metric": "Drone Altitude", "Value": f"{m.get('drone_altitude_m', 60)}m"},
                    {"Metric": "Safety Margin", "Value": f"{m.get('safety_margin_m', 10)}m"},
                    {"Metric": "Grid Resolution", "Value": f"{m['grid_resolution_m']:.0f}m"},
                    {"Metric": "Computation Time", "Value": f"{m['computation_time_s']:.3f}s"},
                ])
                st.dataframe(metrics_df, hide_index=True, width="stretch")

            with col_b:
                st.markdown("#### 🟡 Yellow Zones Used (With Permission)")
                used = comp.get("yellow_zones_used", [])
                if used:
                    used_df = pd.DataFrame(used)[["zone_id", "name"]]
                    used_df.columns = ["Zone ID", "Name"]
                    st.dataframe(used_df, hide_index=True, width="stretch")
                else:
                    st.info("No yellow zones were crossed by this route.")

            # Waypoint coordinates
            st.markdown("#### 📌 Waypoint Coordinates")
            if path_coords:
                wp_df = pd.DataFrame(
                    [(i, f"{lon:.8f}", f"{lat:.8f}") for i, (lon, lat) in enumerate(path_coords)],
                    columns=["Waypoint", "Longitude", "Latitude"]
                )
                st.dataframe(wp_df, hide_index=True, width="stretch")

            # ── Download Section ──
            st.markdown("---")
            st.markdown("#### ⬇️ Download Output Files")
            st.markdown("""
            <style>
            .dl-section {
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border: 1px solid rgba(0,255,136,0.2);
                border-radius: 10px;
                padding: 1rem 1.2rem;
                margin-top: 0.5rem;
            }
            </style>
            """, unsafe_allow_html=True)

            dl1, dl2, dl3 = st.columns(3)

            algo_dl = m.get('algorithm', 'A*')

            # PNG download
            with dl1:
                png_bytes_dl = generate_path_image(
                    red_zones, yellow_zones,
                    path_coords=path_coords,
                    pickup=st.session_state.pickup,
                    drop=st.session_state.drop,
                    permitted_ids=permitted_ids,
                    algorithm=algo_dl,
                    metrics=m,
                )
                st.download_button(
                    label="📸 Map Image (PNG)",
                    data=png_bytes_dl,
                    file_name="drone_route_map.png",
                    mime="image/png",
                    width="stretch",
                    key="dl_png_compliance",
                    help="High-quality static route map image",
                )

            # CSV download
            with dl2:
                if path_coords:
                    import csv
                    csv_buf = io.StringIO()
                    writer = csv.writer(csv_buf)
                    writer.writerow(["waypoint", "longitude", "latitude"])
                    for i, (lon, lat) in enumerate(path_coords):
                        writer.writerow([i, round(lon, 8), round(lat, 8)])
                    st.download_button(
                        label="📄 Waypoints (CSV)",
                        data=csv_buf.getvalue().encode("utf-8"),
                        file_name="mission_path.csv",
                        mime="text/csv",
                        width="stretch",
                        key="dl_csv",
                        help="Waypoint coordinates as CSV file",
                    )

            # GeoJSON download
            with dl3:
                if path_coords:
                    geojson_data = {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"type": "flight_path", "algorithm": algo_dl},
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": [[lon, lat] for lon, lat in path_coords]
                                }
                            },
                            {
                                "type": "Feature",
                                "properties": {"type": "start_point", "label": "PICKUP"},
                                "geometry": {"type": "Point",
                                             "coordinates": [path_coords[0][0], path_coords[0][1]]}
                            },
                            {
                                "type": "Feature",
                                "properties": {"type": "end_point", "label": "DROP"},
                                "geometry": {"type": "Point",
                                             "coordinates": [path_coords[-1][0], path_coords[-1][1]]}
                            }
                        ]
                    }
                    st.download_button(
                        label="🌍 Flight Path (GeoJSON)",
                        data=json.dumps(geojson_data, indent=2).encode("utf-8"),
                        file_name="mission_path.geojson",
                        mime="application/geo+json",
                        width="stretch",
                        key="dl_geojson",
                        help="GeoJSON flight path for GIS tools (QGIS, Google Maps etc.)",
                    )

        else:
            if path_result and path_result["status"] != "SUCCESS":
                st.error(f"❌ Path computation failed: {path_result.get('error', 'Unknown error')}")
            else:
                st.info("👈 Set pickup/drop points and click 'Compute Optimal Path' to see the compliance report.")


if __name__ == "__main__":
    main()
