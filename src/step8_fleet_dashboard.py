"""
============================================================
Step 8: Multi-Drone Fleet Dashboard
============================================================
Streamlit dashboard for fleet management:
  - 1 Warehouse + multiple drop points
  - Multiple drones with capacity/battery constraints
  - OR-Tools optimization for route assignments
  - All paths avoid buildings and restricted zones (A*)
  - 2D + 3D visualization with per-drone coloring

Run: streamlit run src/step8_fleet_dashboard.py
============================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from step7_fleet_optimizer import solve_fleet

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MAP_FILE = os.path.join(DATA_DIR, "map.geojson")
BUILDINGS_FILE = os.path.join(OUTPUT_DIR, "buildings_simulated.geojson")

JAIPUR_CENTER = [26.8514, 75.8064]
BOUNDS = {
    "min_lon": 75.7006, "max_lon": 75.9006,
    "min_lat": 26.7573, "max_lat": 26.9373,
}

# Drone colors for visualization (up to 10 drones)
DRONE_COLORS = [
    "#00ff88",  # Green
    "#3399ff",  # Blue
    "#ff6633",  # Orange
    "#cc33ff",  # Purple
    "#ffcc00",  # Yellow
    "#ff3366",  # Pink
    "#33ffcc",  # Cyan
    "#ff9933",  # Amber
    "#66ff33",  # Lime
    "#ff33cc",  # Magenta
]

DRONE_COLORS_RGB = [
    [0, 255, 136],
    [51, 153, 255],
    [255, 102, 51],
    [204, 51, 255],
    [255, 204, 0],
    [255, 51, 102],
    [51, 255, 204],
    [255, 153, 51],
    [102, 255, 51],
    [255, 51, 204],
]


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
# Only set page config when running standalone
import inspect as _inspect
_caller = _inspect.stack()
_is_imported = any("app.py" in str(frame.filename) for frame in _caller)
if not _is_imported:
    st.set_page_config(
        page_title="Drone Fleet Planner - Jaipur",
        page_icon="🚁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# Custom CSS (same theme as step6)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a3e 50%, #0a2a4a 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 255, 136, 0.2);
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.1);
    }
    .main-header h1 { color: #00ff88; margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p { color: #8892b0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(0, 255, 136, 0.15); border-radius: 10px;
        padding: 1rem 1.2rem; text-align: center;
    }
    .metric-card .metric-value { color: #00ff88; font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-card .metric-label { color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .custom-divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,255,136,0.3), transparent); margin: 1rem 0; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%); }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #00ff88 !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_zone_data():
    with open(MAP_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    red_zones, yellow_zones = [], []
    for feature in raw["features"]:
        props = feature.get("properties", {})
        zone_id = props.get("zone_id") or props.get("zone-id") or ""
        zone_type = (props.get("type") or props.get("type ") or "").strip().lower()
        name = props.get("name") or props.get("nam") or ""
        if zone_type == "red" or "Red" in str(zone_id):
            red_zones.append({"zone_id": zone_id, "name": name or "Unnamed", "geometry": feature["geometry"]})
        elif zone_type == "yellow" or "Yellow" in str(zone_id):
            yellow_zones.append({"zone_id": zone_id, "name": name or "Unnamed", "geometry": feature["geometry"]})
    return red_zones, yellow_zones


@st.cache_data
def load_buildings():
    import geopandas as gpd
    if not os.path.exists(BUILDINGS_FILE):
        return []
    gdf = gpd.read_file(BUILDINGS_FILE)
    buildings = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom and geom.is_valid and geom.geom_type == "Polygon":
            buildings.append({
                "polygon": list(geom.exterior.coords),
                "height": float(row.get("height", 30)),
                "name": str(row.get("building_name", "")),
            })
    return buildings


# ──────────────────────────────────────────────
# 2D MAP
# ──────────────────────────────────────────────
def create_fleet_2d_map(red_zones, yellow_zones, buildings,
                        warehouse=None, drops=None, fleet_result=None):
    m = folium.Map(location=JAIPUR_CENTER, zoom_start=12, tiles="CartoDB dark_matter")

    # Boundary
    folium.Rectangle(
        bounds=[[BOUNDS["min_lat"], BOUNDS["min_lon"]], [BOUNDS["max_lat"], BOUNDS["max_lon"]]],
        color="#00ff88", weight=2, fill=False, dash_array="10 5",
    ).add_to(m)

    # Red zones
    for zone in red_zones:
        coords = [[c[1], c[0]] for c in zone["geometry"]["coordinates"][0]]
        folium.Polygon(locations=coords, color="#ff3333", fill=True,
                      fill_color="#ff3333", fill_opacity=0.4, weight=2,
                      tooltip=f"🔴 {zone['zone_id']}: {zone['name']}").add_to(m)

    # Yellow zones
    for zone in yellow_zones:
        coords = [[c[1], c[0]] for c in zone["geometry"]["coordinates"][0]]
        folium.Polygon(locations=coords, color="#ffcc00", fill=True,
                      fill_color="#ffcc00", fill_opacity=0.3, weight=1,
                      tooltip=f"🟡 {zone['zone_id']}: {zone['name']}").add_to(m)

    # Buildings (limited for performance)
    if buildings:
        for b in buildings[:200]:
            if b["polygon"] and len(b["polygon"]) > 2:
                coords = [[c[1], c[0]] for c in b["polygon"]]
                folium.Polygon(locations=coords, color="#aabbcc", fill=True,
                              fill_color="#667788", fill_opacity=0.5, weight=1,
                              tooltip=f"🏢 {b['name']} ({b['height']:.0f}m)").add_to(m)

    # Fleet paths (different color per drone)
    if fleet_result and fleet_result.get("status") == "SUCCESS":
        for a in fleet_result["assignments"]:
            drone_idx = a["drone_id"] - 1
            color = DRONE_COLORS[drone_idx % len(DRONE_COLORS)]
            path = a.get("full_path", [])
            if path and len(path) > 1:
                path_coords = [[lat, lon] for lon, lat in path]
                folium.PolyLine(
                    locations=path_coords, color=color, weight=4, opacity=0.9,
                    tooltip=f"🚁 Drone {a['drone_id']}: {a['distance_km']:.2f}km",
                ).add_to(m)

    # Warehouse marker
    if warehouse:
        folium.Marker(
            location=[warehouse[0], warehouse[1]],
            tooltip=f"🏭 Warehouse ({warehouse[0]:.6f}, {warehouse[1]:.6f})",
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(m)

    # Drop markers
    if drops:
        for i, d in enumerate(drops):
            folium.Marker(
                location=[d["lat"], d["lon"]],
                tooltip=f"📦 Drop {i+1}: {d['name']} ({d['weight_kg']}kg)",
                icon=folium.Icon(color="blue", icon="box", prefix="fa"),
            ).add_to(m)

    folium.LatLngPopup().add_to(m)
    return m


# ──────────────────────────────────────────────
# 3D MAP
# ──────────────────────────────────────────────
def create_fleet_3d_map(red_zones, yellow_zones, buildings,
                        warehouse=None, drops=None, fleet_result=None,
                        drone_altitude=60):
    layers = []

    # Buildings
    if buildings:
        building_polys = [{"polygon": b["polygon"], "height": b["height"], "name": b["name"]}
                         for b in buildings if b["polygon"]]
        if building_polys:
            layers.append(pdk.Layer("PolygonLayer", data=building_polys,
                get_polygon="polygon", get_elevation="height",
                get_fill_color=[200, 200, 200, 160], get_line_color=[255, 255, 255, 80],
                extruded=True, wireframe=True, pickable=True))

    # Red zones
    red_polys = [{"polygon": z["geometry"]["coordinates"][0], "height": 100,
                  "name": f"RED: {z['zone_id']}"} for z in red_zones]
    if red_polys:
        layers.append(pdk.Layer("PolygonLayer", data=red_polys,
            get_polygon="polygon", get_elevation="height",
            get_fill_color=[255, 50, 50, 100], get_line_color=[255, 0, 0, 200],
            extruded=True, pickable=True))

    # Yellow zones
    yellow_polys = [{"polygon": z["geometry"]["coordinates"][0], "height": 80,
                     "name": f"YELLOW: {z['zone_id']}"} for z in yellow_zones]
    if yellow_polys:
        layers.append(pdk.Layer("PolygonLayer", data=yellow_polys,
            get_polygon="polygon", get_elevation="height",
            get_fill_color=[255, 204, 0, 80], get_line_color=[255, 170, 0, 180],
            extruded=True, pickable=True))

    # Fleet paths (different color per drone)
    if fleet_result and fleet_result.get("status") == "SUCCESS":
        for a in fleet_result["assignments"]:
            drone_idx = a["drone_id"] - 1
            color = DRONE_COLORS_RGB[drone_idx % len(DRONE_COLORS_RGB)] + [240]
            path = a.get("full_path", [])
            if path and len(path) > 1:
                path_data = [{"path": [[lon, lat, drone_altitude] for lon, lat in path]}]
                layers.append(pdk.Layer("PathLayer", data=path_data,
                    get_path="path", get_color=color, width_min_pixels=4, pickable=True))

    # Warehouse marker
    if warehouse:
        layers.append(pdk.Layer("ScatterplotLayer",
            data=[{"position": [warehouse[1], warehouse[0], 80], "label": "Warehouse"}],
            get_position="position", get_radius=80,
            get_fill_color=[0, 255, 100, 255], pickable=True))

    # Drop markers
    if drops:
        drop_data = [{"position": [d["lon"], d["lat"], 70],
                      "label": f"Drop {i+1}: {d['name']}"} for i, d in enumerate(drops)]
        layers.append(pdk.Layer("ScatterplotLayer", data=drop_data,
            get_position="position", get_radius=50,
            get_fill_color=[50, 150, 255, 255], pickable=True))

    view = pdk.ViewState(longitude=JAIPUR_CENTER[1], latitude=JAIPUR_CENTER[0],
                         zoom=12, pitch=45, bearing=0)
    return pdk.Deck(layers=layers, initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{name}\nHeight: {height}m"})


# ──────────────────────────────────────────────
# MAIN DASHBOARD
# ──────────────────────────────────────────────
def main():
    red_zones, yellow_zones = load_zone_data()
    buildings = load_buildings()

    # Initialize session state
    if "warehouse" not in st.session_state:
        st.session_state.warehouse = None
    if "drops" not in st.session_state:
        st.session_state.drops = []
    if "fleet_result" not in st.session_state:
        st.session_state.fleet_result = None
    if "selecting_mode" not in st.session_state:
        st.session_state.selecting_mode = "warehouse"

    # ═══════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════
    with st.sidebar:
        st.markdown("# 🚁 Fleet Planner")
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Warehouse ──
        st.markdown("### 🏭 Warehouse Location")
        wh_method = st.radio("Input:", ["Manual", "Click Map"], horizontal=True, key="wh_method")

        if wh_method == "Manual":
            c1, c2 = st.columns(2)
            with c1:
                wh_lat = st.number_input("Lat", value=26.920, format="%.6f", key="wh_lat",
                                         min_value=26.7573, max_value=26.9373)
            with c2:
                wh_lon = st.number_input("Lon", value=75.780, format="%.6f", key="wh_lon",
                                         min_value=75.7006, max_value=75.9006)
            st.session_state.warehouse = (wh_lat, wh_lon)
        else:
            if st.session_state.warehouse:
                w = st.session_state.warehouse
                st.success(f"🏭 ({w[0]:.6f}, {w[1]:.6f})")
            else:
                st.info("👆 Click map to set warehouse")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Drop Points ──
        st.markdown("### 📦 Drop Points")

        # Add drop form
        with st.expander("➕ Add a Drop Point", expanded=len(st.session_state.drops) == 0):
            drop_method = st.radio("Input:", ["Manual", "Click Map"],
                                   horizontal=True, key="drop_method")

            if drop_method == "Manual":
                dc1, dc2 = st.columns(2)
                with dc1:
                    drop_lat = st.number_input("Lat", value=26.850, format="%.6f", key="new_drop_lat",
                                               min_value=26.7573, max_value=26.9373)
                with dc2:
                    drop_lon = st.number_input("Lon", value=75.810, format="%.6f", key="new_drop_lon",
                                               min_value=75.7006, max_value=75.9006)
            else:
                drop_lat = st.session_state.get("clicked_drop_lat")
                drop_lon = st.session_state.get("clicked_drop_lon")
                if drop_lat:
                    st.success(f"📍 ({drop_lat:.6f}, {drop_lon:.6f})")
                else:
                    st.info("👆 Click map, then fill details below")
                    st.session_state.selecting_mode = "drop"

            drop_name = st.text_input("Name", value=f"Drop {len(st.session_state.drops)+1}",
                                      key="new_drop_name")
            drop_weight = st.number_input("Weight (kg)", value=0.5, min_value=0.1,
                                          max_value=10.0, step=0.1, key="new_drop_weight")

            if st.button("✅ Add Drop Point", width="stretch"):
                lat = drop_lat if drop_method == "Manual" else st.session_state.get("clicked_drop_lat")
                lon = drop_lon if drop_method == "Manual" else st.session_state.get("clicked_drop_lon")
                if lat and lon:
                    st.session_state.drops.append({
                        "lat": lat, "lon": lon,
                        "weight_kg": drop_weight, "name": drop_name
                    })
                    st.session_state.clicked_drop_lat = None
                    st.session_state.clicked_drop_lon = None
                    st.rerun()
                else:
                    st.error("Set location first!")

        # List current drops
        if st.session_state.drops:
            for i, d in enumerate(st.session_state.drops):
                col_info, col_del = st.columns([4, 1])
                with col_info:
                    st.caption(f"📦 **D{i+1}**: {d['name']} ({d['weight_kg']}kg)")
                with col_del:
                    if st.button("🗑️", key=f"del_drop_{i}"):
                        st.session_state.drops.pop(i)
                        st.rerun()

            total_wt = sum(d["weight_kg"] for d in st.session_state.drops)
            st.caption(f"Total: {len(st.session_state.drops)} drops, {total_wt:.1f} kg")

            if st.button("🗑️ Clear All Drops", width="stretch"):
                st.session_state.drops = []
                st.rerun()
        else:
            st.info("No drops added yet")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Drone Configuration ──
        st.markdown("### 🛸 Drone Configuration")

        num_drones = st.slider("Number of Drones", 1, 10, 3)
        drone_capacity = st.number_input("Capacity per Drone (kg)", value=2.5,
                                         min_value=0.5, max_value=20.0, step=0.5)
        drone_altitude = st.slider("Flight Altitude (m)", 20, 120, 60, step=5)
        drone_speed = st.slider("Flight Speed (km/h)", 10, 100, 50, step=5)
        safety_margin = st.slider("Vertical Safety (m)", 0, 30, 10, step=5)
        building_buffer = st.slider("Building Buffer (m)", 0, 20, 3, step=1)

        battery_rate = 3.0  # fixed: 3 km per 1%
        max_range = 100 * battery_rate
        st.caption(f"🔋 Battery: 100% | {battery_rate} km/% | **{max_range:.0f} km** range")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Yellow Zone Permissions ──
        st.markdown("### 🟡 Yellow Zones")
        yellow_options = {f"{z['zone_id']}: {z['name']}": z['zone_id'] for z in yellow_zones}

        ca, cb = st.columns(2)
        with ca:
            if st.button("All", width="stretch", key="sel_all"):
                st.session_state.fleet_yellow = list(yellow_options.keys())
        with cb:
            if st.button("None", width="stretch", key="sel_none"):
                st.session_state.fleet_yellow = []

        selected_yellow = st.multiselect("Permitted:", list(yellow_options.keys()),
            default=st.session_state.get("fleet_yellow", []),
            key="fleet_yellow_multi", placeholder="Select zones...")
        permitted_ids = [yellow_options[s] for s in selected_yellow]

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Performance ──
        st.markdown("### ⚡ Performance")
        show_buildings_2d = st.checkbox("Show Buildings on 2D Map", value=False,
            help="Disable for faster map loading (buildings always visible in 3D)",
            key="fleet_show_bldg")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # ── Compute Button ──
        can_compute = (st.session_state.warehouse is not None and
                       len(st.session_state.drops) > 0)

        if st.button("🚀 Optimize Fleet Routes", type="primary",
                     width="stretch", disabled=not can_compute):
            with st.spinner("Computing fleet routes... (this may take a minute)"):
                result = solve_fleet(
                    warehouse=st.session_state.warehouse,
                    drops=st.session_state.drops,
                    num_drones=num_drones,
                    drone_capacity_kg=drone_capacity,
                    battery_rate=battery_rate,
                    drone_altitude=drone_altitude,
                    drone_speed=drone_speed,
                    safety_margin=safety_margin,
                    building_buffer=building_buffer,
                    permitted_yellow=permitted_ids,
                )
                st.session_state.fleet_result = result

            if result["status"] == "SUCCESS":
                st.success(f"✅ Fleet optimized! {result['summary']['drones_used']} drones used")
            else:
                st.error(f"❌ {result.get('error', 'Optimization failed')}")

        if not can_compute:
            st.caption("⚠️ Set warehouse + at least 1 drop point")

    # ═══════════════════════════════════════════
    # MAIN CONTENT
    # ═══════════════════════════════════════════

    st.markdown("""
    <div class="main-header">
        <h1>🚁 Multi-Drone Fleet Planner — Jaipur Airspace</h1>
        <p>Fleet optimization with safe routes, capacity constraints, and battery management</p>
    </div>
    """, unsafe_allow_html=True)

    fleet_result = st.session_state.fleet_result

    # ── Metrics Row ──
    if fleet_result and fleet_result.get("status") == "SUCCESS":
        s = fleet_result["summary"]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Drones Used</div>
                <div class="metric-value">{s['drones_used']}/{s['total_drones_available']}</div></div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Drops Served</div>
                <div class="metric-value">{s['drops_served']}/{s['total_drops']}</div></div>""",
                unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Distance</div>
                <div class="metric-value">{s['total_distance_km']:.2f} km</div></div>""",
                unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Total Weight</div>
                <div class="metric-value">{s['total_weight_kg']:.1f} kg</div></div>""",
                unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Compute Time</div>
                <div class="metric-value">{s['computation_time_s']:.1f}s</div></div>""",
                unsafe_allow_html=True)
        st.markdown("")

    # ── Tabs ──
    tab_2d, tab_3d, tab_assignments = st.tabs([
        "🗺️ 2D Map", "🌐 3D Visualization", "📋 Fleet Assignments"
    ])

    # TAB 1: 2D Map
    with tab_2d:
        st.markdown("**Click map to set warehouse/drop points** (when using 'Click Map' mode)")

        folium_map = create_fleet_2d_map(
            red_zones, yellow_zones, buildings if show_buildings_2d else None,
            warehouse=st.session_state.warehouse,
            drops=st.session_state.drops,
            fleet_result=fleet_result,
        )
        map_data = st_folium(folium_map, width=None, height=550, key="fleet_map",
                             returned_objects=["last_clicked"])

        # Handle clicks
        if map_data and map_data.get("last_clicked"):
            clicked = map_data["last_clicked"]
            lat, lng = clicked["lat"], clicked["lng"]
            if BOUNDS["min_lat"] <= lat <= BOUNDS["max_lat"] and BOUNDS["min_lon"] <= lng <= BOUNDS["max_lon"]:
                if wh_method == "Click Map" and st.session_state.selecting_mode == "warehouse":
                    st.session_state.warehouse = (lat, lng)
                    st.toast(f"🏭 Warehouse set: ({lat:.6f}, {lng:.6f})")
                elif drop_method == "Click Map":
                    st.session_state.clicked_drop_lat = lat
                    st.session_state.clicked_drop_lon = lng
                    st.toast(f"📦 Drop location captured: ({lat:.6f}, {lng:.6f})")

    # TAB 2: 3D
    with tab_3d:
        st.markdown("**Rotate, zoom, and tilt** to explore fleet routes in 3D")
        deck = create_fleet_3d_map(
            red_zones, yellow_zones, buildings,
            warehouse=st.session_state.warehouse,
            drops=st.session_state.drops,
            fleet_result=fleet_result,
            drone_altitude=drone_altitude,
        )
        st.pydeck_chart(deck, height=600, width="stretch")

        # Legend
        if fleet_result and fleet_result.get("status") == "SUCCESS":
            legend_items = []
            for a in fleet_result["assignments"]:
                idx = a["drone_id"] - 1
                c = DRONE_COLORS[idx % len(DRONE_COLORS)]
                legend_items.append(f'<span style="color:{c};">━━ Drone {a["drone_id"]}</span>')
            legend_html = " &nbsp;|&nbsp; ".join(legend_items)
            st.markdown(f'<div style="text-align:center;margin-top:0.5rem;">{legend_html} &nbsp;|&nbsp; '
                        f'🟢 Warehouse &nbsp;|&nbsp; 🔵 Drops</div>', unsafe_allow_html=True)

    # TAB 3: Assignments
    with tab_assignments:
        if fleet_result and fleet_result.get("status") == "SUCCESS":
            st.markdown("### 🚁 Drone Assignments")

            # Per-drone table
            rows = []
            for a in fleet_result["assignments"]:
                drop_names = [st.session_state.drops[n-1]["name"] for n in a["drop_nodes"]]
                rows.append({
                    "Drone": f"Drone {a['drone_id']}",
                    "Drops": " → ".join(drop_names),
                    "# Drops": a["num_drops"],
                    "Distance": f"{a['distance_km']:.2f} km",
                    "Weight": f"{a['weight_kg']:.2f} kg",
                    "Battery Used": f"{a['battery_used_pct']:.1f}%",
                    "Battery Left": f"{a['battery_remaining_pct']:.1f}%",
                    "Time": f"{a['travel_time_min']:.1f} min",
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, width="stretch")

            # Unserved drops warning
            s = fleet_result["summary"]
            if s["drops_unserved"] > 0:
                st.warning(f"⚠️ {s['drops_unserved']} drops could not be served! "
                          f"Add more drones or increase capacity.")

            # Cost matrix
            st.markdown("### 📊 Cost Matrix (A* Safe Distances in km)")
            cm = fleet_result.get("cost_matrix", [])
            if cm:
                n = len(cm)
                labels = ["Warehouse"] + [f"Drop {i}" for i in range(1, n)]
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                st.dataframe(cm_df.style.format("{:.2f}"), width="stretch")

        else:
            if fleet_result:
                st.error(f"❌ {fleet_result.get('error', 'Unknown error')}")
            else:
                st.info("👈 Configure warehouse, drops, and drones then click 'Optimize Fleet Routes'")


if __name__ == "__main__":
    main()
