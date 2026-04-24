"""
============================================================
Model Comparison - Drone Pathfinding Algorithms
============================================================
Benchmarks 5 pathfinding algorithms on the same obstacle map.

Algorithms: A*, Best-First Search, BFS, Theta*, RRT*

Outputs:
  - output/algorithm_comparison.csv
  - output/comparison_path_distance.png
  - output/comparison_computation_time.png
  - output/comparison_nodes_explored.png
  - output/comparison_success_rate.png

Usage:  python src/model_comparison.py
============================================================
"""

import os
import sys
import csv
import time
import warnings
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Path Setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from step5_pathfinder import (
    ObstacleMap, ALGORITHM_MAP, MASTER_MAP_FILE, OUTPUT_DIR,
    DRONE_SPEED_KMH, DRONE_CRUISE_ALT, DRONE_SAFETY_MARGIN,
    DRONE_BUILDING_BUFFER, haversine,
)

# ---- TEST SCENARIOS ----
TEST_SCENARIOS = [
    {
        "id": "S1_NorthSouth",
        "name": "North to South (long)",
        "start_lat": 26.920, "start_lon": 75.780,
        "end_lat":   26.850, "end_lon":   75.810,
    },
    {
        "id": "S2_EastWest",
        "name": "East to West (medium)",
        "start_lat": 26.870, "start_lon": 75.860,
        "end_lat":   26.870, "end_lon":   75.760,
    },
    {
        "id": "S3_ShortHop",
        "name": "Short hop (nearby)",
        "start_lat": 26.850, "start_lon": 75.800,
        "end_lat":   26.860, "end_lon":   75.810,
    },
    {
        "id": "S4_Diagonal",
        "name": "Diagonal NE to SW (long)",
        "start_lat": 26.910, "start_lon": 75.850,
        "end_lat":   26.800, "end_lon":   75.750,
    },
    {
        "id": "S5_CityCenter",
        "name": "City center traverse",
        "start_lat": 26.860, "start_lon": 75.770,
        "end_lat":   26.880, "end_lon":   75.830,
    },
]

PERMITTED_YELLOW = ["Yellow-101", "Yellow-107", "Yellow-122"]

# ---- Color palette for 5 algorithms ----
ALGO_COLORS = {
    "A*":                "#00ff88",
    "Best-First Search": "#ff9f43",
    "BFS":               "#54a0ff",
    "Theta*":            "#48dbfb",
    "RRT*":              "#f368e0",
}


def run_comparison():
    """Run 5 algorithms on 5 scenarios."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  PATHFINDING ALGORITHM COMPARISON (5 Models)")
    print("=" * 70)

    print("\n  [1/3] Building obstacle map...")
    obs_map = ObstacleMap(
        MASTER_MAP_FILE,
        permitted_yellow_zones=PERMITTED_YELLOW,
        drone_altitude=DRONE_CRUISE_ALT,
        safety_margin=DRONE_SAFETY_MARGIN,
        building_buffer=DRONE_BUILDING_BUFFER,
    )

    algo_names = list(ALGORITHM_MAP.keys())
    total = len(algo_names) * len(TEST_SCENARIOS)
    print(f"\n  [2/3] Running {len(algo_names)} algorithms x {len(TEST_SCENARIOS)} scenarios = {total} runs...\n")

    results = []

    for scenario in TEST_SCENARIOS:
        sid = scenario["id"]
        sname = scenario["name"]
        s_lat, s_lon = scenario["start_lat"], scenario["start_lon"]
        e_lat, e_lon = scenario["end_lat"], scenario["end_lon"]
        direct_dist = haversine(s_lat, s_lon, e_lat, e_lon)

        print(f"  -- Scenario: {sid} | {sname} --")
        print(f"     ({s_lat}, {s_lon}) -> ({e_lat}, {e_lon})  Direct: {direct_dist:.0f}m")

        for algo_name in algo_names:
            PathfinderClass = ALGORITHM_MAP[algo_name]
            print(f"    Running {algo_name:20s} ... ", end="", flush=True)

            random.seed(42)
            np.random.seed(42)

            try:
                pathfinder = PathfinderClass(obs_map)
                pathfinder._drone_speed = DRONE_SPEED_KMH
                pathfinder._drone_altitude = DRONE_CRUISE_ALT
                pathfinder._safety_margin = DRONE_SAFETY_MARGIN

                t0 = time.time()
                result = pathfinder.find_path(s_lon, s_lat, e_lon, e_lat)
                elapsed = time.time() - t0

                success = result["status"] == "SUCCESS"
                metrics = result.get("metrics", {})

                row = {
                    "scenario_id": sid,
                    "scenario_name": sname,
                    "algorithm": algo_name,
                    "success": success,
                    "path_distance_m": metrics.get("path_distance_m", 0) if success else 0,
                    "path_distance_km": metrics.get("path_distance_km", 0) if success else 0,
                    "computation_time_s": round(elapsed, 4),
                    "nodes_explored": metrics.get("nodes_explored", metrics.get("raw_waypoints", 0)),
                    "raw_waypoints": metrics.get("raw_waypoints", 0) if success else 0,
                    "smoothed_waypoints": metrics.get("waypoints", 0) if success else 0,
                    "detour_ratio": metrics.get("detour_ratio", 0) if success else 0,
                    "direct_distance_m": round(direct_dist, 1),
                    "travel_time_min": metrics.get("travel_time_min", 0) if success else 0,
                }
                results.append(row)

                status = "OK" if success else "FAIL"
                dist_str = f"{row['path_distance_m']:.0f}m" if success else "FAILED"
                print(f"{status:>4s}  {dist_str:>8s}  {elapsed:.3f}s  nodes={row['nodes_explored']}")

            except Exception as e:
                print(f"FAIL  ERROR: {e}")
                results.append({
                    "scenario_id": sid, "scenario_name": sname, "algorithm": algo_name,
                    "success": False, "path_distance_m": 0, "path_distance_km": 0,
                    "computation_time_s": 0, "nodes_explored": 0, "raw_waypoints": 0,
                    "smoothed_waypoints": 0, "detour_ratio": 0,
                    "direct_distance_m": round(direct_dist, 1), "travel_time_min": 0,
                })

        print()

    return results


def save_csv(results, output_path):
    """Save results to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved: {output_path}")


def print_summary_table(results):
    """Print formatted summary table."""
    from collections import defaultdict
    agg = defaultdict(lambda: {"runs": 0, "successes": 0, "total_dist": 0, "total_time": 0, "total_nodes": 0, "total_detour": 0})

    for r in results:
        algo = r["algorithm"]
        agg[algo]["runs"] += 1
        if r["success"]:
            agg[algo]["successes"] += 1
            agg[algo]["total_dist"] += r["path_distance_m"]
            agg[algo]["total_time"] += r["computation_time_s"]
            agg[algo]["total_nodes"] += r["nodes_explored"]
            agg[algo]["total_detour"] += r["detour_ratio"]

    print("\n" + "=" * 90)
    print(f"  {'ALGORITHM COMPARISON SUMMARY':^86}")
    print("=" * 90)
    print(f"  {'Algorithm':<22s} {'Success':>8s} {'Avg Dist (m)':>14s} {'Avg Time (s)':>14s} {'Avg Nodes':>12s} {'Avg Detour':>12s}")
    print("-" * 90)

    for algo in ALGORITHM_MAP.keys():
        a = agg[algo]
        n_ok = a["successes"] or 1
        rate = f"{a['successes']}/{a['runs']}"
        print(f"  {algo:<22s} {rate:>8s} {a['total_dist']/n_ok:>14.1f} {a['total_time']/n_ok:>14.4f} {a['total_nodes']/n_ok:>12.0f} {a['total_detour']/n_ok:>12.2f}x")

    print("=" * 90)


# ---- INDIVIDUAL CHART GENERATORS ----

def _setup_chart(title, ylabel, algo_names, values, filename):
    """Create a single professional bar chart and save as PNG."""
    n = len(algo_names)
    colors = [ALGO_COLORS.get(a, "#888888") for a in algo_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#101025")

    x = np.arange(n)
    bar_width = 0.55

    bars = ax.bar(x, values, bar_width, color=colors, alpha=0.9,
                  edgecolor="#ffffff", linewidth=0.8,
                  zorder=3)

    # Gradient-like shadow bars behind
    for i, bar in enumerate(bars):
        ax.bar(x[i], values[i] * 0.97, bar_width + 0.05,
               color=colors[i], alpha=0.15, zorder=2)

    # Value labels on top
    for bar, val in zip(bars, values):
        if val > 0:
            label = f"{val:.0f}" if val >= 10 else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02 + max(values) * 0.01,
                    label, ha="center", va="bottom",
                    color="#ffffff", fontsize=11, fontweight="bold")

    # Title
    ax.set_title(title, color="#00ff88", fontsize=18, fontweight="bold", pad=20,
                 fontfamily="sans-serif")

    # Axes
    ax.set_ylabel(ylabel, color="#8892b0", fontsize=12, fontweight="500")
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, fontsize=11, color="#ccddff", fontweight="600")
    ax.tick_params(axis="y", colors="#8892b0", labelsize=10)

    # Spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_edgecolor("#2a2a4a")

    # Grid
    ax.grid(True, axis="y", color="#2a2a4a", linewidth=0.6, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Y-axis starts at 0
    ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {filepath}")


def generate_separate_charts(results):
    """Generate 4 separate professional PNG charts."""
    from collections import defaultdict

    agg = defaultdict(lambda: {"successes": 0, "runs": 0, "dists": [], "times": [], "nodes": []})
    for r in results:
        algo = r["algorithm"]
        agg[algo]["runs"] += 1
        if r["success"]:
            agg[algo]["successes"] += 1
            agg[algo]["dists"].append(r["path_distance_m"])
            agg[algo]["times"].append(r["computation_time_s"])
            agg[algo]["nodes"].append(r["nodes_explored"])

    algo_names = list(ALGORITHM_MAP.keys())

    avg_dists = [np.mean(agg[a]["dists"]) if agg[a]["dists"] else 0 for a in algo_names]
    avg_times = [np.mean(agg[a]["times"]) if agg[a]["times"] else 0 for a in algo_names]
    avg_nodes = [np.mean(agg[a]["nodes"]) if agg[a]["nodes"] else 0 for a in algo_names]
    success_rates = [agg[a]["successes"] / max(agg[a]["runs"], 1) * 100 for a in algo_names]

    # Chart 1: Path Distance
    _setup_chart(
        "Average Path Distance by Algorithm",
        "Distance (meters)",
        algo_names, avg_dists,
        "comparison_path_distance.png"
    )

    # Chart 2: Computation Time
    _setup_chart(
        "Average Computation Time by Algorithm",
        "Time (seconds)",
        algo_names, avg_times,
        "comparison_computation_time.png"
    )

    # Chart 3: Nodes Explored
    _setup_chart(
        "Average Nodes Explored by Algorithm",
        "Nodes Explored",
        algo_names, avg_nodes,
        "comparison_nodes_explored.png"
    )

    # Chart 4: Success Rate (special - percentage)
    n = len(algo_names)
    colors = [ALGO_COLORS.get(a, "#888888") for a in algo_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#101025")

    x = np.arange(n)
    bars = ax.bar(x, success_rates, 0.55, color=colors, alpha=0.9,
                  edgecolor="#ffffff", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}%", ha="center", va="bottom",
                color="#ffffff", fontsize=13, fontweight="bold")

    ax.set_title("Success Rate by Algorithm", color="#00ff88", fontsize=18,
                 fontweight="bold", pad=20)
    ax.set_ylabel("Success Rate (%)", color="#8892b0", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, fontsize=11, color="#ccddff", fontweight="600")
    ax.tick_params(axis="y", colors="#8892b0", labelsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_edgecolor("#2a2a4a")
    ax.grid(True, axis="y", color="#2a2a4a", linewidth=0.6, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "comparison_success_rate.png")
    fig.savefig(filepath, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved: {filepath}")


# ---- MAIN ----
def main():
    results = run_comparison()

    csv_path = os.path.join(OUTPUT_DIR, "algorithm_comparison.csv")
    save_csv(results, csv_path)

    print_summary_table(results)

    print("\n  [3/3] Generating comparison charts...")
    generate_separate_charts(results)

    print(f"\n  [DONE] Comparison complete!")
    print(f"     CSV:    {csv_path}")
    print(f"     Charts: output/comparison_*.png (4 files)")
    print(f"  {'=' * 55}")


if __name__ == "__main__":
    main()
