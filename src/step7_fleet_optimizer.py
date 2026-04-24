"""
============================================================
Step 7: Multi-Drone Fleet Optimizer (Google OR-Tools)
============================================================
Solves the Vehicle Routing Problem (VRP):
  - 1 Warehouse (depot) + N drop points
  - M drones with capacity and battery constraints
  - Uses A* pathfinder for safe distances (avoids buildings/zones)
  - OR-Tools assigns drops to drones optimally

This file is INDEPENDENT — does NOT modify step1-6 files.
============================================================
"""

import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MASTER_MAP_FILE = os.path.join(OUTPUT_DIR, "jaipur_master_map.geojson")

# Import our A* pathfinder
from step5_pathfinder import compute_path, ObstacleMap, AStarPathfinder
from step5_pathfinder import coord_to_grid, grid_to_coord, haversine
from step5_pathfinder import DRONE_CRUISE_ALT, DRONE_SPEED_KMH, DRONE_SAFETY_MARGIN, DRONE_BUILDING_BUFFER

# ──────────────────────────────────────────────
# DEFAULT FLEET CONFIG
# ──────────────────────────────────────────────
DEFAULT_DRONE_CAPACITY_KG = 2.5      # kg per drone
DEFAULT_NUM_DRONES = 3
DEFAULT_BATTERY_RATE = 3.0           # km per 1% battery
DEFAULT_BATTERY_START = 100          # percent
DEFAULT_DRONE_SPEED = 50             # km/h
DEFAULT_DRONE_ALTITUDE = 60          # meters
DEFAULT_SAFETY_MARGIN = 10           # meters
DEFAULT_BUILDING_BUFFER = 3          # meters


# ──────────────────────────────────────────────
# COST MATRIX BUILDER
# ──────────────────────────────────────────────

def build_cost_matrix(locations, permitted_yellow=None,
                      drone_altitude=None, drone_speed=None,
                      safety_margin=None, building_buffer=None):
    """
    Build cost matrix using A* pathfinder between all location pairs.

    Args:
        locations: List of (lat, lon) tuples. Index 0 = warehouse.
        permitted_yellow: List of permitted yellow zone IDs
        drone_altitude, drone_speed, safety_margin, building_buffer: drone params

    Returns:
        cost_matrix: NxN numpy array of distances in km
        paths: dict of {(i,j): path_coords} for visualization
    """
    n = len(locations)
    cost_matrix = np.zeros((n, n))
    paths = {}

    altitude = drone_altitude or DEFAULT_DRONE_ALTITUDE
    speed = drone_speed or DEFAULT_DRONE_SPEED
    margin = safety_margin or DEFAULT_SAFETY_MARGIN
    buffer = building_buffer or DEFAULT_BUILDING_BUFFER
    permitted = permitted_yellow or []

    print(f"\n  Building cost matrix for {n} locations...")
    print(f"  Total A* computations needed: {n * (n-1) // 2}")

    # Build obstacle map ONCE (shared across all A* runs)
    print(f"  Building shared obstacle map...")
    obs_map = ObstacleMap(
        MASTER_MAP_FILE, permitted,
        drone_altitude=altitude, safety_margin=margin,
        building_buffer=buffer
    )

    pathfinder = AStarPathfinder(obs_map)
    pathfinder._drone_speed = speed
    pathfinder._drone_altitude = altitude
    pathfinder._safety_margin = margin

    total_pairs = n * (n - 1) // 2
    computed = 0

    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = locations[i]
            lat2, lon2 = locations[j]

            # Run A* pathfinder
            result = pathfinder.find_path(lon1, lat1, lon2, lat2)

            if result["status"] == "SUCCESS":
                dist_km = result["metrics"]["path_distance_km"]
                cost_matrix[i][j] = dist_km
                cost_matrix[j][i] = dist_km
                paths[(i, j)] = result["path"]
                paths[(j, i)] = list(reversed(result["path"]))
            else:
                # No path found — set very high cost (will be avoided)
                cost_matrix[i][j] = 99999
                cost_matrix[j][i] = 99999
                paths[(i, j)] = []
                paths[(j, i)] = []

            computed += 1
            if computed % 5 == 0 or computed == total_pairs:
                print(f"    Computed {computed}/{total_pairs} pairs...")

    print(f"\n  Cost Matrix ({n}x{n}):")
    for i in range(n):
        row = [f"{cost_matrix[i][j]:6.2f}" for j in range(n)]
        label = "WH" if i == 0 else f"D{i}"
        print(f"    {label}: [{', '.join(row)}]")

    return cost_matrix, paths


# ──────────────────────────────────────────────
# OR-TOOLS VRP SOLVER
# ──────────────────────────────────────────────

def solve_fleet_routing(cost_matrix, demands, num_drones,
                        drone_capacity_kg, battery_rate_km_per_pct,
                        battery_start_pct=100):
    """
    Solve the Vehicle Routing Problem using Google OR-Tools.

    Args:
        cost_matrix: NxN numpy array of distances in km
        demands: List of weights in kg. Index 0 = warehouse (demand=0)
        num_drones: Number of available drones
        drone_capacity_kg: Max weight each drone can carry (kg)
        battery_rate_km_per_pct: km per 1% battery
        battery_start_pct: Starting battery percentage (default: 100)

    Returns:
        dict with assignments, metrics, and status
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    n = len(cost_matrix)
    max_range_km = battery_start_pct * battery_rate_km_per_pct

    print(f"\n  OR-Tools VRP Solver:")
    print(f"    Locations: {n} (1 warehouse + {n-1} drops)")
    print(f"    Drones: {num_drones}")
    print(f"    Capacity: {drone_capacity_kg} kg per drone")
    print(f"    Max range: {max_range_km:.0f} km ({battery_start_pct}% x {battery_rate_km_per_pct} km/%)")

    # Scale distances to integers (OR-Tools uses integers)
    SCALE = 1000  # Convert km to meters for more precision
    int_cost_matrix = (cost_matrix * SCALE).astype(int).tolist()

    # Create routing model
    manager = pywrapcp.RoutingIndexManager(n, num_drones, 0)  # 0 = depot
    routing = pywrapcp.RoutingModel(manager)

    # ── Distance callback ──
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_cost_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # ── Distance constraint (battery range) ──
    max_range_scaled = int(max_range_km * SCALE)
    routing.AddDimension(
        transit_callback_index,
        0,                    # no slack
        max_range_scaled,     # max distance per drone
        True,                 # start cumul at zero
        "Distance"
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")

    # ── Capacity constraint ──
    # Scale demands to integers (grams)
    WEIGHT_SCALE = 1000
    int_demands = [int(d * WEIGHT_SCALE) for d in demands]

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int_demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,                                              # no slack
        [int(drone_capacity_kg * WEIGHT_SCALE)] * num_drones,  # capacities
        True,                                           # start cumul at zero
        "Capacity"
    )

    # ── Allow dropping visits if infeasible ──
    # Large penalty for dropping a visit (we want to serve all drops)
    penalty = 100000 * SCALE
    for node in range(1, n):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # ── Search parameters ──
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(10)  # Max 10 seconds

    # ── Solve ──
    print(f"    Solving (max 10 seconds)...")
    start_time = time.time()
    solution = routing.SolveWithParameters(search_params)
    elapsed = time.time() - start_time

    if not solution:
        print(f"    [FAILED] No solution found!")
        return {
            "status": "NO_SOLUTION",
            "error": "OR-Tools could not find a feasible solution. Try adding more drones or increasing capacity.",
            "assignments": [],
        }

    # ── Extract solution ──
    print(f"    [OK] Solution found in {elapsed:.2f}s")

    assignments = []
    total_distance = 0
    total_weight = 0
    drones_used = 0

    for vehicle_id in range(num_drones):
        index = routing.Start(vehicle_id)
        route_nodes = []
        route_distance = 0
        route_weight = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            if node > 0:
                route_weight += demands[node]

            next_index = solution.Value(routing.NextVar(index))
            route_distance += cost_matrix[node][manager.IndexToNode(next_index)]
            index = next_index

        # Add return to depot
        route_nodes.append(0)

        # Only include drones that actually visit drops
        drop_nodes = [n for n in route_nodes if n > 0]

        if drop_nodes:
            drones_used += 1
            battery_used = route_distance / battery_rate_km_per_pct
            battery_remaining = battery_start_pct - battery_used

            assignment = {
                "drone_id": vehicle_id + 1,
                "route_nodes": route_nodes,
                "drop_nodes": drop_nodes,
                "num_drops": len(drop_nodes),
                "distance_km": round(route_distance, 3),
                "weight_kg": round(route_weight, 3),
                "battery_used_pct": round(battery_used, 1),
                "battery_remaining_pct": round(battery_remaining, 1),
                "travel_time_min": round((route_distance / DEFAULT_DRONE_SPEED) * 60, 1),
            }
            assignments.append(assignment)
            total_distance += route_distance
            total_weight += route_weight

            print(f"    Drone {vehicle_id+1}: {' -> '.join(['WH' if n==0 else f'D{n}' for n in route_nodes])} "
                  f"({route_distance:.2f}km, {route_weight:.2f}kg, {battery_used:.1f}% battery)")

    # Unserved drops
    unserved = []
    for node in range(1, n):
        served = any(node in a["drop_nodes"] for a in assignments)
        if not served:
            unserved.append(node)

    if unserved:
        print(f"    [WARN] Unserved drops: {unserved}")

    result = {
        "status": "SUCCESS",
        "assignments": assignments,
        "summary": {
            "total_drones_available": num_drones,
            "drones_used": drones_used,
            "total_drops": n - 1,
            "drops_served": sum(a["num_drops"] for a in assignments),
            "drops_unserved": len(unserved),
            "unserved_nodes": unserved,
            "total_distance_km": round(total_distance, 3),
            "total_weight_kg": round(total_weight, 3),
            "computation_time_s": round(elapsed, 3),
            "max_range_km": max_range_km,
        },
    }

    print(f"\n  Fleet Summary:")
    print(f"    Drones used: {drones_used}/{num_drones}")
    print(f"    Drops served: {result['summary']['drops_served']}/{n-1}")
    print(f"    Total distance: {total_distance:.2f} km")
    print(f"    Total weight: {total_weight:.2f} kg")

    return result


# ──────────────────────────────────────────────
# FULL FLEET SOLVE (PUBLIC API)
# ──────────────────────────────────────────────

def solve_fleet(warehouse, drops, num_drones,
                drone_capacity_kg=None, battery_rate=None,
                drone_altitude=None, drone_speed=None,
                safety_margin=None, building_buffer=None,
                permitted_yellow=None):
    """
    Main API — solve multi-drone fleet routing.

    Args:
        warehouse: (lat, lon) of warehouse
        drops: List of {"lat": float, "lon": float, "weight_kg": float, "name": str}
        num_drones: Number of available drones
        drone_capacity_kg: Max weight per drone (kg)
        battery_rate: km per 1% battery
        drone_altitude, drone_speed, safety_margin, building_buffer: drone params
        permitted_yellow: List of permitted yellow zone IDs

    Returns:
        dict with assignments, paths, metrics
    """
    capacity = drone_capacity_kg or DEFAULT_DRONE_CAPACITY_KG
    rate = battery_rate or DEFAULT_BATTERY_RATE

    print(f"\n{'='*60}")
    print(f"  MULTI-DRONE FLEET OPTIMIZATION")
    print(f"{'='*60}")
    print(f"  Warehouse: ({warehouse[0]:.6f}, {warehouse[1]:.6f})")
    print(f"  Drop points: {len(drops)}")
    print(f"  Drones: {num_drones}")
    print(f"  Capacity: {capacity} kg, Battery: 100% x {rate} km/% = {100*rate:.0f}km range")

    # Build location list: [warehouse, drop1, drop2, ...]
    locations = [warehouse]
    demands = [0.0]  # warehouse demand = 0

    for d in drops:
        locations.append((d["lat"], d["lon"]))
        demands.append(d["weight_kg"])

    # 1. Build cost matrix using A*
    print(f"\n  [1/3] Computing cost matrix ({len(locations)}x{len(locations)})...")
    cost_matrix, paths = build_cost_matrix(
        locations, permitted_yellow,
        drone_altitude, drone_speed, safety_margin, building_buffer
    )

    # 2. Solve VRP with OR-Tools
    print(f"\n  [2/3] Solving fleet routing with OR-Tools...")
    vrp_result = solve_fleet_routing(
        cost_matrix, demands, num_drones,
        capacity, rate
    )

    if vrp_result["status"] != "SUCCESS":
        return vrp_result

    # 3. Attach actual paths to assignments
    print(f"\n  [3/3] Attaching safe flight paths...")
    for assignment in vrp_result["assignments"]:
        route_nodes = assignment["route_nodes"]
        full_path = []

        for k in range(len(route_nodes) - 1):
            from_node = route_nodes[k]
            to_node = route_nodes[k + 1]
            leg_path = paths.get((from_node, to_node), [])
            if leg_path:
                if full_path:
                    full_path.extend(leg_path[1:])  # skip duplicate start
                else:
                    full_path.extend(leg_path)

        assignment["full_path"] = full_path

    vrp_result["locations"] = locations
    vrp_result["cost_matrix"] = cost_matrix.tolist()
    vrp_result["drops_info"] = drops

    print(f"\n  {'='*55}")
    print(f"  [OK] Fleet optimization complete!")
    print(f"  {'='*55}")

    return vrp_result


# ──────────────────────────────────────────────
# TEST
# ──────────────────────────────────────────────

def main():
    """Test fleet optimizer with sample data."""
    print("=" * 60)
    print("  STEP 7: Fleet Optimizer Test")
    print("=" * 60)

    # Test data
    warehouse = (26.920, 75.780)
    drops = [
        {"lat": 26.880, "lon": 75.790, "weight_kg": 0.5, "name": "Drop 1"},
        {"lat": 26.850, "lon": 75.810, "weight_kg": 1.2, "name": "Drop 2"},
        {"lat": 26.870, "lon": 75.830, "weight_kg": 0.8, "name": "Drop 3"},
        {"lat": 26.900, "lon": 75.850, "weight_kg": 1.5, "name": "Drop 4"},
    ]

    result = solve_fleet(
        warehouse=warehouse,
        drops=drops,
        num_drones=2,
        drone_capacity_kg=2.5,
        battery_rate=3.0,
        permitted_yellow=["Yellow-101", "Yellow-107"],
    )

    if result["status"] == "SUCCESS":
        print(f"\n  Test PASSED!")
        for a in result["assignments"]:
            print(f"    Drone {a['drone_id']}: {a['num_drops']} drops, "
                  f"{a['distance_km']:.2f}km, {a['battery_used_pct']:.1f}% battery")
    else:
        print(f"\n  Test result: {result['status']}")


if __name__ == "__main__":
    main()
