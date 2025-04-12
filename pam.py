import networkx as nx  # type: ignore
from data import Task, Vehicle
from math import sqrt

def preference_weight(vehicle, task_group):
    # Enhanced Eq. (1): w(v, g) = w_f(g) + w_d(g) + w_e(v) - w_c(v, g)
    total_fee = sum(t.fee for t in task_group)
    group_size = len(task_group)
    service_time = max(t.service_time for t in task_group)
    
    if vehicle.electricity <= service_time + 1:
        return 0
    
    # Calculate average distance to destination (Euclidean distance from vehicle station)
    avg_distance = sum(sqrt(sum((v_loc - t_loc) ** 2 for v_loc, t_loc in zip(vehicle.location, t.location))) / group_size 
                       for t in task_group) if group_size else 0
    
    # Normalized weights
    w_f = total_fee / (vehicle.electricity + 1)  # Revenue per electricity, avoid division by zero
    w_d = 1 / (avg_distance + 1) if avg_distance else 1  # Inverse distance penalty
    w_e = vehicle.electricity / 100  # Normalized electricity
    w_c = service_time / (vehicle.capacity + 1)  # Cost penalty based on service time and capacity
    return w_f + w_d + w_e - w_c

def pam_algorithm(tasks, vehicles, station_id):
    # Algorithm 1: PAM with enhanced weighting
    G = {}  # Task groups by destination
    W = []  # Preference-aware assignments
    
    # Line 3: Group tasks by destination
    for task in tasks:
        if task.origin == station_id:  # Only tasks from this station
            if task.dest not in G:
                G[task.dest] = []
            G[task.dest].append(task)
    
    # Lines 4-6: Compute weights with vehicle-task compatibility
    for dest, group in G.items():
        for v in vehicles:
            if v.station == station_id and v.capacity >= len(group):
                weight = preference_weight(v, group)
                W.append((v.id, dest, weight))
    
    # Line 7: Simplified KM (max weight matching)
    G_nx = nx.Graph()
    vehicle_nodes = [v.id for v in vehicles if v.station == station_id]
    group_nodes = [f"g_{dest}" for dest in G.keys()]
    G_nx.add_nodes_from(vehicle_nodes, bipartite=0)
    G_nx.add_nodes_from(group_nodes, bipartite=1)
    for v_id, dest, w in W:
        G_nx.add_edge(v_id, f"g_{dest}", weight=w)
    
    matching = nx.bipartite.maximum_matching(G_nx, top_nodes=vehicle_nodes)
    
    # Line 8: Format assignments with task details
    M = []
    for v_id, g_id in matching.items():
        if v_id in vehicle_nodes:
            dest = g_id.replace("g_", "")
            tasks_assigned = [t for t in tasks if t.origin == station_id and t.dest == dest]
            M.append(f"{v_id} -> {len(tasks_assigned)} tasks to {dest} (Fee: {sum(t.fee for t in tasks_assigned):.1f})")
    
    return M