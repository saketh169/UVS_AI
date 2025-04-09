import networkx as nx
from data import Task, Vehicle

def preference_weight(vehicle, task_group):
    # Eq. (1): w(v, g) = w_f(g) + w_s(g) + w_e(v) if electricity sufficient
    total_fee = sum(t.fee for t in task_group)
    group_size = len(task_group)
    service_time = max(t.service_time for t in task_group)
    
    if vehicle.electricity <= service_time + 1:
        return 0
    
    # Normalized weights (simplified)
    w_f = total_fee / vehicle.electricity  # Revenue per electricity
    w_s = group_size / 10  # Proxy for supply-demand (assuming max 10 vehicles)
    w_e = vehicle.electricity / 100  # Normalized electricity
    return w_f + w_s + w_e

def pam_algorithm(tasks, vehicles, station_id):
    # Algorithm 1: PAM
    G = {}  # Task groups by destination
    W = []  # Preference-aware assignments
    
    # Line 3: Group tasks by destination
    for task in tasks:
        if task.origin == station_id:  # Only tasks from this station
            if task.dest not in G:
                G[task.dest] = []
            G[task.dest].append(task)
    
    # Lines 4-6: Compute weights
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
    
    # Line 8: Format assignments
    M = []
    for v_id, g_id in matching.items():
        if v_id in vehicle_nodes:
            dest = g_id.replace("g_", "")
            M.append(f"{v_id} -> {len(G[dest])} tasks to {dest}")
    
    return M