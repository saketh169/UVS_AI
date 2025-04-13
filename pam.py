import networkx as nx
from data import Task, Vehicle

def preference_weight(vehicle, task_group, total_vehicles):
    total_fee = sum(t.fee for t in task_group)
    group_size = len(task_group)
    service_time = max(t.service_time for t in task_group)
    
    if vehicle.electricity <= service_time + 1 or group_size > vehicle.capacity:
        return 0
    
    w_f = total_fee / vehicle.electricity
    w_s = group_size / total_vehicles
    w_e = vehicle.electricity / 100
    return w_f + w_s + w_e

def pam_algorithm(tasks, vehicles, station_id, current_time=0):
    G = {}
    W = []
    total_vehicles = len(vehicles)
    
    # Group unassigned tasks by destination
    for task in tasks:
        if task.origin == station_id and not task.assigned and task.deadline >= current_time + task.service_time:
            if task.dest not in G:
                G[task.dest] = []
            G[task.dest].append(task)
    
    # Compute weights
    for dest, group in G.items():
        for v in vehicles:
            if v.station == station_id:
                weight = preference_weight(v, group, total_vehicles)
                if weight > 0:
                    W.append((v.id, dest, weight))
    
    # Bipartite matching
    G_nx = nx.Graph()
    vehicle_nodes = [v.id for v in vehicles if v.station == station_id]
    group_nodes = [f"g_{dest}" for dest in G.keys()]
    G_nx.add_nodes_from(vehicle_nodes, bipartite=0)
    G_nx.add_nodes_from(group_nodes, bipartite=1)
    for v_id, dest, w in W:
        G_nx.add_edge(v_id, f"g_{dest}", weight=w)
    
    matching = nx.bipartite.maximum_matching(G_nx, top_nodes=vehicle_nodes)
    
    # Format assignments and mark tasks
    assignments = []
    for v_id, g_id in matching.items():
        if v_id in vehicle_nodes:
            dest = g_id.replace("g_", "")
            tasks_assigned = G[dest]
            for task in tasks_assigned:
                task.assigned = True
            assignments.append(f"{v_id} -> {len(tasks_assigned)} tasks to {dest} (Fees: {sum(t.fee for t in tasks_assigned):.1f})")
    
    return assignments