import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from data import Station, Vehicle, Task

def visualize_results(stations, vehicles, tasks, assignments, repositioning):
    fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax_map.set_title("UVS Simulation")
    ax_map.set_xlabel("X Coordinate")
    ax_map.set_ylabel("Y Coordinate")
    ax_map.grid(True)
    
    # Initialize plot elements
    station_plots = {}
    vehicle_plots = {}
    arrow_plots = []
    
    # Plot stations
    for s in stations:
        station_plots[s.id] = ax_map.scatter(s.location[0], s.location[1], s=100, label=f"Station {s.id}")
        ax_map.text(s.location[0], s.location[1] + 0.1, s.id)
    
    # Plot vehicles (initial)
    for v in vehicles:
        s = next(s for s in stations if s.id == v.station)
        vehicle_plots[v.id] = ax_map.scatter(s.location[0], s.location[1], s=50, marker='x', label=f"Vehicle {v.id}")
    
    # Info panel
    ax_info.axis('off')
    info_text = ax_info.text(0.1, 0.9, "Time: 0\nAssignments:\nRepositioning:", ha='left', va='top')
    
    def update(frame):
        nonlocal arrow_plots
        # Clear previous arrows
        for arrow in arrow_plots:
            arrow.remove()
        arrow_plots = []
        
        # Update time
        current_time = frame * 5  # 5-minute slots
        
        # Plot assignments
        for a in assignments:
            v_id, dest = a.split(" -> ")[0], a.split(" to ")[1].split(" ")[0]
            v = next(v for v in vehicles if v.id == v_id)
            s_from = next(s for s in stations if s.id == v.station)
            s_to = next(s for s in stations if s.id == dest)
            arrow = ax_map.arrow(s_from.location[0], s_from.location[1], 
                                 s_to.location[0] - s_from.location[0], 
                                 s_to.location[1] - s_from.location[1],
                                 color='blue', linestyle='--')
            arrow_plots.append(arrow)
        
        # Plot repositioning (animate over frames)
        if frame < len(repositioning):
            r = repositioning[frame]
            v_id, to_id = r.split(" moves from ")[0], r.split(" to ")[1].strip()
            if "stays" not in r:
                v = next(v for v in vehicles if v.id == v_id)
                s_from = next(s for s in stations if s.id == v.station)
                s_to = next(s for s in stations if s.id == to_id)
                arrow = ax_map.arrow(s_from.location[0], s_from.location[1], 
                                     s_to.location[0] - s_from.location[0], 
                                     s_to.location[1] - s_from.location[1],
                                     color='red')
                arrow_plots.append(arrow)
                # Update vehicle position
                vehicle_plots[v_id].set_offsets([s_to.location[0], s_to.location[1]])
        
        # Update info panel
        assigned = "\n".join(assignments[:frame+1]) if frame < len(assignments) else "\n".join(assignments)
        repos = "\n".join(repositioning[:frame+1]) if frame < len(repositioning) else "\n".join(repositioning)
        unassigned = sum(1 for t in tasks if not t.assigned)
        battery_info = "\n".join(f"{v.id}: {v.electricity:.1f}%" for v in vehicles)
        info_text.set_text(f"Time: {current_time} min\nAssignments:\n{assigned}\n\nRepositioning:\n{repos}\n\nUnassigned Tasks: {unassigned}\nBattery Levels:\n{battery_info}")
    
    # Animation
    ani = animation.FuncAnimation(fig, update, frames=10, interval=1000, repeat=False)
    
    # Avoid duplicate legend entries
    handles, labels = ax_map.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_map.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.show()