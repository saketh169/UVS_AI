import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import numpy as np
import logging
from data import Station, Vehicle, Task

logging.basicConfig(level=logging.DEBUG)

def visualize_results(stations, vehicles, tasks, history):
    # Set up figure and axes
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(121, aspect='equal')
    info_ax = fig.add_subplot(122)
    
    # Station coordinates and properties
    station_coords = {s.id: s.location for s in stations}
    station_points = {s.id: s.charging_points for s in stations}
    
    # Set plot limits with padding
    x_coords = [loc[0] for loc in station_coords.values()]
    y_coords = [loc[1] for loc in station_coords.values()]
    x_min, x_max = min(x_coords, default=0) - 1, max(x_coords, default=0) + 1
    y_min, y_max = min(y_coords, default=0) - 1, max(y_coords, default=0) + 1
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Styling
    plt.style.use('ggplot')
    ax.set_title("Urban Vehicle Scheduler", fontsize=18, pad=20)
    ax.set_xlabel("X Coordinate", fontsize=14)
    ax.set_ylabel("Y Coordinate", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Info panel setup
    info_ax.axis('off')
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    
    def update(frame):
        ax.clear()
        info_ax.clear()
        info_ax.axis('off')
        
        # Handle empty history
        if not history:
            ax.text(
                (x_min + x_max) / 2, (y_min + y_max) / 2,
                "No actions to visualize\nRun scheduler to generate actions",
                fontsize=14, ha='center', color='red', weight='bold'
            )
            info_ax.text(
                0, 1, "No history available\nPlease run the scheduler",
                fontsize=12, va='top', fontfamily='monospace', color='red'
            )
            logging.debug("Empty history detected")
            return
        
        # Get current time slot
        time_idx = frame
        if time_idx >= len(history):
            logging.debug(f"Frame {frame} exceeds history length {len(history)}")
            return
        
        try:
            current_time, assignments, repositioning, vehicle_states, tasks_status = history[time_idx]
        except ValueError:
            logging.error(f"Invalid history tuple at index {time_idx}: {history[time_idx]}")
            ax.text(
                (x_min + x_max) / 2, (y_min + y_max) / 2,
                "Invalid history data\nCheck simulation output",
                fontsize=14, ha='center', color='red', weight='bold'
            )
            info_ax.text(
                0, 1, "Error: Invalid history data",
                fontsize=12, va='top', fontfamily='monospace', color='red'
            )
            return
        
        logging.debug(f"Visualizing Time: {current_time} min, Assignments: {assignments}, Repositioning: {repositioning}")
        
        # Validate vehicle_states
        if not vehicle_states or not all(isinstance(state, tuple) and len(state) == 2 for state in vehicle_states.values()):
            ax.text(
                (x_min + x_max) / 2, (y_min + y_max) / 2,
                "Invalid vehicle states\nCheck simulation output",
                fontsize=14, ha='center', color='red', weight='bold'
            )
            info_ax.text(
                0, 1, "Error: Invalid vehicle states in history",
                fontsize=12, va='top', fontfamily='monospace', color='red'
            )
            logging.error(f"Invalid vehicle_states: {vehicle_states}")
            return
        
        # Initialize vehicle positions and batteries
        vehicle_current = {v_id: state[0] for v_id, state in vehicle_states.items()}
        vehicle_batteries = {v_id: float(state[1]) for v_id, state in vehicle_states.items()}
        logging.debug(f"Vehicle states: {vehicle_states}")
        
        # Redraw stations
        for s_id, (x, y) in station_coords.items():
            ax.scatter(x, y, c='#1f77b4', s=200, marker='s', edgecolors='black', linewidth=1)
            ax.text(x, y + 0.3, f"{s_id} ({station_points[s_id]})", fontsize=10, ha='center', weight='bold')
        
        # Draw vehicles
        for v_id, s_id in vehicle_current.items():
            x, y = station_coords[s_id]
            offset = np.random.uniform(-0.1, 0.1, 2)
            ax.scatter(x + offset[0], y + offset[1], c='#ff7f0e', s=100, marker='^', alpha=0.8)
            ax.text(x + offset[0], y + offset[1] - 0.2, v_id, fontsize=9, ha='center', color='#ff7f0e')
        
        # Process assignments with dynamic offset
        num_assignments = len(assignments)
        for idx, assignment in enumerate(assignments):
            vehicle_id = assignment.split()[0]
            dest = assignment.split()[-3]
            origin = vehicle_current[vehicle_id]
            x1, y1 = station_coords[origin]
            x2, y2 = station_coords[dest]
            fee = float(assignment.split('Fees: ')[1].split(')')[0])
            
            # Dynamic offset to avoid overlap
            offset_y = 0.15 * ((idx % 2) * 2 - 1) * (1 + idx // 2) if num_assignments > 1 else 0
            arrow = patches.FancyArrowPatch(
                (x1, y1 + offset_y), (x2, y2 + offset_y), color='#1f77b4',
                arrowstyle='->', mutation_scale=20, linewidth=2
            )
            ax.add_patch(arrow)
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 0.25 + offset_y,
                f"{vehicle_id} (Task: ${fee})", fontsize=9, ha='center', color='#1f77b4', weight='bold'
            )
            vehicle_current[vehicle_id] = dest
            logging.debug(f"Assignment: {vehicle_id} from {origin} to {dest}, Battery: {vehicle_batteries[vehicle_id]}")
        
        # Process repositioning
        num_repositions = len(repositioning)
        for idx, reposition in enumerate(repositioning):
            vehicle_id = reposition.split()[0]
            origin = reposition.split()[3]
            dest = reposition.split()[-1]
            x1, y1 = station_coords[origin]
            x2, y2 = station_coords[dest]
            
            # Dynamic offset
            offset_y = -0.15 * ((idx % 2) * 2 - 1) * (1 + idx // 2) if num_repositions > 1 or assignments else 0
            arrow = patches.FancyArrowPatch(
                (x1, y1 + offset_y), (x2, y2 + offset_y), color='#d62728',
                arrowstyle='->', mutation_scale=20, linewidth=2
            )
            ax.add_patch(arrow)
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 0.25 + offset_y,
                f"{vehicle_id} (Reposition)", fontsize=9, ha='center', color='#d62728', weight='bold'
            )
            vehicle_current[vehicle_id] = dest
            logging.debug(f"Repositioning: {vehicle_id} from {origin} to {dest}, Battery: {vehicle_batteries[vehicle_id]}")
        
        # Draw legend
        ax.scatter([], [], c='#1f77b4', s=200, marker='s', label='Stations')
        ax.scatter([], [], c='#ff7f0e', s=100, marker='^', label='Vehicles')
        ax.plot([], [], c='#1f77b4', linewidth=2, label='Assignments')
        ax.plot([], [], c='#d62728', linewidth=2, label='Repositioning')
        ax.legend(loc='upper left', fontsize=10)
        
        # Calculate total fees
        total_fees = sum(float(a.split('Fees: ')[1].split(')')[0]) for a in assignments) if assignments else 0
        
        # Info panel content
        unassigned_tasks = [
            (origin, dest, fee, deadline) for origin, dest, fee, deadline, assigned in tasks_status if not assigned
        ]
        unassigned_tasks.sort(key=lambda x: x[3])
        info_text = [
            f"Time: {current_time} min",
            f"Total Fees Earned: ${total_fees:.1f}",
            "",
            "• Stations:",
            *[
                f"  - {s_id}: ({x:.1f}, {y:.1f}), {station_points[s_id]} chargers"
                for s_id, (x, y) in sorted(station_coords.items())
            ],
            "",
            "• Vehicles:",
            *[
                f"  - {v_id}: At {vehicle_current[v_id]}, Battery: {vehicle_batteries[v_id]:.1f}"
                for v_id in sorted(vehicle_current)
            ],
            "",
            f"• Assignments (Time {current_time} min):",
            *(
                [f"  - {a}" for a in assignments]
                if assignments else ["  - No assignments scheduled"]
            ),
            "",
            f"• Repositioning (Time {current_time} min):",
            *(
                [f"  - {r}" for r in repositioning]
                if repositioning else ["  - No repositioning scheduled"]
            ),
            "",
            "• Unassigned Tasks:",
            *(
                [
                    f"  - {origin}->{dest}: Fee ${fee}, Deadline {deadline}h"
                    for origin, dest, fee, deadline in unassigned_tasks
                ]
                if unassigned_tasks else ["  - No unassigned tasks"]
            )
        ]

        info_ax.text(0, 1, '\n'.join(info_text), fontsize=12, va='top', fontfamily='monospace')
        
        # Add time label to plot
        ax.text(
            x_min + 0.1, y_max - 0.3, f"Time: {current_time} min",
            fontsize=14, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
        )
        
        # Ensure axes are styled
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Create animation
    total_frames = max(len(history), 1)
    logging.debug(f"Total frames: {total_frames}")
    ani = FuncAnimation(
        fig, update, frames=total_frames,
        interval=3000, repeat=False
    )
    
    # Add pause/resume functionality
    paused = False
    def toggle_pause(event):
        nonlocal paused
        if event.key == ' ':
            paused = not paused
            if paused:
                ani.pause()
                logging.debug("Animation paused")
            else:
                ani.resume()
                logging.debug("Animation resumed")
    
    fig.canvas.mpl_connect('key_press_event', toggle_pause)
    
    plt.tight_layout()
    plt.show()