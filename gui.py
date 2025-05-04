import tkinter as tk
from tkinter import scrolledtext, ttk
from data import Station, Task, Vehicle
from pam import pam_algorithm
from reposition import reposition_vehicles
from visualize import visualize_results
import logging
import re

logging.basicConfig(level=logging.DEBUG)

class UVSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Urban Vehicle Scheduler")
        self.root.geometry("900x700")
        
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 12))
        style.configure("TEntry", font=("Arial", 12))
        
        main_frame = ttk.Frame(root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Stations (e.g., M-0-0-5, B-2-2-3):", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.stations_input = ttk.Entry(main_frame, width=60)
        self.stations_input.grid(row=0, column=1, columnspan=2, pady=5, padx=5)
        
        ttk.Label(main_frame, text="Tasks (e.g., M-B-6-1-10, M-Q-4-2-12):", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.tasks_input = ttk.Entry(main_frame, width=60)
        self.tasks_input.grid(row=1, column=1, columnspan=2, pady=5, padx=5)
        
        ttk.Label(main_frame, text="Vehicles (e.g., V1-M-85-5, V2-B-60-3):", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.vehicles_input = ttk.Entry(main_frame, width=60)
        self.vehicles_input.grid(row=2, column=1, columnspan=2, pady=5, padx=5)
        
        ttk.Label(main_frame, text="Station to Process:", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.station_process = ttk.Entry(main_frame, width=10)
        self.station_process.grid(row=3, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(main_frame, text="Algorithm (rdr/rar):", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky=tk.W, pady=5)
        self.algo_choice = ttk.Entry(main_frame, width=10)
        self.algo_choice.grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)
        
        self.time_label = ttk.Label(main_frame, text="Current Time: 0 min", font=("Arial", 12, "bold"))
        self.time_label.grid(row=5, column=0, sticky=tk.W, pady=10)
        
        ttk.Button(main_frame, text="Run Scheduler", command=self.run).grid(row=6, column=0, pady=10, padx=5)
        ttk.Button(main_frame, text="Start Visualization", command=self.visualize).grid(row=6, column=1, pady=10, padx=5)
        ttk.Button(main_frame, text="Reset Time", command=self.reset_time).grid(row=6, column=2, pady=10, padx=5)
        ttk.Button(main_frame, text="Clear Inputs", command=self.clear_inputs).grid(row=7, column=0, pady=10, padx=5)
        
        ttk.Label(main_frame, text="Output:", font=("Arial", 12, "bold")).grid(row=8, column=0, sticky=tk.W, pady=5)
        self.output = scrolledtext.ScrolledText(main_frame, width=70, height=15, font=("Arial", 10))
        self.output.grid(row=9, column=0, columnspan=3, pady=5, padx=5)
        
        ttk.Label(main_frame, text="Errors:", font=("Arial", 12, "bold")).grid(row=10, column=0, sticky=tk.W, pady=5)
        self.error_output = scrolledtext.ScrolledText(main_frame, width=70, height=5, font=("Arial", 10))
        self.error_output.grid(row=11, column=0, columnspan=3, pady=5, padx=5)
        
        self.stations = []
        self.vehicles = []
        self.tasks = []
        self.current_time = 0
        self.history = []  # List of (time, assignments, repositioning, vehicle_states, tasks_status)
    
    def reset_time(self):
        self.current_time = 0
        self.history = []
        self.vehicles = []  # Clear vehicles to reset state
        self.tasks = []  # Clear tasks
        self.stations = []  # Clear stations
        self.time_label.config(text=f"Current Time: {self.current_time} min")
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "Time reset to 0 min\n")
        logging.debug("Time reset to 0")
    
    def clear_inputs(self):
        self.stations_input.delete(0, tk.END)
        self.tasks_input.delete(0, tk.END)
        self.vehicles_input.delete(0, tk.END)
        self.station_process.delete(0, tk.END)
        self.algo_choice.delete(0, tk.END)
        self.output.delete(1.0, tk.END)
        self.error_output.delete(1.0, tk.END)
        self.current_time = 0
        self.history = []
        self.vehicles = []
        self.tasks = []
        self.stations = []
        self.time_label.config(text=f"Current Time: {self.current_time} min")
        logging.debug("Inputs cleared")
    
    def validate_inputs(self):
        try:
            raw_stations = self.stations_input.get()
            logging.debug(f"Raw stations input: '{raw_stations}'")
            
            stations = [s.strip() for s in raw_stations.split(",") if s.strip()]
            tasks = [t.strip() for t in self.tasks_input.get().split(",") if t.strip()]
            vehicles = [v.strip() for v in self.vehicles_input.get().split(",") if v.strip()]
            station_id = self.station_process.get().strip()
            algo = self.algo_choice.get().lower().strip()
            
            logging.debug(f"Parsed stations: {stations}")
            
            for s in stations:
                parts = s.split("-")
                if len(parts) != 4:
                    raise ValueError("Stations format: id-x-y-charging_points (e.g., M-0-0-5)")
                id_part, x_part, y_part, charge_part = parts
                if not re.match(r'^[a-zA-Z0-9]+$', id_part):
                    raise ValueError("Stations: ID must be alphanumeric (e.g., M, B, Q)")
                try:
                    float(x_part)
                    float(y_part)
                    int(charge_part)
                except ValueError:
                    raise ValueError("Stations: x, y, charging_points must be numbers")
            
            if tasks:
                if not all(len(t.split("-")) == 5 for t in tasks):
                    raise ValueError("Tasks format: origin-dest-fee-deadline-service_time")
                for t in tasks:
                    parts = t.split("-")
                    try:
                        float(parts[2])
                        float(parts[3])
                        float(parts[4])
                    except ValueError:
                        raise ValueError("Tasks: fee, deadline, service_time must be numbers")
            
            if not all(len(v.split("-")) == 4 for v in vehicles):
                raise ValueError("Vehicles format: id-station-electricity-capacity")
            for v in vehicles:
                parts = v.split("-")
                try:
                    float(parts[2])
                    int(parts[3])
                except ValueError:
                    raise ValueError("Vehicles: electricity must be number, capacity must be integer")
            
            if not station_id:
                raise ValueError("Enter a station ID")
            if algo not in ['rdr', 'rar']:
                raise ValueError("Algorithm must be 'rdr' or 'rar'")
            
            station_ids = [s.split("-")[0] for s in stations]
            if station_id not in station_ids:
                raise ValueError(f"Station ID '{station_id}' not found in stations")
            
            return stations, tasks, vehicles, station_id, algo
        except Exception as e:
            raise ValueError(f"Input error: {str(e)}")
    
    def run(self):
        self.error_output.delete(1.0, tk.END)
        try:
            stations_str, tasks_str, vehicles_str, station_id, algo = self.validate_inputs()
            
            # Update stations only if empty (persist across runs)
            if not self.stations:
                self.stations = [
                    Station(
                        id=s.split("-")[0],
                        location=(float(s.split("-")[1]), float(s.split("-")[2])),
                        charging_points=int(s.split("-")[3])
                    )
                    for s in stations_str
                ]
            
            # Update tasks only if empty
            if not self.tasks:
                self.tasks = [Task(*t.split("-")) for t in tasks_str] if tasks_str else []
            
            # Update vehicles only if empty
            if not self.vehicles:
                self.vehicles = [Vehicle(*v.split("-")) for v in vehicles_str]
            
            # Reset task assignments
            for t in self.tasks:
                t.assigned = False
            
            logging.debug(f"Running PAM at time {self.current_time}")
            assignments = pam_algorithm(self.tasks, self.vehicles, station_id, self.current_time)
            
            logging.debug(f"Running repositioning with algo {algo}")
            repositioning = reposition_vehicles(self.stations, self.vehicles, self.tasks, assignments, algo)
            
            # Store vehicle states and task status
            vehicle_states = {v.id: (v.station, float(v.electricity)) for v in self.vehicles}
            tasks_status = [(t.origin, t.dest, t.fee, t.deadline, t.assigned) for t in self.tasks]
            logging.debug(f"History append: time={self.current_time}, vehicle_states={vehicle_states}")
            
            # Append to history
            self.history.append((self.current_time, assignments, repositioning, vehicle_states, tasks_status))
            
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, f"Time: {self.current_time} min\n\nAssignments:\n")
            if assignments:
                for a in assignments:
                    self.output.insert(tk.END, f"{a}\n")
            else:
                self.output.insert(tk.END, "No assignments\n")
            self.output.insert(tk.END, "\nRepositioning:\n")
            if repositioning:
                for r in repositioning:
                    self.output.insert(tk.END, f"{r}\n")
            else:
                self.output.insert(tk.END, "No repositioning\n")
            
            self.current_time += 5
            self.time_label.config(text=f"Current Time: {self.current_time} min")
        except Exception as e:
            self.error_output.insert(tk.END, f"Error: {str(e)}\n")
            logging.error(f"Run error: {str(e)}")
    
    def visualize(self):
        try:
            visualize_results(self.stations, self.vehicles, self.tasks, self.history)
        except Exception as e:
            self.error_output.insert(tk.END, f"Visualization error: {str(e)}\n")
            logging.error(f"Visualize error: {str(e)}")