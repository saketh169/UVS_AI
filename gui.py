import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from data import Station, Task, Vehicle
from pam import pam_algorithm
from reposition import reposition_vehicles
from visualize import visualize_results
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)

class UVSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("UVS Scheduler - Simulation")
        self.root.geometry("800x600")
        
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TButton", font=("Arial", 12))
        style.configure("TEntry", font=("Arial", 12))
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(main_frame, text="Stations (e.g., M-0-0-5, B-1-1-3):").grid(row=0, column=0, sticky=tk.W)
        self.stations_input = ttk.Entry(main_frame, width=50)
        self.stations_input.grid(row=0, column=1, pady=5)
        
        ttk.Label(main_frame, text="Tasks (e.g., M-B-5-1-10, B-Q-3-2-15):").grid(row=1, column=0, sticky=tk.W)
        self.tasks_input = ttk.Entry(main_frame, width=50)
        self.tasks_input.grid(row=1, column=1, pady=5)
        
        ttk.Label(main_frame, text="Vehicles (e.g., V1-M-80-5, V2-B-50-3):").grid(row=2, column=0, sticky=tk.W)
        self.vehicles_input = ttk.Entry(main_frame, width=50)
        self.vehicles_input.grid(row=2, column=1, pady=5)
        
        ttk.Label(main_frame, text="Station to Process:").grid(row=3, column=0, sticky=tk.W)
        self.station_process = ttk.Entry(main_frame, width=10)
        self.station_process.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(main_frame, text="Algorithm (rdr/rar):").grid(row=4, column=0, sticky=tk.W)
        self.algo_choice = ttk.Entry(main_frame, width=10)
        self.algo_choice.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        ttk.Button(main_frame, text="Run Scheduler", command=self.run).grid(row=5, column=0, pady=10)
        ttk.Button(main_frame, text="Start Simulation", command=self.visualize).grid(row=5, column=1, pady=10, sticky=tk.W)
        
        self.output = scrolledtext.ScrolledText(main_frame, width=60, height=20, font=("Arial", 10))
        self.output.grid(row=6, column=0, columnspan=2, pady=10)
        
        self.stations = []
        self.vehicles = []
        self.tasks = []
        self.assignments = []
        self.repositioning = []
        self.current_time = 0
    
    def validate_inputs(self):
        try:
            # Log raw input
            raw_stations = self.stations_input.get()
            logging.debug(f"Raw stations input: '{raw_stations}'")
            
            # Split and clean inputs
            stations = [s.strip() for s in raw_stations.split(",") if s.strip()]
            tasks = [t.strip() for t in self.tasks_input.get().split(",") if t.strip()]
            vehicles = [v.strip() for v in self.vehicles_input.get().split(",") if v.strip()]
            station_id = self.station_process.get().strip()
            algo = self.algo_choice.get().lower().strip()
            
            logging.debug(f"Parsed stations: {stations}")
            
            # Validate stations (id-x-y-charging_points)
            for s in stations:
                parts = s.split("-")
                logging.debug(f"Station parts: {parts}")
                if len(parts) != 4:
                    raise ValueError("Stations format: id-x-y-charging_points (e.g., M-0-0-5)")
                id_part, x_part, y_part, charge_part = parts
                try:
                    float(x_part)  # x
                    float(y_part)  # y
                    int(charge_part)  # charging_points
                except ValueError:
                    raise ValueError("Stations: x, y, charging_points must be numbers")
            
            # Validate tasks
            if not all(len(t.split("-")) == 5 for t in tasks):
                raise ValueError("Tasks format: origin-dest-fee-deadline-service_time")
            for t in tasks:
                parts = t.split("-")
                try:
                    float(parts[2])  # fee
                    float(parts[3])  # deadline
                    float(parts[4])  # service_time
                except ValueError:
                    raise ValueError("Tasks: fee, deadline, service_time must be numbers")
            
            # Validate vehicles
            if not all(len(v.split("-")) == 4 for v in vehicles):
                raise ValueError("Vehicles format: id-station-electricity-capacity")
            for v in vehicles:
                parts = v.split("-")
                try:
                    float(parts[2])  # electricity
                    int(parts[3])   # capacity
                except ValueError:
                    raise ValueError("Vehicles: electricity must be number, capacity must be integer")
            
            if not station_id:
                raise ValueError("Enter a station ID")
            if algo not in ['rdr', 'rar']:
                raise ValueError("Algorithm must be 'rdr' or 'rar'")
            
            return stations, tasks, vehicles, station_id, algo
        except Exception as e:
            raise ValueError(f"Input error: {str(e)}")
    
    def run(self):
        try:
            stations_str, tasks_str, vehicles_str, station_id, algo = self.validate_inputs()
            
            # Parse stations as id-x-y-charging_points
            self.stations = [
                Station(
                    id=s.split("-")[0],
                    location=(float(s.split("-")[1]), float(s.split("-")[2])),
                    charging_points=int(s.split("-")[3])
                )
                for s in stations_str
            ]
            self.tasks = [Task(*t.split("-")) for t in tasks_str]
            self.vehicles = [Vehicle(*v.split("-")) for v in vehicles_str]
            
            for t in self.tasks:
                t.assigned = False
            
            self.assignments = pam_algorithm(self.tasks, self.vehicles, station_id, self.current_time)
            
            self.repositioning = reposition_vehicles(self.stations, self.vehicles, self.tasks, self.assignments, algo)
            
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, f"Time: {self.current_time} min\n\nAssignments:\n")
            if self.assignments:
                for a in self.assignments:
                    self.output.insert(tk.END, f"{a}\n")
            else:
                self.output.insert(tk.END, "No assignments\n")
            self.output.insert(tk.END, "\nRepositioning:\n")
            if self.repositioning:
                for r in self.repositioning:
                    self.output.insert(tk.END, f"{r}\n")
            else:
                self.output.insert(tk.END, "No repositioning\n")
            
            self.current_time += 5
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def visualize(self):
        try:
            visualize_results(self.stations, self.vehicles, self.tasks, self.assignments, self.repositioning)
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {str(e)}")