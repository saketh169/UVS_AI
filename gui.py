import tkinter as tk
from tkinter import scrolledtext
from data import Station, Task, Vehicle
from pam import pam_algorithm
from reposition import reposition_vehicles

class UVSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("UVS Scheduler")

        # Input fields
        tk.Label(root, text="Stations (e.g., A-0-0-5, B-1-1-3):").pack()
        self.stations_input = tk.Entry(root, width=50)
        self.stations_input.pack()

        tk.Label(root, text="Tasks (e.g., A-B-5-1-10, B-C-3-2-15):").pack()
        self.tasks_input = tk.Entry(root, width=50)
        self.tasks_input.pack()

        tk.Label(root, text="Vehicles (e.g., V1-A-80-5, V2-B-50-3):").pack()
        self.vehicles_input = tk.Entry(root, width=50)
        self.vehicles_input.pack()

        tk.Label(root, text="Station to Process:").pack()
        self.station_process = tk.Entry(root, width=10)
        self.station_process.pack()

        tk.Label(root, text="Reposition Algorithm (rdr/rar):").pack()
        self.algo_choice = tk.Entry(root, width=10)
        self.algo_choice.pack()

        # Run button
        tk.Button(root, text="Run Scheduler", command=self.run).pack()

        # Output area
        self.output = scrolledtext.ScrolledText(root, width=60, height=20)
        self.output.pack()

    def run(self):
        # Parse inputs
        stations = [
           Station(
            id=s.split("-")[0],  # This is the id (first part)
            location=(int(s.split("-")[1]), int(s.split("-")[2])),  # This is the location tuple
            charging_points=int(s.split("-")[3])  # This is the charging points (last part)
           )
         for s in self.stations_input.get().split(",")
        ]
        tasks = [Task(*t.split("-")) for t in self.tasks_input.get().split(",")]
        vehicles = [Vehicle(*v.split("-")) for v in self.vehicles_input.get().split(",")]
        station_id = self.station_process.get()
        algo = self.algo_choice.get().lower()

        # Run PAM
        assignments = pam_algorithm(tasks, vehicles, station_id)
        
        # Run RDR or RAR
        repositioning = reposition_vehicles(stations, vehicles, tasks, assignments, algo)

        # Display results
        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, "Assignments:\n")
        for a in assignments:
            self.output.insert(tk.END, f"{a}\n")
        self.output.insert(tk.END, "\nRepositioning:\n")
        for r in repositioning:
            self.output.insert(tk.END, f"{r}\n")