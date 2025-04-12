import tkinter as tk
from tkinter import scrolledtext
from data import Station, Task, Vehicle
from pam import pam_algorithm
from reposition import reposition_vehicles

class UVSGui:
    def __init__(self, root):
        self.root = root
        self.root.title("UVS Scheduler")

        # Stations input
        tk.Label(root, text="Stations (e.g., A-0-0-5, B-1-1-3):").pack()
        self.stations_input = tk.Text(root, height=3, width=60)
        self.stations_input.pack()

        # Tasks input
        tk.Label(root, text="Tasks (e.g., A-B-5-1-10, B-C-3-2-15):").pack()
        self.tasks_input = tk.Text(root, height=4, width=60)
        self.tasks_input.pack()

        # Vehicles input
        tk.Label(root, text="Vehicles (e.g., V1-A-80-5, V2-B-50-3):").pack()
        self.vehicles_input = tk.Text(root, height=3, width=60)
        self.vehicles_input.pack()

        # Station to process
        tk.Label(root, text="Station to Process:").pack()
        self.station_process = tk.Entry(root, width=10)
        self.station_process.pack()

        # Reposition algorithm
        tk.Label(root, text="Reposition Algorithm (rdr/rar):").pack()
        self.algo_choice = tk.Entry(root, width=10)
        self.algo_choice.pack()

        # Buttons frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        # Buttons
        tk.Button(button_frame, text="Run Scheduler", command=self.run).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear Input", command=self.clear_input).pack(side=tk.LEFT, padx=5)

        # Output
        self.output = scrolledtext.ScrolledText(root, width=80, height=20)
        self.output.pack(pady=10)

    def run(self):
        try:
            stations_text = self.stations_input.get("1.0", tk.END).strip()
            tasks_text = self.tasks_input.get("1.0", tk.END).strip()
            vehicles_text = self.vehicles_input.get("1.0", tk.END).strip()

            stations = [
                Station(
                    id=s.split("-")[0],
                    location=(int(s.split("-")[1]), int(s.split("-")[2])),
                    charging_points=int(s.split("-")[3])
                )
                for s in stations_text.split(",") if s.strip()
            ]

            tasks = [Task(*t.split("-"), stations=stations) for t in tasks_text.split(",") if t.strip()]
            vehicles = [Vehicle(*v.split("-"), stations=stations) for v in vehicles_text.split(",") if v.strip()]
            station_id = self.station_process.get().strip()
            algo = self.algo_choice.get().lower().strip()

            if not stations or not tasks or not vehicles or not station_id or not algo:
                raise ValueError("All fields must be filled with valid data.")

            assignments = pam_algorithm(tasks, vehicles, station_id)
            repositioning = reposition_vehicles(stations, vehicles, tasks, assignments, algo)

            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, "Assignments:\n")
            for a in assignments:
                self.output.insert(tk.END, f"{a}\n")
            self.output.insert(tk.END, "\nRepositioning:\n")
            for r in repositioning:
                self.output.insert(tk.END, f"{r}\n")
        except Exception as e:
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, f"Error: {str(e)}")

    def clear_input(self):
        # Clear all inputs and output
        self.stations_input.delete("1.0", tk.END)
        self.tasks_input.delete("1.0", tk.END)
        self.vehicles_input.delete("1.0", tk.END)
        self.station_process.delete(0, tk.END)
        self.algo_choice.delete(0, tk.END)
        self.output.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = UVSGui(root)
    root.mainloop()
