import tkinter as tk
from gui import UVSGui
from reposition import pre_train_and_save
from data import Station, Task, Vehicle

# Pre-train models (run once)
def run_pre_training():
    stations = [
        Station("M-0-0-5"),
        Station("B-2-2-3"),
        Station("Q-4-4-4")
    ]
    tasks = [
        Task("M-B-6-1-10"),
        Task("B-Q-3-2-15"),
        Task("M-Q-4-1-12"),
        Task("B-M-5-1-15")
    ]
    vehicles = [
        Vehicle("V1-M-85-5"),
        Vehicle("V2-B-60-3"),
        Vehicle("V3-M-70-4"),
        Vehicle("V4-Q-50-2")
    ]
    pre_train_and_save(stations, vehicles, tasks, algo='rdr', episodes=1000)
    pre_train_and_save(stations, vehicles, tasks, algo='rar', episodes=1000)

# Uncomment to pre-train once
# run_pre_training()

if __name__ == "__main__":
    root = tk.Tk()
    app = UVSGui(root)
    root.mainloop()