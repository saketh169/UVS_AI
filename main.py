import tkinter as tk
import os
import sys

# Disable OneDNN optimizations for TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import GUI class
from gui import UVSGui

# Debug print (optional)
print("Starting application...")

if __name__ == "__main__":
    root = tk.Tk()
    app = UVSGui(root)
    root.mainloop()
