import tkinter as tk
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from gui import UVSGui
print("dont")

if __name__ == "__main__":
    root = tk.Tk()
    app = UVSGui(root)
    root.mainloop()