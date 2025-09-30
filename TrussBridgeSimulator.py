import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
import pandas as pd
import json
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tkinter import ttk

# -----------------------------
# TRUSS BUILDER
# -----------------------------
def build_truss(n_panels=4, panel_length=1.0, height=0.5):
    joints = {}
    for i in range(n_panels + 1):
        joints[f"B{i}"] = (i * panel_length, 0.0)  # bottom chord
    for i in range(n_panels):
        joints[f"T{i}"] = (i * panel_length + panel_length / 2, height)  # top chord

    members = []
    for i in range(n_panels):
        members.append((f"B{i}", f"B{i+1}"))  # bottom
    for i in range(n_panels - 1):
        members.append((f"T{i}", f"T{i+1}"))  # top
    for i in range(n_panels):
        members.append((f"B{i}", f"T{i}"))    # vertical
        members.append((f"B{i+1}", f"T{i}"))  # diagonal
    return joints, members


# -----------------------------
# SOLVER (forces + reactions)
# -----------------------------
def solve_truss(joints, members, loads, supports):
    n_members = len(members)
    joint_names = list(joints.keys())

    A = []
    b = []
    support_vars = []
    member_vars = list(range(n_members))  # indices for members

    # Add unknowns for supports
    for name, (sx, sy) in supports.items():
        if sx:
            support_vars.append((name, "x"))
        if sy:
            support_vars.append((name, "y"))

    total_unknowns = n_members + len(support_vars)

    for name in joint_names:
        eq_fx = [0] * total_unknowns
        eq_fy = [0] * total_unknowns

        for m, (n1, n2) in enumerate(members):
            if name == n1 or name == n2:
                x1, y1 = joints[n1]
                x2, y2 = joints[n2]
                L = np.hypot(x2 - x1, y2 - y1)
                cx = (x2 - x1) / L
                cy = (y2 - y1) / L
                if name == n1:
                    eq_fx[member_vars[m]] = cx
                    eq_fy[member_vars[m]] = cy
                else:
                    eq_fx[member_vars[m]] = -cx
                    eq_fy[member_vars[m]] = -cy

        for si, (snode, comp) in enumerate(support_vars):
            if name == snode:
                if comp == "x":
                    eq_fx[n_members + si] = 1
                if comp == "y":
                    eq_fy[n_members + si] = 1

        loadx, loady = loads.get(name, (0, 0))

        A.append(eq_fx)
        b.append(-loadx)
        A.append(eq_fy)
        b.append(-loady)

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    forces = dict(zip(members, x[:n_members]))
    reactions = dict(zip(support_vars, x[n_members:]))
    return forces, reactions


# -----------------------------
# GUI CLASS
# -----------------------------
class TrussGUI:
    SAVE_FILE = "truss_data.json"

    def __init__(self, root):
        self.root = root
        self.root.title("Truss Bridge Simulator")

        # Parameters
        self.n_panels = tk.IntVar(value=4)
        self.panel_length = tk.DoubleVar(value=2.0)
        self.height = tk.DoubleVar(value=1.0)

        self.loads = {}  # {joint: (Fx, Fy)}

        # Input fields
        frm = ttk.Frame(root)
        frm.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        ttk.Label(frm, text="Panels:").pack()
        ttk.Entry(frm, textvariable=self.n_panels).pack()

        ttk.Label(frm, text="Panel Length (m):").pack()
        ttk.Entry(frm, textvariable=self.panel_length).pack()

        ttk.Label(frm, text="Height (m):").pack()
        ttk.Entry(frm, textvariable=self.height).pack()

        ttk.Button(frm, text="Build Truss", command=self.build_and_plot).pack(pady=5)
        ttk.Button(frm, text="Clear Loads", command=self.clear_loads).pack(pady=5)
        ttk.Button(frm, text="Save", command=self.save_state).pack(pady=5)
        ttk.Button(frm, text="Load", command=self.load_state).pack(pady=5)
        ttk.Button(frm, text="Export to Excel", command=self.export_excel).pack(pady=5)

        self.info_label = ttk.Label(frm, text="Click a joint to add load")
        self.info_label.pack(pady=5)

        self.support_label = ttk.Label(frm, text="")
        self.support_label.pack(pady=5)

        # Member forces table
        self.tree = ttk.Treeview(frm, columns=("Member", "Force"), show="headings", height=15)
        self.tree.heading("Member", text="Member")
        self.tree.heading("Force", text="Force (N)")
        self.tree.column("Member", anchor="center", width=100)
        self.tree.column("Force", anchor="center", width=120)
        self.tree.pack(fill="both", expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(frm, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Storage
        self.current_forces = {}
        self.current_stats = {}

        # Try loading saved state on startup
        if os.path.exists(self.SAVE_FILE):
            self.load_state()
        else:
            self.build_and_plot()

    def build_and_plot(self):
        n = self.n_panels.get()
        L = self.panel_length.get()
        H = self.height.get()

        self.joints, self.members = build_truss(n, L, H)
        self.supports = {"B0": (True, True), f"B{n}": (False, True)}  # pin + roller
        self.total_length = n * L
        self.update_plot()
        self.save_state()  # autosave

    def clear_loads(self):
        self.loads = {}
        self.update_plot()
        self.save_state()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        # Find nearest joint
        click_point = np.array([event.xdata, event.ydata])
        nearest = min(
            self.joints.items(),
            key=lambda item: np.linalg.norm(click_point - np.array(item[1])),
        )[0]

        if nearest:
            load_value = simpledialog.askfloat("Input Load", f"Enter load (N) for {nearest}:")
            if load_value is not None:
                Fx, Fy = self.loads.get(nearest, (0, 0))
                self.loads[nearest] = (Fx, Fy - load_value)
        self.update_plot()
        self.save_state()

    def update_plot(self):
        self.ax.clear()

        forces, reactions = solve_truss(self.joints, self.members, self.loads, self.supports)
        self.current_forces = forces  # keep for Excel export

        # Heatmap scaling
        max_force = max(abs(f) for f in forces.values()) if forces else 1.0
        norm = mcolors.Normalize(vmin=-max_force, vmax=max_force)
        cmap = cm.get_cmap("plasma")  # red-blue diverging

        # Draw members with heatmap and labels
        for (n1, n2), f in forces.items():
            x1, y1 = self.joints[n1]
            x2, y2 = self.joints[n2]

            # Heatmap color based on normalized force
            color = cmap(norm(f))

            # Draw member line
            self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)

            # Midpoint for label
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
            label = f"{n1}-{n2}\n{f:.2f} N"

            self.ax.text(
                xm, ym, label,
                fontsize=7, ha="center", va="center", color="black",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
            )

        # Draw joints
        for name, (x, y) in self.joints.items():
            self.ax.scatter(x, y, color="k")
            self.ax.text(x, y - 0.05, name, ha="center", fontsize=7)

            # Fixed-size load arrows (NOT scaled by force)
            if name in self.loads:
                self.ax.arrow(x, y, 0, -0.2, head_width=0.05, head_length=0.05, color="g")

        # Title and formatting
        self.ax.set_title(f"Truss Bridge (Length={self.total_length:.1f} m)")
        self.ax.axis("equal")
        self.ax.axis("off")
        self.canvas.draw()

        # Show support reactions
        txt = "Support Reactions:\n"
        for (node, comp), val in reactions.items():
            txt += f"{node} {comp}-dir: {val:.2f} N\n"
        self.support_label.config(text=txt)

        # Update the member table
        self.update_table(forces)


    def update_table(self, forces):
        # Clear previous rows
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Insert member forces
        for (n1, n2), f in forces.items():
            self.tree.insert("", "end", values=(f"{n1}-{n2}", f"{f:.2f}"))

        # Add statistics summary
        if forces:
            max_force = max(forces.values(), key=abs)
            max_tension = max([f for f in forces.values() if f > 0], default=0)
            max_compression = min([f for f in forces.values() if f < 0], default=0)

            self.tree.insert("", "end", values=("--- Stats ---", ""))
            self.tree.insert("", "end", values=("Max Force", f"{max_force:.2f}"))
            self.tree.insert("", "end", values=("Max Tension", f"{max_tension:.2f}"))
            self.tree.insert("", "end", values=("Max Compression", f"{max_compression:.2f}"))


    def save_state(self):
        state = {
            "n_panels": self.n_panels.get(),
            "panel_length": self.panel_length.get(),
            "height": self.height.get(),
            "loads": self.loads,
        }
        with open(self.SAVE_FILE, "w") as f:
            json.dump(state, f, indent=4)

    def load_state(self):
        try:
            with open(self.SAVE_FILE, "r") as f:
                state = json.load(f)
            self.n_panels.set(state["n_panels"])
            self.panel_length.set(state["panel_length"])
            self.height.set(state["height"])
            self.loads = {k: tuple(v) for k, v in state["loads"].items()}
            self.build_and_plot()
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load state: {e}")

    def export_excel(self):
        if not self.current_forces:
            messagebox.showwarning("Export Warning", "No results to export.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")]
        )
        if not filepath:
            return

        # Prepare forces DataFrame
        df = pd.DataFrame(
            [(f"{n1}-{n2}", f) for (n1, n2), f in self.current_forces.items()],
            columns=["Member", "Force (N)"]
        )
        df["Type"] = df["Force (N)"].apply(lambda x: "Tension" if x > 0 else "Compression")

        # Export to Excel
        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Member Forces", index=False)
            stats_df = pd.DataFrame(list(self.current_stats.items()), columns=["Metric", "Value"])
            stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        messagebox.showinfo("Export", f"Data exported to {filepath}")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = TrussGUI(root)
    root.mainloop()
