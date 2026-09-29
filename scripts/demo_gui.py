

import tkinter as tk
from datetime import datetime, timedelta
from tkinter import ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from classical_soh_rul import (
    EOL_SOH_PCT,
    SohReading,
    classify_soh,
    estimate_rul,
    internal_resistance_ohms,
    soh_from_resistance,
)

CLASS_COLORS = {
    "Healthy": "#2e7d32", "Degrading": "#e08e00", "Critical": "#c62828",
    "Long": "#2e7d32", "Medium": "#e08e00", "Short": "#c62828", "Unknown": "#666666",
}

# Example degrading trend, pre-filled so the panel sees a coherent story immediately.
EXAMPLE_HISTORY = [(90, 90.0), (60, 85.0), (30, 80.0), (15, 76.0), (5, 73.0)]


class DemoApp:
    def __init__(self, root):
        self.root = root
        root.title("SmartCell - Classical SOH/RUL Estimator (Demo)")
        self.today_soh = None

        step1 = ttk.LabelFrame(root, text="Step 1 - Today's cranking-test reading", padding=10)
        step1.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.entries = {}
        fields = [
            ("v_rest", "Resting voltage before crank (V)", "12.2"),
            ("v_min", "Minimum voltage during crank (V)", "10.8"),
            ("i_crank", "Cranking current (A)", "40"),
            ("temp", "Ambient temperature (C)", "25"),
        ]
        for i, (key, label, default) in enumerate(fields):
            ttk.Label(step1, text=label).grid(row=i, column=0, sticky="w", pady=3)
            e = ttk.Entry(step1, width=10)
            e.insert(0, default)
            e.grid(row=i, column=1, pady=3, padx=5)
            self.entries[key] = e

        ttk.Button(step1, text="Compute today's SOH", command=self.compute_soh).grid(
            row=len(fields), column=0, columnspan=2, pady=8
        )
        self.soh_result = ttk.Label(step1, text="SOH: -", font=("Segoe UI", 12, "bold"))
        self.soh_result.grid(row=len(fields) + 1, column=0, columnspan=2)
        ttk.Button(step1, text="Add today's SOH to history ->", command=self.add_today_to_history).grid(
            row=len(fields) + 2, column=0, columnspan=2, pady=8
        )

        step2 = ttk.LabelFrame(root, text="Step 2 - Recent SOH history -> RUL", padding=10)
        step2.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ttk.Label(step2, text="Days ago").grid(row=0, column=0)
        ttk.Label(step2, text="SOH %").grid(row=0, column=1)
        self.history_rows = []
        for i in range(8):
            days_e = ttk.Entry(step2, width=8)
            soh_e = ttk.Entry(step2, width=8)
            days_e.grid(row=i + 1, column=0, pady=2, padx=3)
            soh_e.grid(row=i + 1, column=1, pady=2, padx=3)
            if i < len(EXAMPLE_HISTORY):
                days_e.insert(0, str(EXAMPLE_HISTORY[i][0]))
                soh_e.insert(0, str(EXAMPLE_HISTORY[i][1]))
            self.history_rows.append((days_e, soh_e))

        ttk.Button(step2, text="Compute RUL", command=self.compute_rul).grid(
            row=len(self.history_rows) + 1, column=0, columnspan=2, pady=8
        )
        self.rul_result = ttk.Label(step2, text="RUL: -", font=("Segoe UI", 12, "bold"))
        self.rul_result.grid(row=len(self.history_rows) + 2, column=0, columnspan=2)

        plot_frame = ttk.LabelFrame(root, text="SOH trend", padding=10)
        plot_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.fig, self.ax = plt.subplots(figsize=(7, 3.2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_plot([], None)

    def _read_history_rows(self):
        rows = []
        for days_e, soh_e in self.history_rows:
            d, s = days_e.get().strip(), soh_e.get().strip()
            if d and s:
                rows.append((float(d), float(s)))
        return rows

    def compute_soh(self):
        try:
            v_rest = float(self.entries["v_rest"].get())
            v_min = float(self.entries["v_min"].get())
            i_crank = float(self.entries["i_crank"].get())
            temp = float(self.entries["temp"].get())
            r = internal_resistance_ohms(v_rest, v_min, i_crank)
            soh = soh_from_resistance(r, temp)
            cls = classify_soh(soh)
        except (ValueError, ZeroDivisionError) as exc:
            self.soh_result.config(text=f"Input error: {exc}", foreground="#c62828")
            return
        self.today_soh = soh
        self.soh_result.config(
            text=f"SOH: {soh:.1f}%  ->  {cls}", foreground=CLASS_COLORS.get(cls, "#000")
        )

    def add_today_to_history(self):
        if self.today_soh is None:
            self.soh_result.config(text="Compute today's SOH first", foreground="#c62828")
            return
        for days_e, soh_e in self.history_rows:
            if not days_e.get().strip():
                days_e.insert(0, "0")
                soh_e.insert(0, f"{self.today_soh:.1f}")
                return

    def compute_rul(self):
        rows = self._read_history_rows()
        if len(rows) < 2:
            self.rul_result.config(text="Need at least 2 history rows", foreground="#c62828")
            return
        now = datetime.now()
        history = sorted(
            [SohReading(now - timedelta(days=d), s) for d, s in rows], key=lambda r: r.timestamp
        )
        days, cls = estimate_rul(history)
        if days is None:
            self.rul_result.config(text=f"RUL: not forecastable ({cls})", foreground="#666")
        else:
            self.rul_result.config(
                text=f"RUL: {days:.0f} days  ->  {cls}", foreground=CLASS_COLORS.get(cls, "#000")
            )
        self._draw_plot(history, days)

    def _draw_plot(self, history, days_remaining):
        self.ax.clear()
        if history:
            t0 = history[0].timestamp
            xs = [(r.timestamp - t0).total_seconds() / 86400.0 for r in history]
            ys = [r.soh_pct for r in history]
            self.ax.scatter(xs, ys, color="#2e6fa3", zorder=3, label="SOH readings")
            if len(xs) >= 2 and days_remaining is not None:
                slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0]) if xs[-1] != xs[0] else 0
                x_end = xs[-1] + days_remaining
                self.ax.plot(
                    [xs[0], x_end],
                    [ys[0], ys[0] + slope * (x_end - xs[0])],
                    "--", color="#2e6fa3", alpha=0.6, label="trend",
                )
        self.ax.axhline(EOL_SOH_PCT, color="#c62828", linestyle=":", linewidth=1, label="70% EOL")
        self.ax.set_xlabel("Days since first reading")
        self.ax.set_ylabel("SOH (%)")
        self.ax.legend(fontsize=8, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()
