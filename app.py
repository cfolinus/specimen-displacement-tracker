"""
Dot Tracker GUI — batch-process Instron tensile test videos.

Launch:  python app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import csv
import threading
import queue
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tracker_core import (VideoTracker, annotate_frame,
                          extract_initial_distance_mm)

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output_data"
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

FRAME_SKIP_OPTIONS = {"Every frame": 1, "Every 2nd frame": 2, "Every 4th frame": 4}


# ── Messages from worker thread to UI ────────────────────────────────────────
class MsgProgress:
    def __init__(self, vid_idx, frame_idx, total_frames, frame_bgr):
        self.vid_idx = vid_idx
        self.frame_idx = frame_idx
        self.total_frames = total_frames
        self.frame_bgr = frame_bgr  # may be None (not every frame sent)

class MsgDone:
    def __init__(self, vid_idx, tracker):
        self.vid_idx = vid_idx
        self.tracker = tracker  # finished VideoTracker with results/positions

class MsgError:
    def __init__(self, vid_idx, error_msg):
        self.vid_idx = vid_idx
        self.error_msg = error_msg

class MsgAllDone:
    pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Instron Dot Tracker")
        self.geometry("1280x900")
        self.minsize(1000, 720)

        # State
        self.video_files = []           # list of Path
        self.trackers = {}              # {idx: finished VideoTracker}
        self.cleaned_data = {}          # {idx: {'results': [...], 'positions': [...]}}
        self.processing = False
        self.stop_requested = False
        self.msg_queue = queue.Queue()
        self.worker_thread = None
        self._photo = None
        self._current_processing_idx = -1
        self._current_processing_frame = None  # latest BGR from worker

        # Review state
        self._review_cap = None         # cv2.VideoCapture for scrubbing
        self._review_idx = None         # which video index we're reviewing
        self._results_idx = None        # last video whose results are shown in Data/Plot tabs
        self._playing = False
        self._play_after_id = None
        self._scrub_blocked = False

        # Tracking option flags (set before running)
        self.track_pixel_pos    = tk.BooleanVar(value=True)
        self.track_mm_pos       = tk.BooleanVar(value=True)
        self.track_dot_disp     = tk.BooleanVar(value=True)
        self.track_interdot_disp = tk.BooleanVar(value=True)
        self.track_interdot_dist = tk.BooleanVar(value=True)

        # Plot axis selection, style, and title
        self.plot_x_var     = tk.StringVar(value='Time (s)')
        self.plot_y_var     = tk.StringVar(value='')
        self.plot_title_var = tk.StringVar(value='')
        self.plot_style_var = tk.StringVar(value='Line')
        self._plot_suspend  = False  # block combobox callbacks during programmatic set

        self._build_ui()
        self._poll_queue()

    def destroy(self):
        self._playing = False
        self.stop_requested = True
        if self._review_cap is not None:
            self._review_cap.release()
        super().destroy()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # ── Toolbar row 1: file controls + run ──────────────────────────
        toolbar = ttk.Frame(self, padding=(6, 4))
        toolbar.pack(fill="x")

        ttk.Button(toolbar, text="Add Videos...", command=self._add_videos).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Add from input_videos/", command=self._add_from_default).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Clear List", command=self._clear_list).pack(side="left", padx=3)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(toolbar, text="Frame skip:").pack(side="left")
        self.skip_var = tk.StringVar(value="Every frame")
        ttk.Combobox(toolbar, textvariable=self.skip_var,
                     values=list(FRAME_SKIP_OPTIONS.keys()),
                     state="readonly", width=16).pack(side="left", padx=3)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)

        self.run_btn = ttk.Button(toolbar, text="Run All", command=self._run_all)
        self.run_btn.pack(side="left", padx=3)
        self.stop_btn = ttk.Button(toolbar, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=3)

        # ── Toolbar row 2: output variable selection ─────────────────────
        options_bar = ttk.Frame(self, padding=(6, 2))
        options_bar.pack(fill="x")

        ttk.Label(options_bar, text="Output variables:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(2, 6))

        checks = [
            ("Pixel position (x,y)",        self.track_pixel_pos),
            ("Scaled position mm (x,y)",     self.track_mm_pos),
            ("Per-dot displacement",         self.track_dot_disp),
            ("Displacement between dots",    self.track_interdot_disp),
            ("Distance between dots (mm)",   self.track_interdot_dist),
        ]
        for label, var in checks:
            ttk.Checkbutton(options_bar, text=label, variable=var).pack(side="left", padx=6)

        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=6)

        # ── Main paned layout ────────────────────────────────────────────
        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=(4, 6))

        # Left: video list
        left = ttk.Frame(pane, width=280)
        pane.add(left, weight=0)

        ttk.Label(left, text="Videos", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=4, pady=(4, 2))
        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_frame, selectmode="browse", font=("Consolas", 10))
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # Right: tabs
        right = ttk.Frame(pane)
        pane.add(right, weight=1)

        # Status bar
        status_frame = ttk.Frame(right)
        status_frame.pack(fill="x", padx=4, pady=4)
        self.status_label = ttk.Label(status_frame, text="Ready", font=("Segoe UI", 10))
        self.status_label.pack(side="left")
        self.progress_bar = ttk.Progressbar(status_frame, mode="determinate", length=300)
        self.progress_bar.pack(side="right")

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Video preview + controls
        self.video_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.video_tab, text="  Video Preview  ")

        # Pack bottom controls FIRST so they get space before the canvas fills the rest
        self.video_info_label = ttk.Label(self.video_tab, text="", font=("Consolas", 9))
        self.video_info_label.pack(side="bottom", fill="x", padx=4)

        ctrl_frame = ttk.Frame(self.video_tab)
        ctrl_frame.pack(side="bottom", fill="x", padx=4, pady=4)

        self.play_btn = ttk.Button(ctrl_frame, text="Play", width=6, command=self._toggle_play)
        self.play_btn.pack(side="left", padx=2)

        self.scrub_var = tk.IntVar(value=0)
        self.scrub_scale = ttk.Scale(ctrl_frame, from_=0, to=100,
                                     orient="horizontal", variable=self.scrub_var,
                                     command=self._on_scrub)
        self.scrub_scale.pack(side="left", fill="x", expand=True, padx=6)

        self.time_label = ttk.Label(ctrl_frame, text="0.0s / 0.0s", width=18, anchor="center")
        self.time_label.pack(side="right")

        # Canvas fills remaining space
        self.canvas_label = ttk.Label(self.video_tab, anchor="center")
        self.canvas_label.pack(fill="both", expand=True)

        # Tab 2: Data table
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="  Data Table  ")

        tree_frame = ttk.Frame(self.data_tab)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.tree = ttk.Treeview(tree_frame, show="headings", height=25)
        self.tree["columns"] = ("time", "displacement")
        self.tree.heading("time", text="Time (s)")
        self.tree.heading("displacement", text="Displacement")
        self.tree.column("time", width=100, anchor="center")
        self.tree.column("displacement", width=140, anchor="center")

        tree_sb_y = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        tree_sb_x = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_sb_y.set, xscrollcommand=tree_sb_x.set)
        tree_sb_y.pack(side="right", fill="y")
        tree_sb_x.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)

        export_frame = ttk.Frame(self.data_tab)
        export_frame.pack(fill="x", padx=4, pady=4)
        ttk.Button(export_frame, text="Export CSV...", command=self._export_csv).pack(side="right")
        ttk.Button(export_frame, text="Export Cleaned CSV...", command=self._export_cleaned_csv).pack(side="right", padx=4)
        ttk.Button(export_frame, text="Clean Outliers", command=self._clean_outliers).pack(side="left", padx=4)
        self.clean_label = ttk.Label(export_frame, text="", font=("Segoe UI", 9))
        self.clean_label.pack(side="left", padx=4)

        # Tab 3: Plot
        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_tab, text="  Plot  ")

        # Control row: X/Y dropdowns + export buttons
        plot_ctrl = ttk.Frame(self.plot_tab, padding=(4, 4))
        plot_ctrl.pack(fill="x")

        ttk.Label(plot_ctrl, text="X:").pack(side="left", padx=(4, 2))
        self.plot_x_combo = ttk.Combobox(
            plot_ctrl, textvariable=self.plot_x_var,
            state="readonly", width=28, values=[])
        self.plot_x_combo.pack(side="left", padx=(0, 12))
        self.plot_x_combo.bind("<<ComboboxSelected>>", self._on_plot_axis_change)

        ttk.Label(plot_ctrl, text="Y:").pack(side="left", padx=(0, 2))
        self.plot_y_combo = ttk.Combobox(
            plot_ctrl, textvariable=self.plot_y_var,
            state="readonly", width=28, values=[])
        self.plot_y_combo.pack(side="left", padx=(0, 12))
        self.plot_y_combo.bind("<<ComboboxSelected>>", self._on_plot_axis_change)

        ttk.Combobox(plot_ctrl, textvariable=self.plot_style_var,
                     state="readonly", width=10,
                     values=["Line", "Scatter"]).pack(side="left", padx=(0, 8))
        self.plot_style_var.trace_add("write", lambda *_: self._on_plot_axis_change())

        ttk.Button(plot_ctrl, text="↺ Refresh",
                   command=self._on_plot_axis_change).pack(side="left", padx=(0, 8))

        ttk.Button(plot_ctrl, text="Copy",
                   command=self._copy_plot).pack(side="right", padx=2)
        ttk.Button(plot_ctrl, text="Save as...",
                   command=self._save_plot).pack(side="right", padx=2)

        # Title row
        title_row = ttk.Frame(self.plot_tab, padding=(4, 0, 4, 2))
        title_row.pack(fill="x")
        ttk.Label(title_row, text="Title:").pack(side="left", padx=(4, 4))
        self.plot_title_entry = ttk.Entry(title_row, textvariable=self.plot_title_var)
        self.plot_title_entry.pack(side="left", fill="x", expand=True)
        self.plot_title_entry.bind("<Return>",    self._on_plot_axis_change)
        self.plot_title_entry.bind("<FocusOut>",  self._on_plot_axis_change)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout()
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_tab)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

    # --------------------------------------------------------- File management
    def _add_videos(self):
        files = filedialog.askopenfilenames(
            title="Select video files", initialdir=str(INPUT_DIR),
            filetypes=[("Video files", "*.mov *.mp4 *.avi *.mkv *.wmv"), ("All", "*.*")])
        for f in files:
            p = Path(f)
            if p not in self.video_files:
                self.video_files.append(p)
                self.listbox.insert("end", p.name)

    def _add_from_default(self):
        exts = {'.mov', '.mp4', '.avi', '.mkv', '.wmv'}
        found = sorted(f for f in INPUT_DIR.iterdir() if f.suffix.lower() in exts)
        if not found:
            messagebox.showinfo("Info", f"No video files in:\n{INPUT_DIR}")
            return
        for p in found:
            if p not in self.video_files:
                self.video_files.append(p)
                self.listbox.insert("end", p.name)

    def _clear_list(self):
        if self.processing:
            return
        self.video_files.clear()
        self.listbox.delete(0, "end")
        self.trackers.clear()
        self.cleaned_data.clear()
        self._close_review()

    # --------------------------------------------------------- Tracking options
    def _get_track_opts(self):
        return dict(
            track_pixel_pos     = self.track_pixel_pos.get(),
            track_mm_pos        = self.track_mm_pos.get(),
            track_dot_disp      = self.track_dot_disp.get(),
            track_interdot_disp = self.track_interdot_disp.get(),
            track_interdot_dist = self.track_interdot_dist.get(),
        )

    # --------------------------------------------------------- List selection / review
    def _on_list_select(self, _event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]

        if idx in self.trackers:
            # Completed video — open for review
            self._open_review(idx)
            self._show_results_tabs(idx)
        elif idx == self._current_processing_idx and self._current_processing_frame is not None:
            # Currently processing — show latest frame
            self._show_frame(self._current_processing_frame)
            self.notebook.select(self.video_tab)

    def _open_review(self, idx):
        self._playing = False
        self._close_review()
        self._review_idx = idx
        tracker = self.trackers[idx]
        path = self.video_files[idx]

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Cannot open video for review: {path}")
            return
        self._review_cap = cap
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Temporarily block scrub callback while reconfiguring
        self._scrub_blocked = True
        self.scrub_scale.configure(to=max(total - 1, 1))
        self.scrub_var.set(0)
        self._scrub_blocked = False

        self.video_info_label.configure(text=tracker.info_text)
        self.play_btn.configure(text="Play")
        self.notebook.select(self.video_tab)

        # Show first frame
        self._show_review_frame(0)

    def _close_review(self):
        self._playing = False
        if self._play_after_id is not None:
            self.after_cancel(self._play_after_id)
            self._play_after_id = None
        if self._review_cap is not None:
            self._review_cap.release()
            self._review_cap = None
        self._review_idx = None

    def _show_review_frame(self, frame_idx):
        """Read a specific frame and overlay tracked positions."""
        if self._review_cap is None or self._review_idx is None:
            return
        tracker = self.trackers.get(self._review_idx)
        if tracker is None:
            return

        self._review_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._review_cap.read()
        if not ret:
            return

        # Find the closest tracked position for this frame
        dots, dist_val = [], None
        if tracker.frame_indices:
            fi = np.array(tracker.frame_indices)
            closest = np.searchsorted(fi, frame_idx, side='right') - 1
            closest = max(0, min(closest, len(tracker.positions) - 1))
            dots = tracker.positions[closest]
            dist_val = tracker.results[closest][1]

        annotated = annotate_frame(frame, dots, dist_val, tracker.unit)
        self._show_frame(annotated)

        fps = self._review_cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(self._review_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        t_cur = frame_idx / fps
        t_total = total / fps
        self.time_label.configure(text=f"{t_cur:.1f}s / {t_total:.1f}s")

    def _on_scrub(self, val):
        if getattr(self, '_scrub_blocked', False):
            return
        if self._review_cap is None:
            return
        frame_idx = int(float(val))
        self._show_review_frame(frame_idx)

    def _toggle_play(self):
        if self._review_cap is None:
            sel = self.listbox.curselection()
            if sel and sel[0] in self.trackers:
                self._open_review(sel[0])
            return
        self._playing = not self._playing
        self.play_btn.configure(text="Pause" if self._playing else "Play")
        if self._playing:
            self._play_step()

    def _play_step(self):
        if not self._playing or self._review_cap is None:
            return
        fps = self._review_cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(self._review_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cur = self.scrub_var.get() + 1
        if cur >= total:
            self._playing = False
            self.play_btn.configure(text="Play")
            return
        self._scrub_blocked = True
        self.scrub_var.set(cur)
        self._scrub_blocked = False
        self._show_review_frame(cur)
        delay = max(1, int(1000 / fps))
        self._play_after_id = self.after(delay, self._play_step)

    def _show_results_tabs(self, idx):
        # Only reset the plot title when switching to a different video
        if idx != self._results_idx:
            self.plot_title_var.set(self.video_files[idx].stem)
        self._results_idx = idx
        has_cleaned = idx in self.cleaned_data
        self._show_data_table(idx, cleaned=has_cleaned)
        self._show_plot(idx, cleaned=has_cleaned)
        if has_cleaned:
            raw_n = len(self.trackers[idx].results)
            clean_n = len(self.cleaned_data[idx]['results'])
            removed = raw_n - clean_n
            self.clean_label.configure(
                text=f"Removed {removed} outliers ({removed/raw_n*100:.1f}%)")
        else:
            self.clean_label.configure(text="")

    # --------------------------------------------------------- Processing
    def _run_all(self):
        if not self.video_files:
            messagebox.showinfo("Info", "Add some videos first.")
            return
        if not any([self.track_pixel_pos.get(), self.track_mm_pos.get(),
                    self.track_dot_disp.get(), self.track_interdot_disp.get(),
                    self.track_interdot_dist.get()]):
            messagebox.showinfo("Info", "Select at least one output variable.")
            return

        self.processing = True
        self.stop_requested = False
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._close_review()

        skip = FRAME_SKIP_OPTIONS[self.skip_var.get()]
        opts = self._get_track_opts()
        indices = list(range(len(self.video_files)))

        self.worker_thread = threading.Thread(
            target=self._worker, args=(indices, skip, opts), daemon=True)
        self.worker_thread.start()

    def _stop(self):
        self.stop_requested = True
        self.stop_btn.configure(state="disabled")

    def _worker(self, indices, skip, opts):
        """Runs in background thread — processes videos as fast as possible."""
        for vid_idx in indices:
            if self.stop_requested:
                break

            path = self.video_files[vid_idx]
            init_dist = extract_initial_distance_mm(path.stem)
            tracker = VideoTracker(path, frame_skip=skip, initial_distance_mm=init_dist)

            first = tracker.open()
            if first is None:
                self.msg_queue.put(MsgError(vid_idx, tracker.error or "Unknown error"))
                continue

            self.msg_queue.put(MsgProgress(vid_idx, 0, tracker.total_frames, first))

            frame_count = 0
            while not tracker.finished and not self.stop_requested:
                frame = tracker.step()
                frame_count += 1
                send_frame = frame if (frame_count % 20 == 0) else None
                self.msg_queue.put(MsgProgress(
                    vid_idx, tracker.current_frame_idx, tracker.total_frames, send_frame))

            if self.stop_requested:
                tracker.release()
                break

            tracker.release()

            # Auto-save CSV with selected output variables
            csv_path = OUTPUT_DIR / (path.stem + ".csv")
            tracker.save_csv(csv_path, **opts)

            self.msg_queue.put(MsgDone(vid_idx, tracker))

        self.msg_queue.put(MsgAllDone())

    def _poll_queue(self):
        """Drain the message queue from the worker thread (runs on UI thread)."""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if isinstance(msg, MsgProgress):
                    self._handle_progress(msg)
                elif isinstance(msg, MsgDone):
                    self._handle_done(msg)
                elif isinstance(msg, MsgError):
                    self._handle_error(msg)
                elif isinstance(msg, MsgAllDone):
                    self._handle_all_done()
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    def _handle_progress(self, msg):
        self._current_processing_idx = msg.vid_idx
        if msg.total_frames > 0:
            self.progress_bar["value"] = msg.frame_idx / msg.total_frames * 100

        n_total = len(self.video_files)
        self.status_label.configure(
            text=f"Processing [{msg.vid_idx + 1}/{n_total}] {self.video_files[msg.vid_idx].name}  "
                 f"({msg.frame_idx}/{msg.total_frames})")

        if msg.frame_bgr is not None:
            self._current_processing_frame = msg.frame_bgr
            sel = self.listbox.curselection()
            viewing_current = (not sel) or (sel and sel[0] == msg.vid_idx)
            if viewing_current and self._review_idx is None:
                self._show_frame(msg.frame_bgr)
                self.notebook.select(self.video_tab)

    def _handle_done(self, msg):
        self.trackers[msg.vid_idx] = msg.tracker
        name = self.video_files[msg.vid_idx].name
        self.listbox.delete(msg.vid_idx)
        self.listbox.insert(msg.vid_idx, f"\u2713  {name}")
        self.progress_bar["value"] = 100

    def _handle_error(self, msg):
        name = self.video_files[msg.vid_idx].name
        self.listbox.delete(msg.vid_idx)
        self.listbox.insert(msg.vid_idx, f"\u2717  {name}")
        print(f"ERROR [{name}]: {msg.error_msg}")

    def _handle_all_done(self):
        self.processing = False
        self.stop_requested = False
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._current_processing_idx = -1
        n = len(self.trackers)
        self.status_label.configure(text=f"Done — {n} video(s) processed. CSVs saved to output_data/")
        self.progress_bar["value"] = 0

    # --------------------------------------------------------- Display
    def _show_frame(self, bgr_frame):
        self.canvas_label.update_idletasks()
        cw = max(self.canvas_label.winfo_width(), 200)
        ch = max(self.canvas_label.winfo_height(), 200)
        fh, fw = bgr_frame.shape[:2]
        scale = min(cw / fw, ch / fh, 1.0)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas_label.configure(image=self._photo)

    # --------------------------------------------------------- Outlier cleaning
    def _get_selected_tracker(self):
        sel = self.listbox.curselection()
        if not sel:
            return None, None
        idx = sel[0]
        return idx, self.trackers.get(idx)

    def _clean_outliers(self):
        idx, tracker = self._get_selected_tracker()
        if tracker is None:
            messagebox.showinfo("Info", "Select a completed video first.")
            return

        times = np.array([r[0] for r in tracker.results])
        n_orig = len(times)

        # Decide what signal to clean on: inter-dot distance for 2+ dots,
        # else dot1's y-coordinate (the primary motion axis).
        if tracker.n_dots >= 2 and all(r[1] is not None for r in tracker.results):
            signal = np.array([r[1] for r in tracker.results])
        else:
            signal = np.array([
                tracker.positions[i][0][1] if i < len(tracker.positions)
                                              and tracker.positions[i][0] is not None
                else np.nan
                for i in range(n_orig)
            ])

        _, _, kept_indices = clean_data(times, signal)
        n_removed = n_orig - len(kept_indices)

        self.cleaned_data[idx] = {
            'results':   [tracker.results[i] for i in kept_indices],
            'positions': [tracker.positions[i] for i in kept_indices],
        }

        removed_pct = n_removed / n_orig * 100
        self.clean_label.configure(
            text=f"Removed {n_removed} outliers ({removed_pct:.1f}%)")

        self._results_idx = idx
        self._show_data_table(idx, cleaned=True)
        self._show_plot(idx, cleaned=True)

    # --------------------------------------------------------- Data table
    def _show_data_table(self, idx, cleaned=False):
        tracker = self.trackers.get(idx)
        if tracker is None:
            return

        opts = self._get_track_opts()
        h    = tracker.height
        ppm  = tracker.px_per_mm
        unit = tracker.unit
        n    = tracker.n_dots

        # ── Build dynamic column list per dot ────────────────────────────
        col_ids  = ['time']
        col_hdrs = ['Time (s)']
        col_wids = [90]

        def add_col(label, w=110):
            col_ids.append(label)
            col_hdrs.append(label)
            col_wids.append(w)

        if opts['track_pixel_pos']:
            for i in range(n):
                add_col(f'Dot{i+1} X (px)', 100)
                add_col(f'Dot{i+1} Y (px)', 100)
        if opts['track_mm_pos']:
            for i in range(n):
                add_col(f'Dot{i+1} X (mm)')
                add_col(f'Dot{i+1} Y (mm)')
        if opts['track_dot_disp']:
            for i in range(n):
                add_col(f'Dot{i+1} dX ({unit})')
                add_col(f'Dot{i+1} dY ({unit})')
        if opts['track_interdot_disp'] and n >= 2:
            add_col(f'Displacement ({unit})', 130)
        if opts['track_interdot_dist'] and n >= 2:
            add_col(f'Distance ({unit})', 120)

        self.tree['columns'] = col_ids
        self.tree['show'] = 'headings'
        for cid, hdr, wid in zip(col_ids, col_hdrs, col_wids):
            self.tree.heading(cid, text=hdr)
            self.tree.column(cid, width=wid, anchor='center', minwidth=70)

        # ── Choose data source ───────────────────────────────────────────
        if cleaned and idx in self.cleaned_data:
            cd = self.cleaned_data[idx]
            data      = cd['results']
            positions = cd['positions']
        else:
            data      = tracker.results
            positions = tracker.positions

        if not data:
            self.tree.delete(*self.tree.get_children())
            return

        d0 = data[0][1] if data[0][1] is not None else None
        ref_positions = positions[0] if positions else [None] * n

        # ── Fill rows ────────────────────────────────────────────────────
        self.tree.delete(*self.tree.get_children())
        for i, (t, d) in enumerate(data):
            pts = positions[i] if i < len(positions) else [None] * n
            if len(pts) < n:
                pts = list(pts) + [None] * (n - len(pts))
            row = [f'{t:.4f}']

            # Pixel position
            if opts['track_pixel_pos']:
                for j in range(n):
                    p = pts[j]
                    if p is not None:
                        row += [f'{p[0]:.1f}', f'{h - p[1]:.1f}']
                    else:
                        row += ['', '']

            # mm position
            if opts['track_mm_pos']:
                for j in range(n):
                    p = pts[j]
                    if p is not None and ppm:
                        row += [f'{p[0]/ppm:.3f}', f'{(h - p[1])/ppm:.3f}']
                    else:
                        row += ['', '']

            # Per-dot displacement
            if opts['track_dot_disp']:
                for j in range(n):
                    p = pts[j]
                    p0 = ref_positions[j] if j < len(ref_positions) else None
                    if p is not None and p0 is not None:
                        dx = p[0] - p0[0]
                        dy = (h - p[1]) - (h - p0[1])
                        if ppm:
                            row += [f'{dx/ppm:.4f}', f'{dy/ppm:.4f}']
                        else:
                            row += [f'{dx:.2f}', f'{dy:.2f}']
                    else:
                        row += ['', '']

            # Inter-dot displacement / distance
            if opts['track_interdot_disp'] and n >= 2:
                if d is not None and d0 is not None:
                    row.append(f'{d - d0:.4f}')
                else:
                    row.append('')
            if opts['track_interdot_dist'] and n >= 2:
                row.append(f'{d:.4f}' if d is not None else '')

            self.tree.insert('', 'end', values=row)

    # --------------------------------------------------------- Plot
    def _variable_labels(self, tracker):
        """List of selectable plot variables available for this tracker."""
        labels = ['Time (s)']
        n = tracker.n_dots
        unit = tracker.unit
        has_mm = tracker.px_per_mm is not None

        for i in range(1, n + 1):
            labels += [f'Dot{i} X (px)', f'Dot{i} Y (px)']
        if has_mm:
            for i in range(1, n + 1):
                labels += [f'Dot{i} X (mm)', f'Dot{i} Y (mm)']
        for i in range(1, n + 1):
            labels += [f'Dot{i} dX ({unit})', f'Dot{i} dY ({unit})']
        if n >= 2:
            labels.append(f'Inter-dot displacement ({unit})')
            labels.append(f'Inter-dot distance ({unit})')
        return labels

    def _default_y_label(self, tracker):
        """Sensible default Y variable for this tracker."""
        unit = tracker.unit
        if tracker.n_dots >= 2:
            return f'Inter-dot displacement ({unit})'
        return f'Dot1 dY ({unit})'

    def _compute_var(self, tracker, label, results, positions):
        """Return a numpy array of values for `label` over the given rows."""
        h = tracker.height
        ppm = tracker.px_per_mm
        n = tracker.n_dots
        ref = (tracker.positions[0]
               if tracker.positions else [None] * n)

        if label == 'Time (s)':
            return np.array([r[0] for r in results], dtype=float)

        if label.startswith('Inter-dot displacement'):
            d0 = (results[0][1] if (results and results[0][1] is not None)
                  else 0.0)
            return np.array(
                [r[1] - d0 if r[1] is not None else np.nan for r in results],
                dtype=float)
        if label.startswith('Inter-dot distance'):
            return np.array(
                [r[1] if r[1] is not None else np.nan for r in results],
                dtype=float)

        # Dot-indexed: 'Dot1 X (px)', 'Dot2 dY (mm)', etc.
        if label.startswith('Dot'):
            space = label.index(' ')
            i = int(label[3:space]) - 1
            rest = label[space + 1:]
            ref_p = (ref[i] if (0 <= i < len(ref) and ref[i] is not None)
                     else None)

            out = np.full(len(positions), np.nan, dtype=float)
            for k, pts in enumerate(positions):
                p = pts[i] if (pts and i < len(pts) and pts[i] is not None) \
                    else None
                if p is None:
                    continue
                if rest == 'X (px)':
                    out[k] = p[0]
                elif rest == 'Y (px)':
                    out[k] = h - p[1]
                elif rest == 'X (mm)' and ppm:
                    out[k] = p[0] / ppm
                elif rest == 'Y (mm)' and ppm:
                    out[k] = (h - p[1]) / ppm
                elif rest.startswith('dX') and ref_p is not None:
                    v = p[0] - ref_p[0]
                    out[k] = v / ppm if ('(mm)' in rest and ppm) else v
                elif rest.startswith('dY') and ref_p is not None:
                    v = (h - p[1]) - (h - ref_p[1])
                    out[k] = v / ppm if ('(mm)' in rest and ppm) else v
            return out

        return np.full(len(positions), np.nan, dtype=float)

    def _refresh_plot_controls(self, idx):
        """Populate X/Y dropdown values for the selected video."""
        tracker = self.trackers.get(idx)
        if tracker is None:
            return
        labels = self._variable_labels(tracker)
        default_y = self._default_y_label(tracker)
        if default_y not in labels:
            default_y = labels[1] if len(labels) > 1 else labels[0]

        self._plot_suspend = True
        self.plot_x_combo['values'] = labels
        self.plot_y_combo['values'] = labels
        if self.plot_x_var.get() not in labels:
            self.plot_x_var.set('Time (s)')
        if self.plot_y_var.get() not in labels:
            self.plot_y_var.set(default_y)
        self._plot_suspend = False

    def _on_plot_axis_change(self, _event=None):
        if self._plot_suspend:
            return
        # Use the last video whose results were displayed — the listbox may
        # have lost selection when the combobox was clicked.
        idx = self._results_idx
        if idx is None or idx not in self.trackers:
            return
        self._show_plot(idx, cleaned=(idx in self.cleaned_data))

    def _show_plot(self, idx, cleaned=False):
        tracker = self.trackers.get(idx)
        if tracker is None:
            return

        # Make sure dropdown values are valid for this tracker
        self._refresh_plot_controls(idx)

        x_label = self.plot_x_var.get()
        y_label = self.plot_y_var.get()

        self.ax.clear()

        style = self.plot_style_var.get()  # "Line" or "Scatter"

        def draw(x, y, color, label=None, zorder=1):
            kw = dict(color=color, zorder=zorder,
                      label=label if label else "_nolegend_")
            if style == "Scatter":
                self.ax.scatter(x, y, s=6, **kw)
            else:
                self.ax.plot(x, y, linewidth=1.2, **kw)

        raw_x = self._compute_var(tracker, x_label,
                                  tracker.results, tracker.positions)
        raw_y = self._compute_var(tracker, y_label,
                                  tracker.results, tracker.positions)

        if cleaned and idx in self.cleaned_data:
            cd = self.cleaned_data[idx]
            cx = self._compute_var(tracker, x_label, cd['results'], cd['positions'])
            cy = self._compute_var(tracker, y_label, cd['results'], cd['positions'])
            if style == "Line":
                self.ax.plot(raw_x, raw_y, linewidth=0.8, color="#cccccc",
                             label="Raw", zorder=1)
            else:
                self.ax.scatter(raw_x, raw_y, s=4, color="#cccccc",
                                label="Raw", zorder=1)
            draw(cx, cy, "#2563eb", label="Cleaned", zorder=2)
            self.ax.legend(loc="best", fontsize=9)
        else:
            draw(raw_x, raw_y, "#2563eb")

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(self.plot_title_var.get())
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.plot_canvas.draw()

    # --------------------------------------------------------- Plot export
    def _plot_filename_stem(self):
        title = self.plot_title_var.get().strip() or "plot"
        # sanitise for filename
        for ch in r'/\:*?"<>|':
            title = title.replace(ch, '_')
        return title

    def _save_plot(self):
        idx, _ = self._get_selected_tracker()
        if idx is None:
            messagebox.showinfo("Info", "Select a completed video first.")
            return
        stem = self._plot_filename_stem()
        path = filedialog.asksaveasfilename(
            title="Save plot image",
            initialdir=str(OUTPUT_DIR),
            initialfile=f"{stem}.png",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"),
                       ("PDF", "*.pdf"),
                       ("SVG", "*.svg"),
                       ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        messagebox.showinfo("Saved", f"Plot saved to:\n{path}")

    def _copy_plot(self):
        """Copy current plot to the system clipboard as an image."""
        idx, _ = self._get_selected_tracker()
        if idx is None:
            messagebox.showinfo("Info", "Select a completed video first.")
            return

        # Render figure to PNG bytes
        buf = BytesIO()
        try:
            self.fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Copy failed", str(e))
            return
        buf.seek(0)

        # Windows: copy CF_DIB to clipboard via pywin32 if available
        try:
            import win32clipboard
            img = Image.open(buf).convert("RGB")
            out = BytesIO()
            img.save(out, "BMP")
            dib = out.getvalue()[14:]  # strip 14-byte BMP file header
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, dib)
            finally:
                win32clipboard.CloseClipboard()
            return
        except ImportError:
            pass
        except Exception as e:
            messagebox.showerror("Copy failed", str(e))
            return

        # Fallback: write a temp PNG and tell the user where it is
        tmp_dir = OUTPUT_DIR / "_clipboard_tmp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / f"{self._plot_filename_stem()}.png"
        with open(tmp_path, "wb") as f:
            f.write(buf.getvalue())
        messagebox.showinfo(
            "Copy unavailable",
            "Clipboard image copy requires pywin32 on Windows.\n"
            "Install with:  pip install pywin32\n\n"
            f"For now the image was saved to:\n{tmp_path}")

    # --------------------------------------------------------- Export
    def _write_csv(self, path, results, positions, tracker, opts):
        """Write selected tracking variables to a CSV file."""
        tracker.save_csv(path, results=results, positions=positions, **opts)

    def _export_csv(self):
        idx, tracker = self._get_selected_tracker()
        if tracker is None:
            messagebox.showinfo("Info", "Select a completed video first.")
            return
        stem = self.video_files[idx].stem
        path = filedialog.asksaveasfilename(
            title="Save CSV", initialdir=str(OUTPUT_DIR),
            initialfile=f"{stem}.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        self._write_csv(path, tracker.results, tracker.positions, tracker, self._get_track_opts())
        messagebox.showinfo("Saved", f"CSV saved to:\n{path}")

    def _export_cleaned_csv(self):
        idx, tracker = self._get_selected_tracker()
        if tracker is None:
            messagebox.showinfo("Info", "Select a completed video first.")
            return
        if idx not in self.cleaned_data:
            messagebox.showinfo("Info", "Run 'Clean Outliers' first.")
            return
        stem = self.video_files[idx].stem
        path = filedialog.asksaveasfilename(
            title="Save Cleaned CSV", initialdir=str(OUTPUT_DIR),
            initialfile=f"{stem}_cleaned.csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        cd = self.cleaned_data[idx]
        self._write_csv(path, cd['results'], cd['positions'], tracker, self._get_track_opts())
        messagebox.showinfo("Saved", f"Cleaned CSV saved to:\n{path}")


# ── Outlier cleaning ──────────────────────────────────────────────────────────
def clean_data(times, dists):
    """
    Remove outlier points from tensile test distance data.

    Returns (clean_times, clean_dists, kept_indices) where kept_indices
    are the original integer indices of the rows that were retained.
    """
    if len(dists) < 10:
        return times, dists, np.arange(len(dists))

    all_idx = np.arange(len(dists))

    # Pass 1: Remove velocity outliers (sudden jumps)
    dt = np.diff(times)
    dd = np.diff(dists)
    dt[dt == 0] = 1e-6
    velocity = dd / dt

    window = min(51, len(velocity) // 4 * 2 + 1)
    if window < 3:
        window = 3
    half_w = window // 2
    keep = np.ones(len(dists), dtype=bool)

    for i in range(len(velocity)):
        lo = max(0, i - half_w)
        hi = min(len(velocity), i + half_w + 1)
        local_v = velocity[lo:hi]
        med = np.median(local_v)
        mad = np.median(np.abs(local_v - med))
        mad = max(mad, 1e-6)
        if abs(velocity[i] - med) > 5 * mad:
            keep[i + 1] = False

    times_1  = times[keep]
    dists_1  = dists[keep]
    idx_1    = all_idx[keep]

    if len(dists_1) < 10:
        return times_1, dists_1, idx_1

    # Pass 2: Remove position outliers from moving median
    window2 = min(31, len(dists_1) // 4 * 2 + 1)
    if window2 < 3:
        window2 = 3
    half_w2 = window2 // 2

    keep2 = np.ones(len(dists_1), dtype=bool)
    for i in range(len(dists_1)):
        lo = max(0, i - half_w2)
        hi = min(len(dists_1), i + half_w2 + 1)
        local = dists_1[lo:hi]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        mad = max(mad, 1e-6)
        if abs(dists_1[i] - med) > 5 * mad:
            keep2[i] = False

    return times_1[keep2], dists_1[keep2], idx_1[keep2]


if __name__ == "__main__":
    app = App()
    app.mainloop()
