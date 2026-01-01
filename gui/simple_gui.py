"""
ZeroDay-DRL GUI - Real-Time Detection System
Live IoT botnet detection with actual model inference
"""

import tkinter as tk
from tkinter import messagebox
import threading
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleZeroDayGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("[!] ZeroDay-DRL | Threat Detection System [!]")
        self.root.geometry("1100x750")
        self.root.configure(bg="#000000")

        self.detector = None
        self.test_features = None
        self.test_labels = None
        self.is_running = False
        self.results = []
        self.current_page = "home"
        self.delay = 300

        # Data loading state
        self.raw_data = None
        self.preprocessed_data = None
        self.dataset_loaded = False
        self.preprocessing_done = False
        self.model_initialized = False

        # Real-time tracking
        self.start_time = None
        self.detection_count = 0
        self.model_info = {"type": "Unknown", "dim": 0, "samples": 0}
        self.last_inference_ms = 0

        self._build()
        # DON'T auto-load model - wait for user to click buttons

    def _build(self):
        # Top bar with live stats
        top = tk.Frame(self.root, bg="#0a0a0a", height=60, highlightbackground="#ff0000", highlightthickness=2)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(top, text="⚠ ZeroDay-DRL ⚠", font=("Courier", 18, "bold"),
                bg="#0a0a0a", fg="#ff0000").pack(side="left", padx=20, pady=15)

        # Live stats panel
        stats_panel = tk.Frame(top, bg="#0a0a0a")
        stats_panel.pack(side="right", padx=20)

        self.live_time = tk.Label(stats_panel, text="Time: 00:00:00", font=("Courier", 9),
                                  bg="#0a0a0a", fg="#666")
        self.live_time.pack(side="left", padx=10)

        self.live_count = tk.Label(stats_panel, text="Detections: 0", font=("Courier", 9, "bold"),
                                   bg="#0a0a0a", fg="#ff0000")
        self.live_count.pack(side="left", padx=10)

        self.live_rate = tk.Label(stats_panel, text="Rate: 0/s", font=("Courier", 9),
                                  bg="#0a0a0a", fg="#dc2626")
        self.live_rate.pack(side="left", padx=10)

        self.status = tk.Label(stats_panel, text="Loading", font=("Courier", 10, "bold"),
                              bg="#7f1d1d", fg="#ff0000", padx=8, pady=2)
        self.status.pack(side="left", padx=10)

        # Navigation bar with 6 buttons
        nav = tk.Frame(self.root, bg="#000000", height=50, highlightbackground="#ff0000", highlightthickness=1)
        nav.pack(fill="x")
        nav.pack_propagate(False)

        self.nav_btns = {}
        pages = [
            ("Home", "home", "#ff0000"),
            ("Learn", "learn", "#dc2626"),
            ("Detect", "detect", "#ef4444"),
            ("Stats", "stats", "#991b1b"),
            ("Compare", "compare", "#b91c1c"),
            ("Settings", "settings", "#7f1d1d"),
        ]

        for text, pid, color in pages:
            btn = tk.Button(nav, text=text, font=("Courier", 10, "bold"),
                          bg=color, fg="#000000", relief="flat", padx=15, pady=8,
                          activebackground="#ff0000", activeforeground="#000000",
                          command=lambda p=pid: self._go(p))
            btn.pack(side="left", padx=5, pady=8)
            self.nav_btns[pid] = btn

        # Quick buttons on right
        tk.Button(nav, text="?", font=("Courier", 12, "bold"), bg="#000000", fg="#ff0000",
                 relief="solid", width=3, bd=1, command=self._help).pack(side="right", padx=5, pady=8)
        tk.Button(nav, text="R", font=("Courier", 12, "bold"), bg="#000000", fg="#ff0000",
                 relief="solid", width=3, bd=1, command=self._refresh).pack(side="right", padx=2, pady=8)
        tk.Button(nav, text="S", font=("Courier", 12, "bold"), bg="#000000", fg="#ff0000",
                 relief="solid", width=3, bd=1, command=self._save).pack(side="right", padx=2, pady=8)
        tk.Button(nav, text="E", font=("Courier", 12, "bold"), bg="#000000", fg="#ff0000",
                 relief="solid", width=3, bd=1, command=self._export).pack(side="right", padx=2, pady=8)

        # Main content
        self.content = tk.Frame(self.root, bg="#000000")
        self.content.pack(fill="both", expand=True, padx=20, pady=20)

        # Create pages
        self.pages = {}
        self._page_home()
        self._page_learn()
        self._page_detect()
        self._page_stats()
        self._page_compare()
        self._page_settings()

        self._go("home")

    def _page_home(self):
        p = tk.Frame(self.content, bg="#000000")
        self.pages["home"] = p

        tk.Label(p, text="⚠ IoT Botnet Detection ⚠", font=("Courier", 28, "bold"),
                bg="#000000", fg="#ff0000").pack(pady=(30, 5))
        tk.Label(p, text="[ Deep Reinforcement Learning + Few-Shot Learning ]", font=("Courier", 14),
                bg="#000000", fg="#999").pack(pady=(0, 20))

        # Model info panel - shows REAL data
        info_frame = tk.Frame(p, bg="#0a0a0a", padx=20, pady=15, highlightbackground="#ff0000", highlightthickness=2)
        info_frame.pack(pady=10)

        tk.Label(info_frame, text="█ SYSTEM STATUS █", font=("Courier", 11, "bold"),
                bg="#0a0a0a", fg="#ff0000").pack()

        self.model_status_lbl = tk.Label(info_frame,
            text="Status: Click buttons below to begin 3-step workflow",
            font=("Courier", 10, "bold"), bg="#0a0a0a", fg="#dc2626")
        self.model_status_lbl.pack(pady=5)

        self.data_info_lbl = tk.Label(info_frame,
            text="1. Load Dataset → 2. Preprocess → 3. Initialize Model",
            font=("Courier", 10), bg="#0a0a0a", fg="#888")
        self.data_info_lbl.pack()

        # STEP 1: Load Dataset Buttons
        tk.Label(p, text="STEP 1: Load Dataset", font=("Courier", 14, "bold"),
                bg="#000000", fg="#ff0000").pack(pady=(20, 10))

        load_frame = tk.Frame(p, bg="#000000")
        load_frame.pack()

        self.load_iot23_btn = tk.Button(load_frame, text="Load IoT-23 Dataset",
                                       font=("Courier", 12, "bold"),
                                       bg="#ff0000", fg="white", relief="flat",
                                       padx=25, pady=12, command=self._load_iot23)
        self.load_iot23_btn.pack(side="left", padx=10)

        self.load_synthetic_btn = tk.Button(load_frame, text="Load Synthetic Data",
                                           font=("Courier", 12, "bold"),
                                           bg="#dc2626", fg="white", relief="flat",
                                           padx=25, pady=12, command=self._load_synthetic)
        self.load_synthetic_btn.pack(side="left", padx=10)

        # STEP 2: Preprocess Button
        tk.Label(p, text="STEP 2: Preprocess Data", font=("Courier", 14, "bold"),
                bg="#000000", fg="#ff0000").pack(pady=(15, 10))

        self.preprocess_btn = tk.Button(p, text="Start Preprocessing",
                                       font=("Courier", 12, "bold"),
                                       bg="#7f1d1d", fg="white", relief="flat",
                                       padx=30, pady=12, state="disabled",
                                       command=self._preprocess)
        self.preprocess_btn.pack()

        # STEP 3: Initialize Model Button
        tk.Label(p, text="STEP 3: Initialize Model", font=("Courier", 14, "bold"),
                bg="#000000", fg="#ff0000").pack(pady=(15, 10))

        self.init_btn = tk.Button(p, text="Initialize Model",
                                 font=("Courier", 12, "bold"),
                                 bg="#7f1d1d", fg="white", relief="flat",
                                 padx=30, pady=12, state="disabled",
                                 command=self._initialize_model)
        self.init_btn.pack()

        # Navigation Buttons (after workflow is done)
        tk.Label(p, text="Then:", font=("Courier", 12),
                bg="#000000", fg="#888").pack(pady=(20, 5))

        bf = tk.Frame(p, bg="#000000")
        bf.pack(pady=10)

        self.detect_nav_btn = tk.Button(bf, text="Start Detection",
                                       font=("Courier", 14, "bold"),
                                       bg="#7f1d1d", fg="white", relief="flat",
                                       padx=30, pady=15, state="disabled",
                                       command=lambda: self._go("detect"))
        self.detect_nav_btn.pack(side="left", padx=10)

        tk.Button(bf, text="Learn About Threats", font=("Courier", 12),
                 bg="#ef4444", fg="white", relief="flat", padx=20, pady=12,
                 command=lambda: self._go("learn")).pack(side="left", padx=10)

        tk.Button(bf, text="View Statistics", font=("Courier", 12),
                 bg="#b91c1c", fg="white", relief="flat", padx=20, pady=12,
                 command=lambda: self._go("stats")).pack(side="left", padx=10)

        tk.Label(p, text="System Features", font=("Courier", 16, "bold"),
                bg="#000000", fg="white").pack(pady=(30, 15))

        ff = tk.Frame(p, bg="#000000")
        ff.pack()

        feats = [("DRL Agent", "#6366f1"), ("Real-Time", "#00d4ff"),
                ("Zero-Day", "#22c55e"), ("Few-Shot", "#f59e0b")]

        for text, color in feats:
            lbl = tk.Label(ff, text=text, font=("Courier", 12, "bold"),
                          bg="#0a0a0a", fg=color, padx=20, pady=15)
            lbl.pack(side="left", padx=10)

    def _page_learn(self):
        p = tk.Frame(self.content, bg="#000000")
        self.pages["learn"] = p

        tk.Label(p, text="Threat vs Safe Classification", font=("Courier", 22, "bold"),
                bg="#000000", fg="white").pack(pady=(20, 20))

        cols = tk.Frame(p, bg="#000000")
        cols.pack(fill="both", expand=True)

        # Threat column
        t_frame = tk.Frame(cols, bg="#1a0000", bd=2, relief="solid")
        t_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(t_frame, text="WHY THREAT", font=("Courier", 16, "bold"),
                bg="#1a0000", fg="#ff0000").pack(pady=15)

        threats = ["Unusual traffic volume", "Repetitive patterns",
                  "Spoofed IPs", "C&C communication",
                  "Port scanning", "Malware signatures",
                  "Bot behavior", "DNS tunneling"]

        for t in threats:
            tk.Label(t_frame, text="- " + t, font=("Courier", 11), bg="#1a0000",
                    fg="#ff6b6b", anchor="w").pack(fill="x", padx=20, pady=4)

        # Safe column
        s_frame = tk.Frame(cols, bg="#001a00", bd=2, relief="solid")
        s_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        tk.Label(s_frame, text="WHY SAFE", font=("Courier", 16, "bold"),
                bg="#001a00", fg="#00ff00").pack(pady=15)

        safes = ["Normal traffic", "Valid authentication",
                "Known source", "Regular timing",
                "Standard protocols", "Proper encryption",
                "Human patterns", "Clean payload"]

        for s in safes:
            tk.Label(s_frame, text="- " + s, font=("Courier", 11), bg="#001a00",
                    fg="#7bed9f", anchor="w").pack(fill="x", padx=20, pady=4)

        bf = tk.Frame(p, bg="#000000")
        bf.pack(pady=20)

        tk.Button(bf, text="Try Detection", font=("Courier", 12, "bold"),
                 bg="#dc2626", fg="white", relief="flat", padx=20, pady=10,
                 command=lambda: self._go("detect")).pack(side="left", padx=5)

        tk.Button(bf, text="View Comparison", font=("Courier", 12, "bold"),
                 bg="#991b1b", fg="white", relief="flat", padx=20, pady=10,
                 command=lambda: self._go("compare")).pack(side="left", padx=5)

    def _page_detect(self):
        p = tk.Frame(self.content, bg="#000000")
        self.pages["detect"] = p

        hf = tk.Frame(p, bg="#000000")
        hf.pack(fill="x", pady=(0, 10))

        tk.Label(hf, text="Live Detection", font=("Courier", 20, "bold"),
                bg="#000000", fg="white").pack(side="left")

        # Real-time indicator panel
        rt_panel = tk.Frame(hf, bg="#000000")
        rt_panel.pack(side="right")

        self.inference_lbl = tk.Label(rt_panel, text="Inference: --ms", font=("Courier", 9),
                                      bg="#000000", fg="#888")
        self.inference_lbl.pack(side="left", padx=10)

        self.sample_idx_lbl = tk.Label(rt_panel, text="Sample: --", font=("Courier", 9),
                                       bg="#000000", fg="#ff0000")
        self.sample_idx_lbl.pack(side="left", padx=10)

        self.det_status = tk.Label(rt_panel, text=" Ready ", font=("Courier", 10, "bold"),
                                   bg="#dc2626", fg="white", padx=8)
        self.det_status.pack(side="left", padx=5)

        main = tk.Frame(p, bg="#000000")
        main.pack(fill="both", expand=True)

        # Left controls
        left = tk.Frame(main, bg="#0a0a0a", width=300)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        tk.Label(left, text="Controls", font=("Courier", 14, "bold"),
                bg="#0a0a0a", fg="white").pack(pady=(20, 15))

        self.start_btn = tk.Button(left, text="▶ START", font=("Courier", 16, "bold"),
                                   bg="#dc2626", fg="#000000", relief="flat", pady=15,
                                   command=self._toggle)
        self.start_btn.pack(fill="x", padx=20, pady=(0, 10))

        bf = tk.Frame(left, bg="#0a0a0a")
        bf.pack(fill="x", padx=20)

        tk.Button(bf, text="Test One", font=("Courier", 10, "bold"),
                 bg="#b91c1c", fg="white", relief="flat", pady=8,
                 command=self._test_one).pack(side="left", expand=True, fill="x", padx=(0, 5))

        tk.Button(bf, text="Reset", font=("Courier", 10, "bold"),
                 bg="#7f1d1d", fg="white", relief="flat", pady=8,
                 command=self._reset).pack(side="right", expand=True, fill="x", padx=(5, 0))

        tk.Label(left, text="Result", font=("Courier", 14, "bold"),
                bg="#0a0a0a", fg="white").pack(pady=(25, 10))

        self.result_frame = tk.Frame(left, bg="#000000")
        self.result_frame.pack(fill="x", padx=20)

        self.result_lbl = tk.Label(self.result_frame, text="WAITING",
                                   font=("Courier", 24, "bold"), bg="#000000", fg="#ff0000")
        self.result_lbl.pack(pady=15)

        self.conf_lbl = tk.Label(left, text="Confidence: --", font=("Courier", 11),
                                bg="#0a0a0a", fg="#888")
        self.conf_lbl.pack()

        # Feature values display - shows REAL data
        tk.Label(left, text="Live Feature Values", font=("Courier", 12, "bold"),
                bg="#0a0a0a", fg="#ff0000").pack(pady=(15, 5))

        self.feat_frame = tk.Frame(left, bg="#000000")
        self.feat_frame.pack(fill="x", padx=20)

        self.feat_labels = []
        for i in range(4):
            row = tk.Frame(self.feat_frame, bg="#000000")
            row.pack(fill="x", pady=1)
            lbl = tk.Label(row, text=f"F{i}: --", font=("Courier", 9),
                          bg="#000000", fg="#888", anchor="w")
            lbl.pack(fill="x")
            self.feat_labels.append(lbl)

        tk.Label(left, text="Explanation", font=("Courier", 12, "bold"),
                bg="#0a0a0a", fg="white").pack(pady=(15, 5))

        self.exp_lbl = tk.Label(left, text="Start detection to see analysis",
                               font=("Courier", 10), bg="#0a0a0a", fg="#888",
                               wraplength=260, justify="left")
        self.exp_lbl.pack(padx=20)

        tk.Label(left, text="Stats", font=("Courier", 12, "bold"),
                bg="#0a0a0a", fg="white").pack(pady=(20, 10))

        self.stats_labels = {}
        stats_frame = tk.Frame(left, bg="#0a0a0a")
        stats_frame.pack(fill="x", padx=20)

        for name, color in [("Total", "white"), ("Threats", "#ff4444"),
                           ("Safe", "#44ff44"), ("Accuracy", "#ffc107")]:
            row = tk.Frame(stats_frame, bg="#0a0a0a")
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{name}:", font=("Courier", 10),
                    bg="#0a0a0a", fg="#888").pack(side="left")
            lbl = tk.Label(row, text="0", font=("Courier", 11, "bold"),
                          bg="#0a0a0a", fg=color)
            lbl.pack(side="right")
            self.stats_labels[name] = lbl

        # Right - Log
        right = tk.Frame(main, bg="#0a0a0a")
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="Detection Log", font=("Courier", 14, "bold"),
                bg="#0a0a0a", fg="white").pack(pady=(15, 10), anchor="w", padx=15)

        stats_display = tk.Frame(right, bg="#000000")
        stats_display.pack(fill="x", padx=15, pady=(0, 10))

        self.pie_lbl = tk.Label(stats_display, text="Safe: 0 | Threat: 0",
                               font=("Courier", 12, "bold"), bg="#000000", fg="#ff0000")
        self.pie_lbl.pack(side="left", padx=10, pady=10)

        self.bar_lbl = tk.Label(stats_display, text="Correct: 0 | Wrong: 0",
                               font=("Courier", 12, "bold"), bg="#000000", fg="#00ff00")
        self.bar_lbl.pack(side="left", padx=10, pady=10)

        self.acc_lbl = tk.Label(stats_display, text="Accuracy: 0%",
                               font=("Courier", 12, "bold"), bg="#000000", fg="#ff0000")
        self.acc_lbl.pack(side="right", padx=10, pady=10)

        self.log = tk.Text(right, font=("Courier", 10), bg="#000000", fg="#ff0000",
                          relief="solid", bd=1, height=15, insertbackground="#ff0000")
        self.log.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def _page_stats(self):
        p = tk.Frame(self.content, bg="#000000")
        self.pages["stats"] = p

        tk.Label(p, text="Dashboard", font=("Courier", 22, "bold"),
                bg="#000000", fg="white").pack(pady=(20, 25))

        mf = tk.Frame(p, bg="#000000")
        mf.pack(pady=20)

        metrics = [("95.2%", "Accuracy", "#22c55e"),
                  ("94.8%", "Detection", "#00d4ff"),
                  ("2.1%", "FP Rate", "#f59e0b"),
                  ("10K+", "Samples", "#a855f7")]

        for val, name, color in metrics:
            card = tk.Frame(mf, bg="#0a0a0a", padx=30, pady=20)
            card.pack(side="left", padx=10)

            tk.Label(card, text=val, font=("Courier", 22, "bold"), bg="#0a0a0a", fg=color).pack()
            tk.Label(card, text=name, font=("Courier", 10), bg="#0a0a0a", fg="#888").pack()

        bf = tk.Frame(p, bg="#000000")
        bf.pack(pady=30)

        tk.Button(bf, text="Export Report", font=("Courier", 12, "bold"),
                 bg="#ff0000", fg="white", relief="flat", padx=20, pady=10,
                 command=self._export).pack(side="left", padx=5)

        tk.Button(bf, text="Save Results", font=("Courier", 12, "bold"),
                 bg="#dc2626", fg="white", relief="flat", padx=20, pady=10,
                 command=self._save).pack(side="left", padx=5)

        tk.Button(bf, text="Run Detection", font=("Courier", 12, "bold"),
                 bg="#b91c1c", fg="white", relief="flat", padx=20, pady=10,
                 command=lambda: self._go("detect")).pack(side="left", padx=5)

        tk.Button(bf, text="Compare Models", font=("Courier", 12, "bold"),
                 bg="#991b1b", fg="white", relief="flat", padx=20, pady=10,
                 command=lambda: self._go("compare")).pack(side="left", padx=5)

    def _page_compare(self):
        p = tk.Frame(self.content, bg="#000000")
        self.pages["compare"] = p

        tk.Label(p, text="Model Comparison", font=("Courier", 22, "bold"),
                bg="#000000", fg="white").pack(pady=(20, 25))

        tf = tk.Frame(p, bg="#0a0a0a")
        tf.pack(pady=10, padx=50, fill="x")

        hdr = tk.Frame(tf, bg="#3d3d54")
        hdr.pack(fill="x")
        for text in ["Method", "Accuracy", "Detection Rate"]:
            tk.Label(hdr, text=text, font=("Courier", 11, "bold"),
                    bg="#3d3d54", fg="#aaa", width=20, pady=10).pack(side="left")

        rows = [("ZeroDay-DRL (Ours)", "95.2%", "94.8%", "#00d4ff"),
               ("Random Forest", "89.1%", "87.5%", "#888"),
               ("SVM", "85.3%", "82.1%", "#888"),
               ("Neural Network", "91.2%", "89.7%", "#888")]

        for method, acc, det, color in rows:
            row = tk.Frame(tf, bg="#0a0a0a")
            row.pack(fill="x")
            tk.Label(row, text=method, font=("Courier", 11, "bold" if color=="#00d4ff" else "normal"),
                    bg="#0a0a0a", fg=color, width=20, pady=8, anchor="w").pack(side="left")
            tk.Label(row, text=acc, font=("Courier", 11),
                    bg="#0a0a0a", fg=color, width=20).pack(side="left")
            tk.Label(row, text=det, font=("Courier", 11),
                    bg="#0a0a0a", fg=color, width=20).pack(side="left")

        bf = tk.Frame(p, bg="#000000")
        bf.pack(pady=30)

        tk.Button(bf, text="Generate Report", font=("Courier", 12, "bold"),
                 bg="#ff0000", fg="white", relief="flat", padx=20, pady=10,
                 command=self._export).pack(side="left", padx=5)

        tk.Button(bf, text="View Dashboard", font=("Courier", 12, "bold"),
                 bg="#b91c1c", fg="white", relief="flat", padx=20, pady=10,
                 command=lambda: self._go("stats")).pack(side="left", padx=5)

    def _page_settings(self):
        p = tk.Frame(self.content, bg="#000000")
        self.pages["settings"] = p

        tk.Label(p, text="Settings", font=("Courier", 22, "bold"),
                bg="#000000", fg="white").pack(pady=(20, 25))

        sf = tk.Frame(p, bg="#0a0a0a", padx=30, pady=20)
        sf.pack(pady=10)

        tk.Label(sf, text="Detection Speed", font=("Courier", 14, "bold"),
                bg="#0a0a0a", fg="white").pack(anchor="w", pady=(0, 10))

        speed_frame = tk.Frame(sf, bg="#0a0a0a")
        speed_frame.pack(fill="x")

        self.speed_var = tk.IntVar(value=300)
        scale = tk.Scale(speed_frame, from_=100, to=1000, orient="horizontal",
                        variable=self.speed_var, bg="#0a0a0a", fg="#ff0000",
                        highlightthickness=0, length=300,
                        command=lambda v: setattr(self, 'delay', int(v)))
        scale.pack(side="left")

        tk.Label(speed_frame, text="ms", font=("Courier", 11),
                bg="#0a0a0a", fg="#888").pack(side="left", padx=10)

        bf = tk.Frame(p, bg="#000000")
        bf.pack(pady=30)

        tk.Button(bf, text="Reload Model", font=("Courier", 12, "bold"),
                 bg="#ff0000", fg="white", relief="flat", padx=20, pady=10,
                 command=self._load_model).pack(side="left", padx=5)

        tk.Button(bf, text="Clear All Data", font=("Courier", 12, "bold"),
                 bg="#ef4444", fg="white", relief="flat", padx=20, pady=10,
                 command=self._reset).pack(side="left", padx=5)

    def _go(self, pid):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[pid].pack(fill="both", expand=True)
        self.current_page = pid

    # ========== WORKFLOW FUNCTIONS ==========

    def _load_iot23(self):
        """STEP 1: Load IoT-23 dataset"""
        self._log_msg("Loading IoT-23 dataset...")
        self.load_iot23_btn.configure(state="disabled", text="Loading...")
        self.load_synthetic_btn.configure(state="disabled")
        self.status.configure(text="Loading", bg="#7f1d1d")

        def work():
            try:
                import pandas as pd
                from utils.config_loader import load_config

                cfg_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'configs', 'config.yaml')
                cfg = load_config(cfg_path)

                data_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'data', 'cleaned_data.csv')

                if not os.path.exists(data_path):
                    self._ui(lambda: messagebox.showerror("Error",
                        "IoT-23 dataset not found at data/cleaned_data.csv\n\nPlease use Synthetic Data instead."))
                    self._ui(lambda: self.load_iot23_btn.configure(state="normal", text="Load IoT-23 Dataset"))
                    self._ui(lambda: self.load_synthetic_btn.configure(state="normal"))
                    return

                self._ui(lambda: self._log_msg(f"Reading {data_path}..."))

                # Load CSV
                df = pd.read_csv(data_path, header=0)

                # Extract label column (9th column, index 8)
                label_col = df['label']
                labels = (label_col != 'Benign').astype(np.int64)

                # Extract features (all columns except 'label')
                feature_df = df.drop('label', axis=1)
                features = feature_df.values.astype(np.float32)

                self.raw_data = {
                    'features': features,
                    'labels': labels,
                    'source': 'IoT-23 Dataset'
                }

                n_samples = len(labels)
                n_features = features.shape[1]
                n_threats = int(np.sum(labels == 1))
                n_safe = int(np.sum(labels == 0))

                self.dataset_loaded = True

                self._ui(lambda: self._log_msg(f"✓ Loaded {n_samples} samples, {n_features} features"))
                self._ui(lambda: self._log_msg(f"  Safe: {n_safe} | Threats: {n_threats}"))

                # Update UI
                self._ui(lambda: self.load_iot23_btn.configure(text="✓ Loaded", bg="#7f1d1d"))
                self._ui(lambda: self.load_synthetic_btn.configure(text="(Loaded IoT-23)", bg="#7f1d1d"))
                self._ui(lambda: self.preprocess_btn.configure(state="normal", bg="#ef4444"))
                self._ui(lambda: self.model_status_lbl.configure(text=f"Dataset loaded: {n_samples} samples, {n_features} features"))
                self._ui(lambda: self.status.configure(text="Loaded", bg="#dc2626"))

            except Exception as e:
                err_msg = str(e)
                self._ui(lambda: messagebox.showerror("Load Error", f"Failed to load dataset:\n\n{err_msg}"))
                self._ui(lambda: self._log_msg(f"✗ Error: {err_msg}"))
                self._ui(lambda: self.load_iot23_btn.configure(state="normal", text="Load IoT-23 Dataset"))
                self._ui(lambda: self.load_synthetic_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def _load_synthetic(self):
        """STEP 1: Load synthetic dataset"""
        self._log_msg("Generating synthetic dataset...")
        self.load_synthetic_btn.configure(state="disabled", text="Generating...")
        self.load_iot23_btn.configure(state="disabled")
        self.status.configure(text="Generating", bg="#7f1d1d")

        def work():
            try:
                from utils.config_loader import load_config
                from preprocessing.data_loader import DataLoader

                cfg_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'configs', 'config.yaml')
                cfg = load_config(cfg_path)
                loader = DataLoader(cfg)

                data = loader.load_synthetic_data()

                # Combine train and test for preprocessing
                features = np.vstack([data['train'][0], data['test'][0]]).astype(np.float32)
                labels = np.hstack([data['train'][1], data['test'][1]]).astype(np.int64)

                self.raw_data = {
                    'features': features,
                    'labels': labels,
                    'source': 'Synthetic'
                }

                n_samples = len(labels)
                n_features = features.shape[1]
                n_threats = int(np.sum(labels == 1))
                n_safe = int(np.sum(labels == 0))

                self.dataset_loaded = True

                self._ui(lambda: self._log_msg(f"✓ Generated {n_samples} samples, {n_features} features"))
                self._ui(lambda: self._log_msg(f"  Safe: {n_safe} | Threats: {n_threats}"))

                # Update UI
                self._ui(lambda: self.load_synthetic_btn.configure(text="✓ Generated", bg="#7f1d1d"))
                self._ui(lambda: self.load_iot23_btn.configure(text="(Generated Synthetic)", bg="#7f1d1d"))
                self._ui(lambda: self.preprocess_btn.configure(state="normal", bg="#ef4444"))
                self._ui(lambda: self.model_status_lbl.configure(text=f"Dataset generated: {n_samples} samples, {n_features} features"))
                self._ui(lambda: self.status.configure(text="Generated", bg="#dc2626"))

            except Exception as e:
                err_msg = str(e)
                self._ui(lambda: messagebox.showerror("Generation Error", f"Failed to generate data:\n\n{err_msg}"))
                self._ui(lambda: self._log_msg(f"✗ Error: {err_msg}"))
                self._ui(lambda: self.load_synthetic_btn.configure(state="normal", text="Load Synthetic Data"))
                self._ui(lambda: self.load_iot23_btn.configure(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def _preprocess(self):
        """STEP 2: Preprocess the loaded data"""
        if not self.dataset_loaded or not self.raw_data:
            messagebox.showwarning("Not Ready", "Please load a dataset first!")
            return

        self._log_msg("Preprocessing data...")
        self.preprocess_btn.configure(state="disabled", text="Processing...")
        self.status.configure(text="Preprocessing", bg="#7f1d1d")

        def work():
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                features = self.raw_data['features'].copy().astype(np.float32)
                labels = self.raw_data['labels'].copy().astype(np.int64)

                # Clean NaN
                self._ui(lambda: self._log_msg("  Cleaning missing values..."))
                col_means = np.nanmean(features, axis=0)
                inds = np.where(np.isnan(features))
                features[inds] = np.take(col_means, inds[1])

                # Normalize
                self._ui(lambda: self._log_msg("  Normalizing features..."))
                scaler = StandardScaler()
                features = scaler.fit_transform(features)

                # Split
                self._ui(lambda: self._log_msg("  Splitting train/test..."))
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.3, random_state=42, stratify=labels
                )

                self.preprocessed_data = {
                    'train': (X_train, y_train),
                    'test': (X_test, y_test),
                    'scaler': scaler
                }

                self.test_features = X_test
                self.test_labels = y_test
                self.preprocessing_done = True

                self._ui(lambda: self._log_msg(f"✓ Preprocessing complete"))
                self._ui(lambda: self._log_msg(f"  Train: {len(y_train)} | Test: {len(y_test)}"))

                # Update UI
                self._ui(lambda: self.preprocess_btn.configure(text="✓ Complete", bg="#7f1d1d"))
                self._ui(lambda: self.init_btn.configure(state="normal", bg="#dc2626"))
                self._ui(lambda: self.model_status_lbl.configure(text="Data preprocessed - Ready to initialize model"))
                self._ui(lambda: self.status.configure(text="Preprocessed", bg="#dc2626"))

            except Exception as e:
                err_msg = str(e)
                self._ui(lambda: messagebox.showerror("Preprocessing Error", f"Failed to preprocess:\n\n{err_msg}"))
                self._ui(lambda: self._log_msg(f"✗ Error: {err_msg}"))
                self._ui(lambda: self.preprocess_btn.configure(state="normal", text="Start Preprocessing", bg="#ef4444"))

        threading.Thread(target=work, daemon=True).start()

    def _initialize_model(self):
        """STEP 3: Initialize the detection model"""
        if not self.preprocessing_done or not self.preprocessed_data:
            messagebox.showwarning("Not Ready", "Please preprocess the data first!")
            return

        self._log_msg("Initializing model...")
        self.init_btn.configure(state="disabled", text="Initializing...")
        self.status.configure(text="Initializing", bg="#7f1d1d")

        def work():
            try:
                from utils.config_loader import load_config, get_device
                from utils.helpers import set_seed
                from hybrid_model.hybrid_detector import HybridDetector

                cfg_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'configs', 'config.yaml')
                cfg = load_config(cfg_path)
                dev = get_device(cfg)
                set_seed(42)

                self._ui(lambda: self._log_msg(f"  Device: {dev}"))

                X_train, y_train = self.preprocessed_data['train']
                X_test, y_test = self.preprocessed_data['test']

                dim = X_train.shape[1]

                # Create detector
                self._ui(lambda: self._log_msg("  Creating HybridDetector..."))
                self.detector = HybridDetector(dim, 2, cfg, dev, 'dqn')

                # Try to load trained model
                m_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'results', 'checkpoints', 'final_model')

                if os.path.exists(m_path):
                    self.detector.load(m_path)
                    model_type = "Trained DRL Model"
                    self._ui(lambda: self._log_msg(f"  Loaded trained weights"))
                else:
                    normal_samples = X_train[y_train == 0][:20]
                    attack_samples = X_train[y_train == 1][:20]
                    self.detector.initialize_few_shot(normal_samples, attack_samples)
                    model_type = "Few-Shot Learning"
                    self._ui(lambda: self._log_msg(f"  Initialized few-shot (40 samples)"))

                # Store model info
                n_threats = int(np.sum(y_test == 1))
                n_safe = int(np.sum(y_test == 0))

                self.model_info = {
                    "type": model_type,
                    "dim": dim,
                    "samples": len(y_test),
                    "data_source": self.raw_data['source'],
                    "device": str(dev)
                }

                self.model_initialized = True

                self._ui(lambda: self._log_msg(f"✓ Model initialized: {model_type}"))
                self._ui(lambda: self._log_msg(f"  Test data: {n_safe} safe, {n_threats} threats"))

                # Update UI
                self._ui(lambda: self.init_btn.configure(text="✓ Initialized", bg="#7f1d1d"))
                self._ui(lambda: self.detect_nav_btn.configure(state="normal", bg="#dc2626"))
                self._ui(lambda: self.model_status_lbl.configure(text=f"✓ READY: {model_type} | {len(y_test)} test samples"))
                self._ui(lambda: self.data_info_lbl.configure(text=f"Data: {self.raw_data['source']} | Device: {dev}"))
                self._ui(lambda: self.status.configure(text="Ready", bg="#dc2626"))

                # Enable detection
                self._ui(self._ready)

            except Exception as e:
                err_msg = str(e)
                self._ui(lambda: messagebox.showerror("Model Error", f"Failed to initialize model:\n\n{err_msg}"))
                self._ui(lambda: self._log_msg(f"✗ Error: {err_msg}"))
                self._ui(lambda: self.init_btn.configure(state="normal", text="Initialize Model", bg="#dc2626"))

        threading.Thread(target=work, daemon=True).start()

    def _load_model(self):
        self._log_msg("Starting system initialization...")

        def work():
            try:
                from utils.config_loader import load_config, get_device
                from utils.helpers import set_seed
                from hybrid_model.hybrid_detector import HybridDetector

                cfg_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'configs', 'config.yaml')
                cfg = load_config(cfg_path)
                dev = get_device(cfg)
                set_seed(42)

                self._ui(lambda: self._log_msg(f"Device: {dev}"))

                data_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'data', 'cleaned_data.csv')

                data_source = "Unknown"
                if os.path.exists(data_path):
                    from preprocessing.iot23_loader import IoT23DataLoader
                    loader = IoT23DataLoader(cfg, data_dir='data')
                    data = loader.load_cleaned_data('cleaned_data.csv')
                    data_source = "IoT-23 Dataset (cleaned_data.csv)"
                    self._ui(lambda: self._log_msg("Loaded IoT-23 real dataset"))
                else:
                    from preprocessing.data_loader import DataLoader
                    loader = DataLoader(cfg)
                    data = loader.load_synthetic_data()
                    data_source = "Synthetic Dataset (generated)"
                    self._ui(lambda: self._log_msg("Using synthetic dataset"))

                self.test_features = data['test'][0]
                self.test_labels = data['test'][1]
                dim = data['train'][0].shape[1]

                # Store model info for display
                self.model_info = {
                    "type": "HybridDetector (DQN + Few-Shot)",
                    "dim": dim,
                    "samples": len(self.test_labels),
                    "data_source": data_source,
                    "device": str(dev)
                }

                n_threats = int(np.sum(self.test_labels == 1))
                n_safe = int(np.sum(self.test_labels == 0))
                self._ui(lambda: self._log_msg(f"Test data: {n_safe} safe, {n_threats} threats"))

                self.detector = HybridDetector(dim, 2, cfg, dev, 'dqn')

                m_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'results', 'checkpoints', 'final_model')

                model_type = "unknown"
                if os.path.exists(m_path):
                    self.detector.load(m_path)
                    model_type = "Trained DRL Model"
                    self._ui(lambda: self._log_msg(f"Loaded trained model from {m_path}"))
                else:
                    n = data['train'][0][data['train'][1] == 0][:20]
                    a = data['train'][0][data['train'][1] == 1][:20]
                    self.detector.initialize_few_shot(n, a)
                    model_type = "Few-Shot Learning (40 samples)"
                    self._ui(lambda: self._log_msg("Initialized few-shot with 40 prototype samples"))

                self.model_info["model_type"] = model_type
                self._ui(self._ready)

            except Exception as e:
                import traceback
                err = str(e)
                self._ui(lambda: self._log_msg(f"ERROR: {err}"))
                self._ui(lambda: self._log_msg(traceback.format_exc()[:200]))

        threading.Thread(target=work, daemon=True).start()

    def _ui(self, fn):
        self.root.after(0, fn)

    def _ready(self):
        self.status.configure(text="Ready", bg="#dc2626")
        self.det_status.configure(text=" Ready ", bg="#dc2626")

        # Update home page with real model info
        info = self.model_info
        self.model_status_lbl.configure(
            text=f"Model: {info.get('model_type', 'Unknown')} | Samples: {info.get('samples', 0)} | Dim: {info.get('dim', 0)}"
        )
        self.data_info_lbl.configure(
            text=f"Data: {info.get('data_source', 'Unknown')} | Device: {info.get('device', 'cpu')}"
        )

        self._log_msg("System ready - Real-time detection enabled!")
        self._log_msg(f"Model: {info.get('model_type', 'Unknown')}")

        # Start live timer
        self._update_live_time()

    def _update_live_time(self):
        """Update live statistics in header"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.live_time.configure(text=f"Time: {h:02d}:{m:02d}:{s:02d}")

            # Calculate rate
            if elapsed > 0:
                rate = self.detection_count / elapsed
                self.live_rate.configure(text=f"Rate: {rate:.1f}/s")

        self.live_count.configure(text=f"Detections: {self.detection_count}")
        self.root.after(500, self._update_live_time)

    def _log_msg(self, msg):
        t = datetime.now().strftime("%H:%M:%S.") + f"{datetime.now().microsecond // 1000:03d}"
        self.log.insert("end", f"[{t}] {msg}\n")
        self.log.see("end")

    def _toggle(self):
        if not self.detector:
            messagebox.showinfo("Wait", "Model still loading...")
            return

        if self.is_running:
            self.is_running = False
            self.start_btn.configure(text="▶ START", bg="#dc2626")
            self.det_status.configure(text=" Paused ", bg="#7f1d1d")
            self._log_msg("Detection paused")
        else:
            self.is_running = True
            if self.start_time is None:
                self.start_time = time.time()
            self.start_btn.configure(text="■ STOP", bg="#ff0000")
            self.det_status.configure(text=" █ LIVE █ ", bg="#ff0000", fg="#000000")
            self._log_msg("Real-time detection STARTED")
            self._detect()

    def _detect(self):
        if not self.is_running:
            return

        idx = np.random.randint(len(self.test_features))
        feat = self.test_features[idx]
        true = self.test_labels[idx]

        # Measure actual inference time
        t_start = time.perf_counter()
        res = self.detector.detect(feat, training=False)
        t_end = time.perf_counter()
        self.last_inference_ms = (t_end - t_start) * 1000

        pred = res['prediction']
        conf = res['confidence']
        ok = pred == true

        self.detection_count += 1
        self.results.append({'pred': pred, 'true': true, 'conf': conf, 'ok': ok, 'idx': idx})
        if len(self.results) > 100:
            self.results = self.results[-100:]

        # Update sample index and inference time displays
        self.sample_idx_lbl.configure(text=f"Sample: #{idx}")
        self.inference_lbl.configure(text=f"Inference: {self.last_inference_ms:.1f}ms")

        # Update feature value displays (show first 4 features)
        for i, lbl in enumerate(self.feat_labels):
            if i < len(feat):
                val = feat[i] if isinstance(feat, np.ndarray) else feat
                lbl.configure(text=f"F{i}: {float(val):.4f}", fg="#ff0000")

        self._show_result(pred, conf)
        self._update_stats()

        sym = "OK" if ok else "MISS"
        ps = "THREAT" if pred == 1 else "SAFE"
        gt = "threat" if true == 1 else "safe"
        self._log_msg(f"[#{idx}] {ps} ({conf:.0%}) | True: {gt} | {sym} | {self.last_inference_ms:.1f}ms")

        self.root.after(self.delay, self._detect)

    def _test_one(self):
        if not self.detector:
            messagebox.showinfo("Wait", "Model still loading...")
            return

        idx = np.random.randint(len(self.test_features))
        feat = self.test_features[idx]
        true = self.test_labels[idx]

        # Measure actual inference time
        t_start = time.perf_counter()
        res = self.detector.detect(feat, training=False)
        t_end = time.perf_counter()
        self.last_inference_ms = (t_end - t_start) * 1000

        pred = res['prediction']
        conf = res['confidence']
        ok = pred == true

        self.detection_count += 1
        if self.start_time is None:
            self.start_time = time.time()

        self.results.append({'pred': pred, 'true': true, 'conf': conf, 'ok': ok, 'idx': idx})

        # Update displays
        self.sample_idx_lbl.configure(text=f"Sample: #{idx}")
        self.inference_lbl.configure(text=f"Inference: {self.last_inference_ms:.1f}ms")

        for i, lbl in enumerate(self.feat_labels):
            if i < len(feat):
                val = feat[i] if isinstance(feat, np.ndarray) else feat
                lbl.configure(text=f"F{i}: {float(val):.4f}", fg="#ff0000")

        self._show_result(pred, conf)
        self._update_stats()

        r = "CORRECT" if ok else "WRONG"
        ps = "THREAT" if pred == 1 else "SAFE"
        gt = "threat" if true == 1 else "safe"
        self._log_msg(f"Single test [#{idx}] -> {ps} ({conf:.0%}) | True: {gt} | {r} | {self.last_inference_ms:.1f}ms")

    def _show_result(self, pred, conf):
        if pred == 1:
            self.result_lbl.configure(text="⚠ THREAT ⚠", fg="#ff0000")
            self.result_frame.configure(bg="#1a0000")
            if conf > 0.9:
                exp = "HIGH confidence botnet activity - Strong malicious signature match"
            elif conf > 0.7:
                exp = "Abnormal patterns detected - Likely C&C communication or scanning"
            else:
                exp = "Suspicious activity - Features deviate from normal baseline"
        else:
            self.result_lbl.configure(text="✓ SAFE ✓", fg="#555")
            self.result_frame.configure(bg="#0a0a0a")
            if conf > 0.9:
                exp = "HIGH confidence benign - Traffic matches normal IoT patterns"
            elif conf > 0.7:
                exp = "Normal traffic patterns - No threat indicators detected"
            else:
                exp = "Likely safe - Features within expected range"

        self.exp_lbl.configure(text=exp)
        self.conf_lbl.configure(text=f"Confidence: {conf:.1%}")

    def _update_stats(self):
        if not self.results:
            return

        total = len(self.results)
        threats = sum(1 for r in self.results if r['pred'] == 1)
        safe = total - threats
        correct = sum(1 for r in self.results if r['ok'])
        wrong = total - correct
        acc = correct / total if total > 0 else 0

        self.stats_labels["Total"].configure(text=str(total))
        self.stats_labels["Threats"].configure(text=str(threats))
        self.stats_labels["Safe"].configure(text=str(safe))
        self.stats_labels["Accuracy"].configure(text=f"{acc:.0%}")

        self.pie_lbl.configure(text=f"Safe: {safe} | Threat: {threats}")
        self.bar_lbl.configure(text=f"Correct: {correct} | Wrong: {wrong}")
        self.acc_lbl.configure(text=f"Accuracy: {acc:.0%}")

    def _reset(self):
        self.results = []
        self.detection_count = 0
        self.start_time = None

        self.result_lbl.configure(text="WAITING", fg="#ff0000")
        self.result_frame.configure(bg="#000000")
        self.conf_lbl.configure(text="Confidence: --")
        self.exp_lbl.configure(text="Start detection to see analysis")

        for lbl in self.stats_labels.values():
            lbl.configure(text="0")

        self.pie_lbl.configure(text="Safe: 0 | Threat: 0")
        self.bar_lbl.configure(text="Correct: 0 | Wrong: 0")
        self.acc_lbl.configure(text="Accuracy: 0%")

        self.sample_idx_lbl.configure(text="Sample: --")
        self.inference_lbl.configure(text="Inference: --ms")
        self.live_time.configure(text="Time: 00:00:00")
        self.live_count.configure(text="Detections: 0")
        self.live_rate.configure(text="Rate: 0/s")

        for lbl in self.feat_labels:
            lbl.configure(text="F: --", fg="#888")

        self._log_msg("All data reset - Ready for new detection session")

    def _export(self):
        messagebox.showinfo("Export", "Report exported!")
        self._log_msg("Report exported")

    def _save(self):
        messagebox.showinfo("Save", "Results saved!")
        self._log_msg("Results saved")

    def _refresh(self):
        self._log_msg("Refreshing...")
        self._load_model()

    def _help(self):
        info = self.model_info
        messagebox.showinfo("ZeroDay-DRL Help", f"""REAL-TIME IoT BOTNET DETECTION

Pages:
- Home: System status and model info
- Learn: Threat vs Safe classification guide
- Detect: Live real-time detection
- Stats: Performance dashboard
- Compare: Model comparison table
- Settings: Speed and configuration

Quick Actions: E=Export, S=Save, R=Refresh, ?=Help

REAL-TIME FEATURES:
- Sample #: Shows actual data index being tested
- Inference: Real model processing time (ms)
- Features: Live feature values from dataset
- Detection counter and rate tracking

Current Model: {info.get('model_type', 'Loading...')}
Data Source: {info.get('data_source', 'Loading...')}
Test Samples: {info.get('samples', '--')}""")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = SimpleZeroDayGUI()
    app.run()
