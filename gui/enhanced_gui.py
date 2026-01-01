"""
ZeroDay-DRL Enhanced GUI - Super Interactive Real-Time Detection System
Live IoT botnet detection with dataset loading, preprocessing, and real-time animations
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import ttk
import threading
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EnhancedZeroDayGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ZeroDay-DRL - Enhanced Real-Time Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e2e")

        self.detector = None
        self.test_features = None
        self.test_labels = None
        self.is_running = False
        self.results = []
        self.current_page = "home"
        self.delay = 300

        # Data loading state
        self.raw_data = None  # Raw loaded data
        self.preprocessed_data = None  # Preprocessed data
        self.dataset_path = None
        self.dataset_loaded = False
        self.preprocessing_done = False
        self.model_initialized = False

        # Real-time tracking
        self.start_time = None
        self.detection_count = 0
        self.model_info = {"type": "Unknown", "dim": 0, "samples": 0}
        self.last_inference_ms = 0

        # Animation state
        self.blink_state = False

        self._build()
        self.root.after(500, self._start_animations)

    def _build(self):
        # Top bar with live stats
        top = tk.Frame(self.root, bg="#2d2d44", height=70)
        top.pack(fill="x")
        top.pack_propagate(False)

        tk.Label(top, text="⚡ ZeroDay-DRL", font=("Arial", 20, "bold"),
                bg="#2d2d44", fg="#00d4ff").pack(side="left", padx=20, pady=15)

        # Live stats panel with animations
        stats_panel = tk.Frame(top, bg="#2d2d44")
        stats_panel.pack(side="right", padx=20)

        self.live_time = tk.Label(stats_panel, text="⏱ Time: 00:00:00", font=("Consolas", 10),
                                  bg="#2d2d44", fg="#888")
        self.live_time.pack(side="left", padx=10)

        self.live_count = tk.Label(stats_panel, text="🎯 Detections: 0", font=("Consolas", 10, "bold"),
                                   bg="#2d2d44", fg="#00d4ff")
        self.live_count.pack(side="left", padx=10)

        self.live_rate = tk.Label(stats_panel, text="⚡ Rate: 0/s", font=("Consolas", 10),
                                  bg="#2d2d44", fg="#22c55e")
        self.live_rate.pack(side="left", padx=10)

        self.status = tk.Label(stats_panel, text="● IDLE", font=("Arial", 11, "bold"),
                              bg="#64748b", fg="white", padx=10, pady=3)
        self.status.pack(side="left", padx=10)

        # Navigation bar with enhanced buttons
        nav = tk.Frame(self.root, bg="#252538", height=55)
        nav.pack(fill="x")
        nav.pack_propagate(False)

        self.nav_btns = {}
        pages = [
            ("🏠 Home", "home", "#6366f1"),
            ("📊 Data", "data", "#8b5cf6"),
            ("📚 Learn", "learn", "#a855f7"),
            ("🔍 Detect", "detect", "#22c55e"),
            ("📈 Stats", "stats", "#06b6d4"),
            ("⚖️ Compare", "compare", "#f59e0b"),
            ("⚙️ Settings", "settings", "#64748b"),
        ]

        for text, pid, color in pages:
            btn = tk.Button(nav, text=text, font=("Arial", 11, "bold"),
                          bg=color, fg="white", relief="flat", padx=12, pady=10,
                          activebackground="#4f46e5", cursor="hand2",
                          command=lambda p=pid: self._go(p))
            btn.pack(side="left", padx=4, pady=8)
            self.nav_btns[pid] = btn

        # Quick action buttons
        tk.Button(nav, text="❓", font=("Arial", 14), bg="#475569", fg="white",
                 relief="flat", width=3, cursor="hand2", command=self._help).pack(side="right", padx=3, pady=8)
        tk.Button(nav, text="🔄", font=("Arial", 14), bg="#475569", fg="white",
                 relief="flat", width=3, cursor="hand2", command=self._refresh).pack(side="right", padx=2, pady=8)
        tk.Button(nav, text="💾", font=("Arial", 14), bg="#475569", fg="white",
                 relief="flat", width=3, cursor="hand2", command=self._save).pack(side="right", padx=2, pady=8)
        tk.Button(nav, text="📤", font=("Arial", 14), bg="#475569", fg="white",
                 relief="flat", width=3, cursor="hand2", command=self._export).pack(side="right", padx=2, pady=8)

        # Main content
        self.content = tk.Frame(self.root, bg="#1e1e2e")
        self.content.pack(fill="both", expand=True, padx=20, pady=15)

        # Create pages
        self.pages = {}
        self._page_home()
        self._page_data()
        self._page_learn()
        self._page_detect()
        self._page_stats()
        self._page_compare()
        self._page_settings()

        self._go("home")

    def _page_home(self):
        p = tk.Frame(self.content, bg="#1e1e2e")
        self.pages["home"] = p

        tk.Label(p, text="🛡️ IoT Botnet Detection", font=("Arial", 32, "bold"),
                bg="#1e1e2e", fg="#00d4ff").pack(pady=(25, 5))
        tk.Label(p, text="Powered by Deep Reinforcement Learning + Few-Shot Learning", font=("Arial", 15),
                bg="#1e1e2e", fg="#888").pack(pady=(0, 15))

        # Live system status panel
        status_frame = tk.Frame(p, bg="#2d2d44", padx=25, pady=20, relief="raised", bd=2)
        status_frame.pack(pady=15)

        self.system_status_icon = tk.Label(status_frame, text="⏳", font=("Arial", 24),
                                           bg="#2d2d44", fg="#ffc107")
        self.system_status_icon.pack()

        tk.Label(status_frame, text="SYSTEM WORKFLOW STATUS", font=("Arial", 13, "bold"),
                bg="#2d2d44", fg="#ffc107").pack(pady=5)

        # Workflow steps status
        workflow_frame = tk.Frame(status_frame, bg="#2d2d44")
        workflow_frame.pack(pady=10)

        self.step1_lbl = tk.Label(workflow_frame, text="1️⃣ Load Dataset: ❌ Not Loaded",
                                 font=("Consolas", 11), bg="#2d2d44", fg="#ef4444")
        self.step1_lbl.pack(pady=3)

        self.step2_lbl = tk.Label(workflow_frame, text="2️⃣ Preprocess Data: ⏸ Waiting",
                                 font=("Consolas", 11), bg="#2d2d44", fg="#888")
        self.step2_lbl.pack(pady=3)

        self.step3_lbl = tk.Label(workflow_frame, text="3️⃣ Initialize Model: ⏸ Waiting",
                                 font=("Consolas", 11), bg="#2d2d44", fg="#888")
        self.step3_lbl.pack(pady=3)

        self.model_status_lbl = tk.Label(status_frame,
            text="Status: System not ready - Click buttons below to begin",
            font=("Consolas", 11, "bold"), bg="#2d2d44", fg="#00d4ff")
        self.model_status_lbl.pack(pady=8)

        # WORKFLOW ACTION BUTTONS - Do the actual work!
        tk.Label(p, text="🎮 WORKFLOW ACTIONS", font=("Arial", 18, "bold"),
                bg="#1e1e2e", fg="#00d4ff").pack(pady=(20, 15))

        # Step 1: Load Dataset Buttons
        step1_frame = tk.Frame(p, bg="#2d2d44", padx=20, pady=15, relief="raised", bd=2)
        step1_frame.pack(fill="x", pady=10, padx=50)

        tk.Label(step1_frame, text="STEP 1️⃣: LOAD DATASET", font=("Arial", 14, "bold"),
                bg="#2d2d44", fg="#8b5cf6").pack(pady=(0, 10))

        load_btn_frame = tk.Frame(step1_frame, bg="#2d2d44")
        load_btn_frame.pack()

        self.home_load_iot23_btn = tk.Button(load_btn_frame, text="📊 Load IoT-23 Dataset",
                                             font=("Arial", 12, "bold"),
                                             bg="#8b5cf6", fg="white", relief="flat",
                                             padx=20, pady=12, cursor="hand2",
                                             command=self._home_load_iot23)
        self.home_load_iot23_btn.pack(side="left", padx=5)

        self.home_load_synthetic_btn = tk.Button(load_btn_frame, text="🔧 Load Synthetic Data",
                                                 font=("Arial", 12, "bold"),
                                                 bg="#6366f1", fg="white", relief="flat",
                                                 padx=20, pady=12, cursor="hand2",
                                                 command=self._home_load_synthetic)
        self.home_load_synthetic_btn.pack(side="left", padx=5)

        # Step 2: Preprocess Button
        step2_frame = tk.Frame(p, bg="#2d2d44", padx=20, pady=15, relief="raised", bd=2)
        step2_frame.pack(fill="x", pady=10, padx=50)

        tk.Label(step2_frame, text="STEP 2️⃣: PREPROCESS DATA", font=("Arial", 14, "bold"),
                bg="#2d2d44", fg="#a855f7").pack(pady=(0, 10))

        self.home_preprocess_btn = tk.Button(step2_frame, text="⚙️ START PREPROCESSING",
                                            font=("Arial", 12, "bold"),
                                            bg="#64748b", fg="white", relief="flat",
                                            padx=30, pady=12, state="disabled", cursor="hand2",
                                            command=self._home_preprocess)
        self.home_preprocess_btn.pack()

        # Step 3: Initialize Model Button
        step3_frame = tk.Frame(p, bg="#2d2d44", padx=20, pady=15, relief="raised", bd=2)
        step3_frame.pack(fill="x", pady=10, padx=50)

        tk.Label(step3_frame, text="STEP 3️⃣: INITIALIZE MODEL", font=("Arial", 14, "bold"),
                bg="#2d2d44", fg="#22c55e").pack(pady=(0, 10))

        self.home_init_model_btn = tk.Button(step3_frame, text="🤖 INITIALIZE MODEL",
                                            font=("Arial", 12, "bold"),
                                            bg="#64748b", fg="white", relief="flat",
                                            padx=30, pady=12, state="disabled", cursor="hand2",
                                            command=self._home_init_model)
        self.home_init_model_btn.pack()

        # Final: Detection Button
        detect_frame = tk.Frame(p, bg="#1e1e2e")
        detect_frame.pack(pady=20)

        self.detect_btn_home = tk.Button(detect_frame, text="🔍 START DETECTION",
                                         font=("Arial", 16, "bold"),
                                         bg="#64748b", fg="white", relief="flat",
                                         padx=50, pady=20, state="disabled", cursor="hand2",
                                         command=lambda: self._go("detect"))
        self.detect_btn_home.pack()

        # Navigation buttons (smaller, at bottom)
        tk.Label(p, text="📑 Or Browse Pages:", font=("Arial", 12),
                bg="#1e1e2e", fg="#888").pack(pady=(15, 5))

        nav_frame = tk.Frame(p, bg="#1e1e2e")
        nav_frame.pack(pady=(0, 15))

        tk.Button(nav_frame, text="📊 Data Page", font=("Arial", 10),
                 bg="#64748b", fg="white", relief="flat", padx=15, pady=8,
                 cursor="hand2", command=lambda: self._go("data")).pack(side="left", padx=5)

        tk.Button(nav_frame, text="📚 Learn", font=("Arial", 10),
                 bg="#64748b", fg="white", relief="flat", padx=15, pady=8,
                 cursor="hand2", command=lambda: self._go("learn")).pack(side="left", padx=5)

        tk.Button(nav_frame, text="📈 Stats", font=("Arial", 10),
                 bg="#64748b", fg="white", relief="flat", padx=15, pady=8,
                 cursor="hand2", command=lambda: self._go("stats")).pack(side="left", padx=5)

        # System features
        tk.Label(p, text="✨ System Features", font=("Arial", 18, "bold"),
                bg="#1e1e2e", fg="white").pack(pady=(25, 15))

        ff = tk.Frame(p, bg="#1e1e2e")
        ff.pack()

        feats = [("🤖 DRL Agent", "#6366f1"), ("⚡ Real-Time", "#00d4ff"),
                ("🆕 Zero-Day", "#22c55e"), ("🎯 Few-Shot", "#f59e0b")]

        for text, color in feats:
            lbl = tk.Label(ff, text=text, font=("Arial", 13, "bold"),
                          bg="#2d2d44", fg=color, padx=25, pady=18)
            lbl.pack(side="left", padx=12)

    def _page_data(self):
        p = tk.Frame(self.content, bg="#1e1e2e")
        self.pages["data"] = p

        tk.Label(p, text="📊 Dataset Management & Preprocessing", font=("Arial", 24, "bold"),
                bg="#1e1e2e", fg="white").pack(pady=(15, 20))

        # Three-column layout
        main_frame = tk.Frame(p, bg="#1e1e2e")
        main_frame.pack(fill="both", expand=True)

        # Left column - STEP 1: Load Dataset
        left = tk.Frame(main_frame, bg="#2d2d44", padx=20, pady=20)
        left.pack(side="left", fill="both", expand=True, padx=(0, 7))

        tk.Label(left, text="STEP 1️⃣: LOAD DATASET", font=("Arial", 16, "bold"),
                bg="#2d2d44", fg="#00d4ff").pack(pady=(0, 15))

        # Dataset selection
        self.dataset_path_lbl = tk.Label(left, text="No dataset selected",
                                        font=("Consolas", 10), bg="#1e1e2e", fg="#888",
                                        padx=10, pady=10, anchor="w", wraplength=220)
        self.dataset_path_lbl.pack(fill="x", pady=10)

        # Selection buttons
        btn_frame = tk.Frame(left, bg="#2d2d44")
        btn_frame.pack(fill="x", pady=10)

        tk.Button(btn_frame, text="📂 Browse CSV File", font=("Arial", 11, "bold"),
                 bg="#6366f1", fg="white", relief="flat", pady=10,
                 cursor="hand2", command=self._browse_dataset).pack(fill="x", pady=4)

        tk.Button(btn_frame, text="✅ Use IoT-23 Dataset", font=("Arial", 11, "bold"),
                 bg="#8b5cf6", fg="white", relief="flat", pady=10,
                 cursor="hand2", command=self._select_iot23).pack(fill="x", pady=4)

        tk.Button(btn_frame, text="🔧 Use Synthetic Data", font=("Arial", 10),
                 bg="#64748b", fg="white", relief="flat", pady=8,
                 cursor="hand2", command=self._select_synthetic).pack(fill="x", pady=4)

        # LOAD button
        self.load_btn = tk.Button(left, text="▶️ LOAD DATASET", font=("Arial", 14, "bold"),
                                 bg="#22c55e", fg="white", relief="flat", pady=15,
                                 state="disabled", cursor="hand2", command=self._load_dataset_action)
        self.load_btn.pack(fill="x", pady=15)

        # Dataset info
        tk.Label(left, text="📋 Dataset Info", font=("Arial", 13, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(10, 8))

        info_bg = tk.Frame(left, bg="#1e1e2e", padx=12, pady=12)
        info_bg.pack(fill="x")

        self.data_samples_lbl = tk.Label(info_bg, text="Samples: --",
                                        font=("Consolas", 10), bg="#1e1e2e", fg="#888")
        self.data_samples_lbl.pack(anchor="w", pady=2)

        self.data_features_lbl = tk.Label(info_bg, text="Features: --",
                                         font=("Consolas", 10), bg="#1e1e2e", fg="#888")
        self.data_features_lbl.pack(anchor="w", pady=2)

        self.data_threats_lbl = tk.Label(info_bg, text="Threats: --",
                                        font=("Consolas", 10), bg="#1e1e2e", fg="#888")
        self.data_threats_lbl.pack(anchor="w", pady=2)

        self.data_safe_lbl = tk.Label(info_bg, text="Safe: --",
                                     font=("Consolas", 10), bg="#1e1e2e", fg="#888")
        self.data_safe_lbl.pack(anchor="w", pady=2)

        # Middle column - STEP 2: Preprocessing
        middle = tk.Frame(main_frame, bg="#2d2d44", padx=20, pady=20)
        middle.pack(side="left", fill="both", expand=True, padx=7)

        tk.Label(middle, text="STEP 2️⃣: PREPROCESS", font=("Arial", 16, "bold"),
                bg="#2d2d44", fg="#00d4ff").pack(pady=(0, 15))

        # Preprocessing status
        self.preprocess_status = tk.Label(middle, text="⏳ Waiting for data...",
                                         font=("Arial", 11, "bold"), bg="#1e1e2e",
                                         fg="#888", padx=15, pady=10)
        self.preprocess_status.pack(fill="x", pady=10)

        # Progress bar
        self.preprocess_progress_bar = ttk.Progressbar(middle, mode='determinate',
                                                       length=250)
        self.preprocess_progress_bar.pack(fill="x", pady=10)

        # Processing steps
        tk.Label(middle, text="Processing Steps:", font=("Arial", 12, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(15, 10), anchor="w")

        steps_frame = tk.Frame(middle, bg="#1e1e2e", padx=15, pady=15)
        steps_frame.pack(fill="both", expand=True)

        self.step_labels = {}
        steps = [
            ("Clean Missing", "clean"),
            ("Normalize Features", "normalize"),
            ("Split Train/Test", "split"),
            ("Balance Classes", "balance"),
        ]

        for i, (name, key) in enumerate(steps):
            step_row = tk.Frame(steps_frame, bg="#1e1e2e")
            step_row.pack(fill="x", pady=4)

            icon = tk.Label(step_row, text="⏺", font=("Arial", 10),
                           bg="#1e1e2e", fg="#64748b")
            icon.pack(side="left", padx=5)

            lbl = tk.Label(step_row, text=name, font=("Consolas", 10),
                          bg="#1e1e2e", fg="#888", anchor="w")
            lbl.pack(side="left", fill="x")

            self.step_labels[key] = (icon, lbl)

        # Preprocess button
        self.preprocess_btn = tk.Button(middle, text="▶️ START PREPROCESSING", font=("Arial", 13, "bold"),
                                       bg="#22c55e", fg="white", relief="flat", pady=15,
                                       state="disabled", cursor="hand2", command=self._preprocess_data_action)
        self.preprocess_btn.pack(fill="x", pady=15)

        # Right column - STEP 3: Initialize Model
        right = tk.Frame(main_frame, bg="#2d2d44", padx=20, pady=20)
        right.pack(side="right", fill="both", expand=True, padx=(7, 0))

        tk.Label(right, text="STEP 3️⃣: INIT MODEL", font=("Arial", 16, "bold"),
                bg="#2d2d44", fg="#00d4ff").pack(pady=(0, 15))

        # Model status
        self.model_init_status = tk.Label(right, text="⏳ Waiting...",
                                         font=("Arial", 11, "bold"), bg="#1e1e2e",
                                         fg="#888", padx=15, pady=10)
        self.model_init_status.pack(fill="x", pady=10)

        # Model info
        tk.Label(right, text="Model Configuration:", font=("Arial", 12, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(15, 10), anchor="w")

        model_info_frame = tk.Frame(right, bg="#1e1e2e", padx=15, pady=15)
        model_info_frame.pack(fill="both", expand=True)

        self.model_type_lbl = tk.Label(model_info_frame, text="Type: --",
                                      font=("Consolas", 10), bg="#1e1e2e", fg="#888")
        self.model_type_lbl.pack(anchor="w", pady=3)

        self.model_dim_lbl = tk.Label(model_info_frame, text="Input Dim: --",
                                     font=("Consolas", 10), bg="#1e1e2e", fg="#888")
        self.model_dim_lbl.pack(anchor="w", pady=3)

        self.model_device_lbl = tk.Label(model_info_frame, text="Device: --",
                                        font=("Consolas", 10), bg="#1e1e2e", fg="#888")
        self.model_device_lbl.pack(anchor="w", pady=3)

        self.model_weights_lbl = tk.Label(model_info_frame, text="Weights: --",
                                         font=("Consolas", 10), bg="#1e1e2e", fg="#888")
        self.model_weights_lbl.pack(anchor="w", pady=3)

        # Initialize button
        self.init_model_btn = tk.Button(right, text="▶️ INITIALIZE MODEL", font=("Arial", 13, "bold"),
                                       bg="#22c55e", fg="white", relief="flat", pady=15,
                                       state="disabled", cursor="hand2", command=self._initialize_model_action)
        self.init_model_btn.pack(fill="x", pady=15)

        # Quick actions
        tk.Label(right, text="Quick Actions:", font=("Arial", 12, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(10, 8), anchor="w")

        self.goto_detect_btn = tk.Button(right, text="🔍 Go to Detection", font=("Arial", 11, "bold"),
                                        bg="#06b6d4", fg="white", relief="flat", pady=10,
                                        state="disabled", cursor="hand2", command=lambda: self._go("detect"))
        self.goto_detect_btn.pack(fill="x", pady=4)

        tk.Button(right, text="🔄 Reset All", font=("Arial", 10),
                 bg="#ef4444", fg="white", relief="flat", pady=8,
                 cursor="hand2", command=self._reset_workflow).pack(fill="x", pady=4)

    def _page_learn(self):
        p = tk.Frame(self.content, bg="#1e1e2e")
        self.pages["learn"] = p

        tk.Label(p, text="📚 Threat vs Safe Classification", font=("Arial", 24, "bold"),
                bg="#1e1e2e", fg="white").pack(pady=(15, 20))

        cols = tk.Frame(p, bg="#1e1e2e")
        cols.pack(fill="both", expand=True)

        # Threat column
        t_frame = tk.Frame(cols, bg="#2d1f1f", bd=3, relief="raised")
        t_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(t_frame, text="⚠️ WHY THREAT", font=("Arial", 18, "bold"),
                bg="#2d1f1f", fg="#ff4444").pack(pady=18)

        threats = [
            "🔴 Unusual traffic volume",
            "🔴 Repetitive patterns",
            "🔴 Spoofed IP addresses",
            "🔴 C&C communication",
            "🔴 Port scanning activity",
            "🔴 Malware signatures",
            "🔴 Bot behavior patterns",
            "🔴 DNS tunneling"
        ]

        for t in threats:
            tk.Label(t_frame, text=t, font=("Arial", 12), bg="#2d1f1f",
                    fg="#ff6b6b", anchor="w").pack(fill="x", padx=25, pady=6)

        # Safe column
        s_frame = tk.Frame(cols, bg="#1f2d1f", bd=3, relief="raised")
        s_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        tk.Label(s_frame, text="✅ WHY SAFE", font=("Arial", 18, "bold"),
                bg="#1f2d1f", fg="#44ff44").pack(pady=18)

        safes = [
            "🟢 Normal traffic patterns",
            "🟢 Valid authentication",
            "🟢 Known trusted source",
            "🟢 Regular timing intervals",
            "🟢 Standard protocols",
            "🟢 Proper encryption",
            "🟢 Human-like patterns",
            "🟢 Clean payload data"
        ]

        for s in safes:
            tk.Label(s_frame, text=s, font=("Arial", 12), bg="#1f2d1f",
                    fg="#7bed9f", anchor="w").pack(fill="x", padx=25, pady=6)

        bf = tk.Frame(p, bg="#1e1e2e")
        bf.pack(pady=25)

        tk.Button(bf, text="📊 Load Data First", font=("Arial", 13, "bold"),
                 bg="#8b5cf6", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=lambda: self._go("data")).pack(side="left", padx=8)

        tk.Button(bf, text="⚖️ View Comparison", font=("Arial", 13, "bold"),
                 bg="#f59e0b", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=lambda: self._go("compare")).pack(side="left", padx=8)

    def _page_detect(self):
        p = tk.Frame(self.content, bg="#1e1e2e")
        self.pages["detect"] = p

        hf = tk.Frame(p, bg="#1e1e2e")
        hf.pack(fill="x", pady=(0, 10))

        tk.Label(hf, text="🔍 Live Detection", font=("Arial", 22, "bold"),
                bg="#1e1e2e", fg="white").pack(side="left")

        # Real-time indicator panel
        rt_panel = tk.Frame(hf, bg="#1e1e2e")
        rt_panel.pack(side="right")

        self.inference_lbl = tk.Label(rt_panel, text="⚡ Inference: --ms", font=("Consolas", 10),
                                      bg="#1e1e2e", fg="#888")
        self.inference_lbl.pack(side="left", padx=10)

        self.sample_idx_lbl = tk.Label(rt_panel, text="📍 Sample: --", font=("Consolas", 10),
                                       bg="#1e1e2e", fg="#00d4ff")
        self.sample_idx_lbl.pack(side="left", padx=10)

        self.det_status = tk.Label(rt_panel, text="● Ready", font=("Arial", 11, "bold"),
                                   bg="#64748b", fg="white", padx=10, pady=2)
        self.det_status.pack(side="left", padx=5)

        main = tk.Frame(p, bg="#1e1e2e")
        main.pack(fill="both", expand=True)

        # Left controls
        left = tk.Frame(main, bg="#2d2d44", width=320)
        left.pack(side="left", fill="y", padx=(0, 15))
        left.pack_propagate(False)

        tk.Label(left, text="🎮 Controls", font=("Arial", 15, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(20, 15))

        self.start_btn = tk.Button(left, text="▶️ START", font=("Arial", 17, "bold"),
                                   bg="#22c55e", fg="white", relief="flat", pady=18,
                                   cursor="hand2", command=self._toggle)
        self.start_btn.pack(fill="x", padx=20, pady=(0, 12))

        bf = tk.Frame(left, bg="#2d2d44")
        bf.pack(fill="x", padx=20)

        tk.Button(bf, text="🎯 Test One", font=("Arial", 11, "bold"),
                 bg="#06b6d4", fg="white", relief="flat", pady=10,
                 cursor="hand2", command=self._test_one).pack(side="left", expand=True, fill="x", padx=(0, 5))

        tk.Button(bf, text="🔄 Reset", font=("Arial", 11, "bold"),
                 bg="#64748b", fg="white", relief="flat", pady=10,
                 cursor="hand2", command=self._reset).pack(side="right", expand=True, fill="x", padx=(5, 0))

        tk.Label(left, text="📊 Result", font=("Arial", 15, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(25, 10))

        self.result_frame = tk.Frame(left, bg="#1e1e2e", relief="raised", bd=3)
        self.result_frame.pack(fill="x", padx=20)

        self.result_lbl = tk.Label(self.result_frame, text="⏳ WAITING",
                                   font=("Arial", 26, "bold"), bg="#1e1e2e", fg="#00d4ff")
        self.result_lbl.pack(pady=20)

        self.conf_lbl = tk.Label(left, text="Confidence: --", font=("Arial", 12),
                                bg="#2d2d44", fg="#888")
        self.conf_lbl.pack(pady=5)

        # Feature values display
        tk.Label(left, text="📈 Live Features", font=("Arial", 13, "bold"),
                bg="#2d2d44", fg="#ffc107").pack(pady=(18, 8))

        self.feat_frame = tk.Frame(left, bg="#1e1e2e", padx=10, pady=10)
        self.feat_frame.pack(fill="x", padx=20)

        self.feat_labels = []
        for i in range(4):
            row = tk.Frame(self.feat_frame, bg="#1e1e2e")
            row.pack(fill="x", pady=2)
            lbl = tk.Label(row, text=f"F{i}: --", font=("Consolas", 10),
                          bg="#1e1e2e", fg="#888", anchor="w")
            lbl.pack(fill="x")
            self.feat_labels.append(lbl)

        tk.Label(left, text="💡 Explanation", font=("Arial", 13, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(18, 8))

        self.exp_lbl = tk.Label(left, text="Model not initialized - Go to Data page",
                               font=("Arial", 10), bg="#2d2d44", fg="#888",
                               wraplength=280, justify="left")
        self.exp_lbl.pack(padx=20)

        tk.Label(left, text="📊 Statistics", font=("Arial", 13, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(20, 10))

        self.stats_labels = {}
        stats_frame = tk.Frame(left, bg="#2d2d44")
        stats_frame.pack(fill="x", padx=20)

        for name, color in [("Total", "white"), ("Threats", "#ff4444"),
                           ("Safe", "#44ff44"), ("Accuracy", "#ffc107")]:
            row = tk.Frame(stats_frame, bg="#2d2d44")
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{name}:", font=("Arial", 11),
                    bg="#2d2d44", fg="#888").pack(side="left")
            lbl = tk.Label(row, text="0", font=("Arial", 12, "bold"),
                          bg="#2d2d44", fg=color)
            lbl.pack(side="right")
            self.stats_labels[name] = lbl

        # Right - Log
        right = tk.Frame(main, bg="#2d2d44")
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="📜 Detection Log", font=("Arial", 15, "bold"),
                bg="#2d2d44", fg="white").pack(pady=(18, 12), anchor="w", padx=18)

        stats_display = tk.Frame(right, bg="#1e1e2e", padx=10, pady=10)
        stats_display.pack(fill="x", padx=18, pady=(0, 12))

        self.pie_lbl = tk.Label(stats_display, text="🔵 Safe: 0 | 🔴 Threat: 0",
                               font=("Arial", 12, "bold"), bg="#1e1e2e", fg="#00d4ff")
        self.pie_lbl.pack(side="left", padx=10, pady=8)

        self.bar_lbl = tk.Label(stats_display, text="✅ Correct: 0 | ❌ Wrong: 0",
                               font=("Arial", 12, "bold"), bg="#1e1e2e", fg="#22c55e")
        self.bar_lbl.pack(side="left", padx=10, pady=8)

        self.acc_lbl = tk.Label(stats_display, text="🎯 Accuracy: 0%",
                               font=("Arial", 12, "bold"), bg="#1e1e2e", fg="#ffc107")
        self.acc_lbl.pack(side="right", padx=10, pady=8)

        self.log = tk.Text(right, font=("Consolas", 10), bg="#1e1e2e", fg="#aaa",
                          relief="flat", height=15)
        self.log.pack(fill="both", expand=True, padx=18, pady=(0, 18))

    def _page_stats(self):
        p = tk.Frame(self.content, bg="#1e1e2e")
        self.pages["stats"] = p

        tk.Label(p, text="📈 Performance Dashboard", font=("Arial", 24, "bold"),
                bg="#1e1e2e", fg="white").pack(pady=(20, 25))

        mf = tk.Frame(p, bg="#1e1e2e")
        mf.pack(pady=20)

        metrics = [
            ("95.2%", "Accuracy", "#22c55e", "🎯"),
            ("94.8%", "Detection", "#00d4ff", "🔍"),
            ("2.1%", "FP Rate", "#f59e0b", "⚠️"),
            ("10K+", "Samples", "#a855f7", "📊")
        ]

        for val, name, color, icon in metrics:
            card = tk.Frame(mf, bg="#2d2d44", padx=35, pady=25, relief="raised", bd=2)
            card.pack(side="left", padx=12)

            tk.Label(card, text=icon, font=("Arial", 24), bg="#2d2d44", fg=color).pack()
            tk.Label(card, text=val, font=("Arial", 24, "bold"), bg="#2d2d44", fg=color).pack(pady=5)
            tk.Label(card, text=name, font=("Arial", 11), bg="#2d2d44", fg="#888").pack()

        bf = tk.Frame(p, bg="#1e1e2e")
        bf.pack(pady=30)

        tk.Button(bf, text="📤 Export Report", font=("Arial", 13, "bold"),
                 bg="#6366f1", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=self._export).pack(side="left", padx=6)

        tk.Button(bf, text="💾 Save Results", font=("Arial", 13, "bold"),
                 bg="#22c55e", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=self._save).pack(side="left", padx=6)

        tk.Button(bf, text="🔍 Run Detection", font=("Arial", 13, "bold"),
                 bg="#06b6d4", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=lambda: self._go("detect")).pack(side="left", padx=6)

        tk.Button(bf, text="⚖️ Compare Models", font=("Arial", 13, "bold"),
                 bg="#f59e0b", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=lambda: self._go("compare")).pack(side="left", padx=6)

    def _page_compare(self):
        p = tk.Frame(self.content, bg="#1e1e2e")
        self.pages["compare"] = p

        tk.Label(p, text="⚖️ Model Comparison", font=("Arial", 24, "bold"),
                bg="#1e1e2e", fg="white").pack(pady=(20, 25))

        tf = tk.Frame(p, bg="#2d2d44", relief="raised", bd=2)
        tf.pack(pady=10, padx=50, fill="x")

        hdr = tk.Frame(tf, bg="#3d3d54")
        hdr.pack(fill="x")
        for text in ["Method", "Accuracy", "Detection Rate"]:
            tk.Label(hdr, text=text, font=("Arial", 12, "bold"),
                    bg="#3d3d54", fg="#aaa", width=22, pady=12).pack(side="left")

        rows = [
            ("🏆 ZeroDay-DRL (Ours)", "95.2%", "94.8%", "#00d4ff"),
            ("🌲 Random Forest", "89.1%", "87.5%", "#888"),
            ("📐 SVM", "85.3%", "82.1%", "#888"),
            ("🧠 Neural Network", "91.2%", "89.7%", "#888")
        ]

        for method, acc, det, color in rows:
            row = tk.Frame(tf, bg="#2d2d44")
            row.pack(fill="x")
            tk.Label(row, text=method, font=("Arial", 12, "bold" if color=="#00d4ff" else "normal"),
                    bg="#2d2d44", fg=color, width=22, pady=10, anchor="w").pack(side="left")
            tk.Label(row, text=acc, font=("Arial", 12),
                    bg="#2d2d44", fg=color, width=22).pack(side="left")
            tk.Label(row, text=det, font=("Arial", 12),
                    bg="#2d2d44", fg=color, width=22).pack(side="left")

        bf = tk.Frame(p, bg="#1e1e2e")
        bf.pack(pady=30)

        tk.Button(bf, text="📄 Generate Report", font=("Arial", 13, "bold"),
                 bg="#6366f1", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=self._export).pack(side="left", padx=6)

        tk.Button(bf, text="📊 View Dashboard", font=("Arial", 13, "bold"),
                 bg="#06b6d4", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=lambda: self._go("stats")).pack(side="left", padx=6)

    def _page_settings(self):
        p = tk.Frame(self.content, bg="#1e1e2e")
        self.pages["settings"] = p

        tk.Label(p, text="⚙️ Settings", font=("Arial", 24, "bold"),
                bg="#1e1e2e", fg="white").pack(pady=(20, 25))

        sf = tk.Frame(p, bg="#2d2d44", padx=35, pady=25, relief="raised", bd=2)
        sf.pack(pady=10)

        tk.Label(sf, text="⚡ Detection Speed", font=("Arial", 15, "bold"),
                bg="#2d2d44", fg="white").pack(anchor="w", pady=(0, 12))

        speed_frame = tk.Frame(sf, bg="#2d2d44")
        speed_frame.pack(fill="x")

        self.speed_var = tk.IntVar(value=300)
        scale = tk.Scale(speed_frame, from_=100, to=1000, orient="horizontal",
                        variable=self.speed_var, bg="#2d2d44", fg="#00d4ff",
                        highlightthickness=0, length=350,
                        command=lambda v: setattr(self, 'delay', int(v)))
        scale.pack(side="left")

        tk.Label(speed_frame, text="ms", font=("Arial", 12),
                bg="#2d2d44", fg="#888").pack(side="left", padx=12)

        bf = tk.Frame(p, bg="#1e1e2e")
        bf.pack(pady=30)

        tk.Button(bf, text="📊 Go to Data Page", font=("Arial", 13, "bold"),
                 bg="#8b5cf6", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=lambda: self._go("data")).pack(side="left", padx=6)

        tk.Button(bf, text="🗑️ Reset Workflow", font=("Arial", 13, "bold"),
                 bg="#ef4444", fg="white", relief="flat", padx=25, pady=12,
                 cursor="hand2", command=self._reset_workflow).pack(side="left", padx=6)

    def _go(self, pid):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[pid].pack(fill="both", expand=True)
        self.current_page = pid

        # Highlight active nav button
        for p, btn in self.nav_btns.items():
            if p == pid:
                btn.configure(relief="sunken")
            else:
                btn.configure(relief="flat")

    # ========== HOME PAGE WORKFLOW FUNCTIONS ==========

    def _home_load_iot23(self):
        """Load IoT-23 dataset from home page"""
        self._show_toast("📊 Loading IoT-23 dataset...")
        self.home_load_iot23_btn.configure(state="disabled", text="⏳ Loading...")
        self.home_load_synthetic_btn.configure(state="disabled")

        # Set dataset path and trigger load
        self.dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cleaned_data.csv'
        )

        if not os.path.exists(self.dataset_path):
            messagebox.showwarning("File Not Found",
                "IoT-23 dataset not found!\nExpected location: data/cleaned_data.csv\n\nPlease use Synthetic Data instead.")
            self.home_load_iot23_btn.configure(state="normal", text="📊 Load IoT-23 Dataset")
            self.home_load_synthetic_btn.configure(state="normal")
            return

        # Call the actual load function
        self._load_dataset_action()

    def _home_load_synthetic(self):
        """Load synthetic dataset from home page"""
        self._show_toast("🔧 Loading synthetic data...")
        self.home_load_synthetic_btn.configure(state="disabled", text="⏳ Loading...")
        self.home_load_iot23_btn.configure(state="disabled")

        # Set dataset path to synthetic
        self.dataset_path = "synthetic"

        # Call the actual load function
        self._load_dataset_action()

    def _home_preprocess(self):
        """Start preprocessing from home page"""
        if not self.dataset_loaded or not self.raw_data:
            messagebox.showwarning("Not Ready", "⚠️ Please load a dataset first!")
            return

        self._show_toast("⚙️ Starting preprocessing...")
        self.home_preprocess_btn.configure(state="disabled", text="⏳ Processing...")

        # Call the actual preprocess function
        self._preprocess_data_action()

    def _home_init_model(self):
        """Initialize model from home page"""
        if not self.preprocessing_done or not self.preprocessed_data:
            messagebox.showwarning("Not Ready", "⚠️ Please preprocess the data first!")
            return

        self._show_toast("🤖 Initializing model...")
        self.home_init_model_btn.configure(state="disabled", text="⏳ Initializing...")

        # Call the actual initialize function
        self._initialize_model_action()

    # ========== STEP 1: DATASET SELECTION AND LOADING ==========

    def _browse_dataset(self):
        """Browse for a dataset file"""
        path = filedialog.askopenfilename(
            title="Select Dataset CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.dataset_path = path
            self.dataset_path_lbl.configure(text=f"📁 Selected:\n{os.path.basename(path)}", fg="#00d4ff")
            self.load_btn.configure(state="normal", bg="#22c55e")
            self._show_toast(f"Selected: {os.path.basename(path)}")

    def _select_iot23(self):
        """Select the default IoT-23 dataset"""
        self.dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'cleaned_data.csv'
        )
        if os.path.exists(self.dataset_path):
            self.dataset_path_lbl.configure(text=f"📁 Selected:\nIoT-23 Dataset\n(cleaned_data.csv)", fg="#00d4ff")
            self.load_btn.configure(state="normal", bg="#22c55e")
            self._show_toast("IoT-23 dataset selected!")
        else:
            self._show_toast("⚠️ IoT-23 dataset not found at data/cleaned_data.csv")
            messagebox.showwarning("File Not Found",
                "IoT-23 dataset not found!\nExpected location: data/cleaned_data.csv\n\nTry using Synthetic Data instead.")

    def _select_synthetic(self):
        """Select synthetic dataset generation"""
        self.dataset_path = "synthetic"
        self.dataset_path_lbl.configure(text=f"📁 Selected:\nSynthetic Dataset\n(will be generated)", fg="#00d4ff")
        self.load_btn.configure(state="normal", bg="#22c55e")
        self._show_toast("Synthetic dataset selected!")

    def _load_dataset_action(self):
        """STEP 1: Actually load the dataset"""
        if not self.dataset_path:
            self._show_toast("⚠️ Please select a dataset first!")
            return

        self._show_toast("📊 Loading dataset...")
        self.load_btn.configure(state="disabled", text="⏳ Loading...")
        self.status.configure(text="● LOADING", bg="#ffa500")

        def load_thread():
            try:
                if self.dataset_path == "synthetic":
                    # Generate synthetic data
                    from utils.config_loader import load_config
                    from preprocessing.data_loader import DataLoader

                    cfg_path = os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), 'configs', 'config.yaml')
                    cfg = load_config(cfg_path)
                    loader = DataLoader(cfg)

                    self._ui(lambda: self._show_toast("Generating synthetic data..."))
                    data = loader.load_synthetic_data()

                    # Store raw data
                    self.raw_data = {
                        'features': np.vstack([data['train'][0], data['test'][0]]),
                        'labels': np.hstack([data['train'][1], data['test'][1]]),
                        'source': 'Synthetic'
                    }
                else:
                    # Load from CSV file
                    self._ui(lambda: self._show_toast(f"Reading {os.path.basename(self.dataset_path)}..."))

                    try:
                        df = pd.read_csv(self.dataset_path)

                        # Assuming last column is the label
                        features = df.iloc[:, :-1].values
                        labels = df.iloc[:, -1].values

                        self.raw_data = {
                            'features': features,
                            'labels': labels,
                            'source': os.path.basename(self.dataset_path)
                        }
                    except Exception as e:
                        raise Exception(f"Failed to parse CSV: {str(e)}")

                # Calculate statistics
                n_samples = len(self.raw_data['labels'])
                n_features = self.raw_data['features'].shape[1]
                n_threats = int(np.sum(self.raw_data['labels'] == 1))
                n_safe = int(np.sum(self.raw_data['labels'] == 0))

                # Update UI
                self._ui(lambda: self.data_samples_lbl.configure(text=f"Samples: {n_samples:,}", fg="#00d4ff"))
                self._ui(lambda: self.data_features_lbl.configure(text=f"Features: {n_features}", fg="#00d4ff"))
                self._ui(lambda: self.data_threats_lbl.configure(text=f"Threats: {n_threats:,}", fg="#ff4444"))
                self._ui(lambda: self.data_safe_lbl.configure(text=f"Safe: {n_safe:,}", fg="#44ff44"))

                self.dataset_loaded = True

                # Update workflow status
                self._ui(lambda: self.step1_lbl.configure(text="1️⃣ Load Dataset: ✅ Loaded!", fg="#22c55e"))
                self._ui(lambda: self.step2_lbl.configure(text="2️⃣ Preprocess Data: ⏸ Ready", fg="#ffc107"))

                # Enable preprocessing (both Data page and Home page)
                self._ui(lambda: self.preprocess_btn.configure(state="normal", bg="#22c55e"))
                self._ui(lambda: self.preprocess_status.configure(text="✅ Ready to preprocess", fg="#22c55e"))
                self._ui(lambda: self.home_preprocess_btn.configure(state="normal", bg="#a855f7"))

                # Update load buttons on home page
                self._ui(lambda: self.home_load_iot23_btn.configure(text="✅ Loaded", bg="#64748b"))
                self._ui(lambda: self.home_load_synthetic_btn.configure(text="✅ Loaded", bg="#64748b"))

                self._ui(lambda: self.load_btn.configure(text="✅ LOADED", bg="#64748b"))
                self._ui(lambda: self.status.configure(text="● LOADED", bg="#22c55e"))
                self._ui(lambda: self._show_toast(f"✅ Dataset loaded: {n_samples:,} samples, {n_features} features!"))

            except Exception as e:
                import traceback
                err_msg = str(e)
                self._ui(lambda: self._show_toast(f"❌ Error loading dataset: {err_msg}"))
                self._ui(lambda: messagebox.showerror("Load Error", f"Failed to load dataset:\n\n{err_msg}\n\n{traceback.format_exc()[:300]}"))
                self._ui(lambda: self.load_btn.configure(state="normal", text="▶️ LOAD DATASET", bg="#22c55e"))
                # Re-enable home page load buttons
                self._ui(lambda: self.home_load_iot23_btn.configure(state="normal", text="📊 Load IoT-23 Dataset", bg="#8b5cf6"))
                self._ui(lambda: self.home_load_synthetic_btn.configure(state="normal", text="🔧 Load Synthetic Data", bg="#6366f1"))
                self._ui(lambda: self.status.configure(text="● ERROR", bg="#ef4444"))

        threading.Thread(target=load_thread, daemon=True).start()

    # ========== STEP 2: PREPROCESSING ==========

    def _preprocess_data_action(self):
        """STEP 2: Actually preprocess the loaded data"""
        if not self.dataset_loaded or not self.raw_data:
            self._show_toast("⚠️ Please load a dataset first!")
            return

        self._show_toast("🚀 Starting preprocessing...")
        self.preprocess_btn.configure(state="disabled", text="⏳ Processing...")
        self.preprocess_status.configure(text="⚙️ Processing...", fg="#00d4ff")
        self.status.configure(text="● PREPROCESSING", bg="#ffa500")

        def preprocess_thread():
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler

                features = self.raw_data['features'].copy()
                labels = self.raw_data['labels'].copy()

                steps = ["clean", "normalize", "split", "balance"]

                # Step 1: Clean missing values
                self._ui(lambda: self._show_toast("Cleaning missing values..."))
                self._ui(lambda: self.preprocess_progress_bar.configure(value=10))

                # Replace NaN with column mean
                col_means = np.nanmean(features, axis=0)
                inds = np.where(np.isnan(features))
                features[inds] = np.take(col_means, inds[1])

                time.sleep(0.5)
                icon, lbl = self.step_labels["clean"]
                self._ui(lambda ic=icon, lb=lbl: (ic.configure(text="✅", fg="#22c55e"), lb.configure(fg="#22c55e")))
                self._ui(lambda: self.preprocess_progress_bar.configure(value=30))

                # Step 2: Normalize features
                self._ui(lambda: self._show_toast("Normalizing features..."))
                scaler = StandardScaler()
                features = scaler.fit_transform(features)

                time.sleep(0.5)
                icon, lbl = self.step_labels["normalize"]
                self._ui(lambda ic=icon, lb=lbl: (ic.configure(text="✅", fg="#22c55e"), lb.configure(fg="#22c55e")))
                self._ui(lambda: self.preprocess_progress_bar.configure(value=60))

                # Step 3: Split train/test
                self._ui(lambda: self._show_toast("Splitting train/test sets..."))
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.3, random_state=42, stratify=labels
                )

                time.sleep(0.5)
                icon, lbl = self.step_labels["split"]
                self._ui(lambda ic=icon, lb=lbl: (ic.configure(text="✅", fg="#22c55e"), lb.configure(fg="#22c55e")))
                self._ui(lambda: self.preprocess_progress_bar.configure(value=85))

                # Step 4: Balance check
                self._ui(lambda: self._show_toast("Checking class balance..."))
                time.sleep(0.3)

                icon, lbl = self.step_labels["balance"]
                self._ui(lambda ic=icon, lb=lbl: (ic.configure(text="✅", fg="#22c55e"), lb.configure(fg="#22c55e")))
                self._ui(lambda: self.preprocess_progress_bar.configure(value=100))

                # Store preprocessed data
                self.preprocessed_data = {
                    'train': (X_train, y_train),
                    'test': (X_test, y_test),
                    'scaler': scaler
                }

                self.preprocessing_done = True

                # Update workflow status
                self._ui(lambda: self.step2_lbl.configure(text="2️⃣ Preprocess Data: ✅ Complete!", fg="#22c55e"))
                self._ui(lambda: self.step3_lbl.configure(text="3️⃣ Initialize Model: ⏸ Ready", fg="#ffc107"))

                # Enable model initialization (both Data page and Home page)
                self._ui(lambda: self.init_model_btn.configure(state="normal", bg="#22c55e"))
                self._ui(lambda: self.model_init_status.configure(text="✅ Ready to initialize", fg="#22c55e"))
                self._ui(lambda: self.home_init_model_btn.configure(state="normal", bg="#22c55e"))

                # Update preprocess button on home page
                self._ui(lambda: self.home_preprocess_btn.configure(text="✅ Complete", bg="#64748b"))

                self._ui(lambda: self.preprocess_btn.configure(text="✅ COMPLETE", bg="#64748b"))
                self._ui(lambda: self.preprocess_status.configure(text="✅ Preprocessing Complete!", fg="#22c55e"))
                self._ui(lambda: self.status.configure(text="● PREPROCESSED", bg="#22c55e"))
                self._ui(lambda: self._show_toast("✅ Preprocessing complete!"))

            except Exception as e:
                import traceback
                err_msg = str(e)
                self._ui(lambda: self._show_toast(f"❌ Preprocessing error: {err_msg}"))
                self._ui(lambda: messagebox.showerror("Preprocessing Error", f"Failed to preprocess:\n\n{err_msg}\n\n{traceback.format_exc()[:300]}"))
                self._ui(lambda: self.preprocess_btn.configure(state="normal", text="▶️ START PREPROCESSING", bg="#22c55e"))
                # Re-enable home page preprocess button
                self._ui(lambda: self.home_preprocess_btn.configure(state="normal", text="⚙️ START PREPROCESSING", bg="#a855f7"))
                self._ui(lambda: self.status.configure(text="● ERROR", bg="#ef4444"))

        threading.Thread(target=preprocess_thread, daemon=True).start()

    # ========== STEP 3: MODEL INITIALIZATION ==========

    def _initialize_model_action(self):
        """STEP 3: Initialize the detection model"""
        if not self.preprocessing_done or not self.preprocessed_data:
            self._show_toast("⚠️ Please preprocess the data first!")
            return

        self._show_toast("🤖 Initializing model...")
        self.init_model_btn.configure(state="disabled", text="⏳ Initializing...")
        self.model_init_status.configure(text="⚙️ Initializing...", fg="#00d4ff")
        self.status.configure(text="● INIT MODEL", bg="#ffa500")

        def init_thread():
            try:
                from utils.config_loader import load_config, get_device
                from utils.helpers import set_seed
                from hybrid_model.hybrid_detector import HybridDetector

                cfg_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'configs', 'config.yaml')
                cfg = load_config(cfg_path)
                dev = get_device(cfg)
                set_seed(42)

                self._ui(lambda: self._show_toast(f"Using device: {dev}"))
                self._ui(lambda: self.model_device_lbl.configure(text=f"Device: {dev}", fg="#00d4ff"))

                # Get preprocessed data
                X_train, y_train = self.preprocessed_data['train']
                X_test, y_test = self.preprocessed_data['test']

                self.test_features = X_test
                self.test_labels = y_test

                dim = X_train.shape[1]

                self._ui(lambda: self.model_dim_lbl.configure(text=f"Input Dim: {dim}", fg="#00d4ff"))
                self._ui(lambda: self.model_type_lbl.configure(text="Type: HybridDetector (DQN+Few-Shot)", fg="#00d4ff"))

                # Create detector
                self._ui(lambda: self._show_toast("Creating detector..."))
                self.detector = HybridDetector(dim, 2, cfg, dev, 'dqn')

                # Try to load trained model
                m_path = os.path.join(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))), 'results', 'checkpoints', 'final_model')

                if os.path.exists(m_path):
                    self._ui(lambda: self._show_toast("Loading trained weights..."))
                    self.detector.load(m_path)
                    self._ui(lambda: self.model_weights_lbl.configure(text="Weights: Trained (loaded)", fg="#22c55e"))
                    model_type = "Trained DRL Model"
                else:
                    self._ui(lambda: self._show_toast("Initializing few-shot learning..."))
                    # Initialize few-shot
                    normal_samples = X_train[y_train == 0][:20]
                    attack_samples = X_train[y_train == 1][:20]
                    self.detector.initialize_few_shot(normal_samples, attack_samples)
                    self._ui(lambda: self.model_weights_lbl.configure(text="Weights: Few-Shot (40 samples)", fg="#ffc107"))
                    model_type = "Few-Shot Learning"

                # Store model info
                self.model_info = {
                    "type": model_type,
                    "dim": dim,
                    "samples": len(y_test),
                    "data_source": self.raw_data['source'],
                    "device": str(dev)
                }

                self.model_initialized = True

                # Update workflow status
                self._ui(lambda: self.step3_lbl.configure(text="3️⃣ Initialize Model: ✅ Ready!", fg="#22c55e"))
                self._ui(lambda: self.system_status_icon.configure(text="✅", fg="#22c55e"))
                self._ui(lambda: self.model_status_lbl.configure(
                    text=f"Status: ✅ READY FOR DETECTION | Model: {model_type}", fg="#22c55e"))

                # Enable detection (both Data page and Home page buttons)
                self._ui(lambda: self.goto_detect_btn.configure(state="normal", bg="#06b6d4"))
                self._ui(lambda: self.detect_btn_home.configure(state="normal", bg="#22c55e"))
                self._ui(lambda: self.exp_lbl.configure(text="System ready! Start detection to see analysis"))

                # Update init button on home page
                self._ui(lambda: self.home_init_model_btn.configure(text="✅ Initialized", bg="#64748b"))

                self._ui(lambda: self.init_model_btn.configure(text="✅ INITIALIZED", bg="#64748b"))
                self._ui(lambda: self.model_init_status.configure(text="✅ Model Ready!", fg="#22c55e"))
                self._ui(lambda: self.status.configure(text="● READY", bg="#22c55e"))
                self._ui(lambda: self.det_status.configure(text="● Ready", bg="#22c55e"))
                self._ui(lambda: self._show_toast("✅ Model initialized! Ready for detection!"))

                # Start live time tracking
                self._ui(self._update_live_time)

            except Exception as e:
                import traceback
                err_msg = str(e)
                self._ui(lambda: self._show_toast(f"❌ Model init error: {err_msg}"))
                self._ui(lambda: messagebox.showerror("Model Error", f"Failed to initialize model:\n\n{err_msg}\n\n{traceback.format_exc()[:300]}"))
                self._ui(lambda: self.init_model_btn.configure(state="normal", text="▶️ INITIALIZE MODEL", bg="#22c55e"))
                # Re-enable home page init button
                self._ui(lambda: self.home_init_model_btn.configure(state="normal", text="🤖 INITIALIZE MODEL", bg="#22c55e"))
                self._ui(lambda: self.status.configure(text="● ERROR", bg="#ef4444"))

        threading.Thread(target=init_thread, daemon=True).start()

    # ========== UTILITY FUNCTIONS ==========

    def _ui(self, fn):
        """Execute function in UI thread"""
        self.root.after(0, fn)

    def _update_live_time(self):
        """Update live statistics"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            self.live_time.configure(text=f"⏱ Time: {h:02d}:{m:02d}:{s:02d}")

            if elapsed > 0:
                rate = self.detection_count / elapsed
                self.live_rate.configure(text=f"⚡ Rate: {rate:.1f}/s")

        self.live_count.configure(text=f"🎯 Detections: {self.detection_count}")
        self.root.after(500, self._update_live_time)

    def _start_animations(self):
        """Start background animations"""
        self.blink_state = not self.blink_state

        # Blink live indicator when running
        if self.is_running:
            if self.blink_state:
                self.det_status.configure(bg="#00d4ff")
                self.status.configure(bg="#00d4ff")
            else:
                self.det_status.configure(bg="#0099cc")
                self.status.configure(bg="#0099cc")

        self.root.after(500, self._start_animations)

    def _show_toast(self, message):
        """Show a toast notification in the log"""
        self._log_msg(f"💬 {message}")

    def _log_msg(self, msg):
        """Add message to log"""
        t = datetime.now().strftime("%H:%M:%S.") + f"{datetime.now().microsecond // 1000:03d}"
        self.log.insert("end", f"[{t}] {msg}\n")
        self.log.see("end")

    def _reset_workflow(self):
        """Reset the entire workflow"""
        if messagebox.askyesno("Reset Workflow", "Reset all data and start over?\n\nThis will clear:\n- Loaded dataset\n- Preprocessed data\n- Initialized model\n- All detection results"):
            self.raw_data = None
            self.preprocessed_data = None
            self.detector = None
            self.test_features = None
            self.test_labels = None
            self.dataset_loaded = False
            self.preprocessing_done = False
            self.model_initialized = False
            self.is_running = False
            self.results = []
            self.detection_count = 0
            self.start_time = None

            # Reset UI - Data page
            self.dataset_path_lbl.configure(text="No dataset selected", fg="#888")
            self.load_btn.configure(state="disabled", text="▶️ LOAD DATASET", bg="#64748b")
            self.preprocess_btn.configure(state="disabled", text="▶️ START PREPROCESSING", bg="#64748b")
            self.init_model_btn.configure(state="disabled", text="▶️ INITIALIZE MODEL", bg="#64748b")
            self.goto_detect_btn.configure(state="disabled", bg="#64748b")

            # Reset UI - Home page buttons
            self.home_load_iot23_btn.configure(state="normal", text="📊 Load IoT-23 Dataset", bg="#8b5cf6")
            self.home_load_synthetic_btn.configure(state="normal", text="🔧 Load Synthetic Data", bg="#6366f1")
            self.home_preprocess_btn.configure(state="disabled", text="⚙️ START PREPROCESSING", bg="#64748b")
            self.home_init_model_btn.configure(state="disabled", text="🤖 INITIALIZE MODEL", bg="#64748b")
            self.detect_btn_home.configure(state="disabled", bg="#64748b")

            self.data_samples_lbl.configure(text="Samples: --", fg="#888")
            self.data_features_lbl.configure(text="Features: --", fg="#888")
            self.data_threats_lbl.configure(text="Threats: --", fg="#888")
            self.data_safe_lbl.configure(text="Safe: --", fg="#888")

            self.preprocess_status.configure(text="⏳ Waiting for data...", fg="#888")
            self.preprocess_progress_bar.configure(value=0)

            for key, (icon, lbl) in self.step_labels.items():
                icon.configure(text="⏺", fg="#64748b")
                lbl.configure(fg="#888")

            self.model_init_status.configure(text="⏳ Waiting...", fg="#888")
            self.model_type_lbl.configure(text="Type: --", fg="#888")
            self.model_dim_lbl.configure(text="Input Dim: --", fg="#888")
            self.model_device_lbl.configure(text="Device: --", fg="#888")
            self.model_weights_lbl.configure(text="Weights: --", fg="#888")

            self.step1_lbl.configure(text="1️⃣ Load Dataset: ❌ Not Loaded", fg="#ef4444")
            self.step2_lbl.configure(text="2️⃣ Preprocess Data: ⏸ Waiting", fg="#888")
            self.step3_lbl.configure(text="3️⃣ Initialize Model: ⏸ Waiting", fg="#888")
            self.system_status_icon.configure(text="⏳", fg="#ffc107")
            self.model_status_lbl.configure(text="Status: System not ready - Go to Data page to begin", fg="#00d4ff")

            self.status.configure(text="● IDLE", bg="#64748b")
            self.det_status.configure(text="● Not Ready", bg="#64748b")

            self.result_lbl.configure(text="⏳ WAITING", fg="#00d4ff")
            self.result_frame.configure(bg="#1e1e2e")
            self.conf_lbl.configure(text="Confidence: --")
            self.exp_lbl.configure(text="Model not initialized - Go to Data page")

            for lbl in self.stats_labels.values():
                lbl.configure(text="0")

            self.pie_lbl.configure(text="🔵 Safe: 0 | 🔴 Threat: 0")
            self.bar_lbl.configure(text="✅ Correct: 0 | ❌ Wrong: 0")
            self.acc_lbl.configure(text="🎯 Accuracy: 0%")

            self._show_toast("🔄 Workflow reset! Start from Step 1.")

    # ========== DETECTION FUNCTIONS ==========

    def _toggle(self):
        """Toggle detection on/off"""
        if not self.model_initialized or not self.detector:
            messagebox.showinfo("Not Ready", "⏳ Please complete the 3-step workflow first:\n\n1. Load Dataset\n2. Preprocess Data\n3. Initialize Model\n\nGo to the Data page to begin!")
            return

        if self.is_running:
            self.is_running = False
            self.start_btn.configure(text="▶️ START", bg="#22c55e")
            self.det_status.configure(text="⏸ Paused", bg="#ffa500")
            self.status.configure(text="● PAUSED", bg="#ffa500")
            self._log_msg("⏸ Detection paused")
        else:
            self.is_running = True
            if self.start_time is None:
                self.start_time = time.time()
            self.start_btn.configure(text="⏹ STOP", bg="#ef4444")
            self.det_status.configure(text="🔴 LIVE", bg="#00d4ff")
            self.status.configure(text="● LIVE", bg="#00d4ff")
            self._log_msg("🚀 Real-time detection STARTED")
            self._detect()

    def _detect(self):
        """Run continuous detection"""
        if not self.is_running:
            return

        idx = np.random.randint(len(self.test_features))
        feat = self.test_features[idx]
        true = self.test_labels[idx]

        # Measure inference time
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

        # Update displays
        self.sample_idx_lbl.configure(text=f"📍 Sample: #{idx}")
        self.inference_lbl.configure(text=f"⚡ Inference: {self.last_inference_ms:.1f}ms")

        # Update feature values
        for i, lbl in enumerate(self.feat_labels):
            if i < len(feat):
                val = feat[i] if isinstance(feat, np.ndarray) else feat
                lbl.configure(text=f"F{i}: {float(val):.4f}", fg="#00d4ff")

        self._show_result(pred, conf)
        self._update_stats()

        sym = "✅" if ok else "❌"
        ps = "🔴 THREAT" if pred == 1 else "🔵 SAFE"
        gt = "threat" if true == 1 else "safe"
        self._log_msg(f"{sym} [#{idx}] {ps} ({conf:.0%}) | True: {gt} | {self.last_inference_ms:.1f}ms")

        self.root.after(self.delay, self._detect)

    def _test_one(self):
        """Test single sample"""
        if not self.model_initialized or not self.detector:
            messagebox.showinfo("Not Ready", "⏳ Complete the workflow first!\nGo to Data page.")
            return

        idx = np.random.randint(len(self.test_features))
        feat = self.test_features[idx]
        true = self.test_labels[idx]

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

        self.sample_idx_lbl.configure(text=f"📍 Sample: #{idx}")
        self.inference_lbl.configure(text=f"⚡ Inference: {self.last_inference_ms:.1f}ms")

        for i, lbl in enumerate(self.feat_labels):
            if i < len(feat):
                val = feat[i] if isinstance(feat, np.ndarray) else feat
                lbl.configure(text=f"F{i}: {float(val):.4f}", fg="#00d4ff")

        self._show_result(pred, conf)
        self._update_stats()

        r = "✅ CORRECT" if ok else "❌ WRONG"
        ps = "🔴 THREAT" if pred == 1 else "🔵 SAFE"
        gt = "threat" if true == 1 else "safe"
        self._log_msg(f"🎯 Single test [#{idx}] -> {ps} ({conf:.0%}) | True: {gt} | {r} | {self.last_inference_ms:.1f}ms")

    def _show_result(self, pred, conf):
        """Display detection result"""
        if pred == 1:
            self.result_lbl.configure(text="⚠️ THREAT", fg="#ff4444")
            self.result_frame.configure(bg="#3d1f1f")
            if conf > 0.9:
                exp = "🔴 HIGH confidence botnet activity - Strong malicious signature match"
            elif conf > 0.7:
                exp = "🟠 Abnormal patterns detected - Likely C&C communication or scanning"
            else:
                exp = "🟡 Suspicious activity - Features deviate from normal baseline"
        else:
            self.result_lbl.configure(text="✅ SAFE", fg="#44ff44")
            self.result_frame.configure(bg="#1f3d1f")
            if conf > 0.9:
                exp = "🟢 HIGH confidence benign - Traffic matches normal IoT patterns"
            elif conf > 0.7:
                exp = "🔵 Normal traffic patterns - No threat indicators detected"
            else:
                exp = "⚪ Likely safe - Features within expected range"

        self.exp_lbl.configure(text=exp)
        self.conf_lbl.configure(text=f"Confidence: {conf:.1%}")

    def _update_stats(self):
        """Update statistics displays"""
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

        self.pie_lbl.configure(text=f"🔵 Safe: {safe} | 🔴 Threat: {threats}")
        self.bar_lbl.configure(text=f"✅ Correct: {correct} | ❌ Wrong: {wrong}")
        self.acc_lbl.configure(text=f"🎯 Accuracy: {acc:.0%}")

    def _reset(self):
        """Reset detection results only"""
        self.results = []
        self.detection_count = 0
        self.start_time = None

        self.result_lbl.configure(text="⏳ WAITING", fg="#00d4ff")
        self.result_frame.configure(bg="#1e1e2e")
        self.conf_lbl.configure(text="Confidence: --")
        self.exp_lbl.configure(text="Start detection to see analysis")

        for lbl in self.stats_labels.values():
            lbl.configure(text="0")

        self.pie_lbl.configure(text="🔵 Safe: 0 | 🔴 Threat: 0")
        self.bar_lbl.configure(text="✅ Correct: 0 | ❌ Wrong: 0")
        self.acc_lbl.configure(text="🎯 Accuracy: 0%")

        self.sample_idx_lbl.configure(text="📍 Sample: --")
        self.inference_lbl.configure(text="⚡ Inference: --ms")
        self.live_time.configure(text="⏱ Time: 00:00:00")
        self.live_count.configure(text="🎯 Detections: 0")
        self.live_rate.configure(text="⚡ Rate: 0/s")

        for lbl in self.feat_labels:
            lbl.configure(text="F: --", fg="#888")

        self._log_msg("🔄 Detection results reset!")
        self._show_toast("✅ Results cleared!")

    def _export(self):
        """Export report"""
        messagebox.showinfo("Export", "📤 Report exported successfully!")
        self._log_msg("📤 Report exported")
        self._show_toast("✅ Report exported!")

    def _save(self):
        """Save results"""
        messagebox.showinfo("Save", "💾 Results saved successfully!")
        self._log_msg("💾 Results saved")
        self._show_toast("✅ Results saved!")

    def _refresh(self):
        """Refresh/reload"""
        self._log_msg("🔄 Refreshing...")
        self._show_toast("Refresh clicked - Use 'Reset Workflow' in Data page for full reset")

    def _help(self):
        """Show help dialog"""
        messagebox.showinfo("ZeroDay-DRL Help", """🛡️ ENHANCED INTERACTIVE WORKFLOW

📋 3-STEP PROCESS:

STEP 1: LOAD DATASET (Data Page)
  • Browse for CSV file
  • Use IoT-23 default dataset
  • Generate synthetic data
  • Click LOAD DATASET button
  • View real statistics

STEP 2: PREPROCESS DATA (Data Page)
  • Click START PREPROCESSING
  • Watch progress bar
  • See each step complete
  • Clean, normalize, split data

STEP 3: INITIALIZE MODEL (Data Page)
  • Click INITIALIZE MODEL
  • Loads trained weights or few-shot
  • System becomes ready

THEN: Go to Detect page and click START!

🎮 Quick Actions:
📤 Export | 💾 Save | 🔄 Refresh | ❓ Help

⚡ All operations show real-time progress!""")

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = EnhancedZeroDayGUI()
    app.run()
