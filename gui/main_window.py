"""
ZeroDay-DRL GUI Application
Modern, attractive interface for IoT botnet detection.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import queue
import os
import sys
import numpy as np
from typing import Dict, Optional
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ZeroDayDRLApp:
    """
    Main GUI Application for ZeroDay-DRL.
    """

    def __init__(self, config: Dict, output_dir: str = 'results'):
        """
        Initialize the application.

        Args:
            config: Configuration dictionary
            output_dir: Output directory
        """
        self.config = config
        self.output_dir = output_dir

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create main window
        self.root = ctk.CTk()
        self.root.title("ZeroDay-DRL - IoT Botnet Detection System")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # State variables
        self.detector = None
        self.data_loader = None
        self.is_training = False
        self.is_detecting = False
        self.message_queue = queue.Queue()

        # Build UI
        self._build_ui()

        # Start message processor
        self._process_messages()

    def _build_ui(self):
        """Build the main user interface."""
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=0)  # Header
        self.root.grid_rowconfigure(1, weight=1)  # Main content
        self.root.grid_rowconfigure(2, weight=0)  # Status bar

        # Header
        self._create_header()

        # Main content area with tabs
        self._create_main_content()

        # Status bar
        self._create_status_bar()

    def _create_header(self):
        """Create the header section."""
        header_frame = ctk.CTkFrame(self.root, height=80, corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)

        # Logo/Title
        title_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_frame.grid(row=0, column=0, padx=20, pady=15)

        title_label = ctk.CTkLabel(
            title_frame,
            text="ZeroDay-DRL",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#00D4FF"
        )
        title_label.pack(side="left")

        subtitle_label = ctk.CTkLabel(
            title_frame,
            text="   Hybrid DRL + Few-Shot IoT Botnet Detection",
            font=ctk.CTkFont(size=14),
            text_color="#888888"
        )
        subtitle_label.pack(side="left", padx=10)

        # Quick actions
        actions_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        actions_frame.grid(row=0, column=2, padx=20, pady=15)

        self.quick_train_btn = ctk.CTkButton(
            actions_frame,
            text="Quick Train",
            width=120,
            command=self._quick_train,
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.quick_train_btn.pack(side="left", padx=5)

        self.quick_detect_btn = ctk.CTkButton(
            actions_frame,
            text="Start Detection",
            width=120,
            command=self._toggle_detection,
            fg_color="#007bff",
            hover_color="#0056b3"
        )
        self.quick_detect_btn.pack(side="left", padx=5)

    def _create_main_content(self):
        """Create the main content area with tabs."""
        # Tab view
        self.tabview = ctk.CTkTabview(self.root, corner_radius=10)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)

        # Add tabs
        self.tabview.add("Dashboard")
        self.tabview.add("Training")
        self.tabview.add("Detection")
        self.tabview.add("Analysis")
        self.tabview.add("Settings")

        # Build each tab
        self._build_dashboard_tab()
        self._build_training_tab()
        self._build_detection_tab()
        self._build_analysis_tab()
        self._build_settings_tab()

    def _build_dashboard_tab(self):
        """Build the dashboard tab."""
        tab = self.tabview.tab("Dashboard")
        tab.grid_columnconfigure((0, 1, 2), weight=1)
        tab.grid_rowconfigure((0, 1), weight=1)

        # Metrics cards
        metrics_frame = ctk.CTkFrame(tab)
        metrics_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Create metric cards
        self.metric_cards = {}
        metrics = [
            ("Accuracy", "0.00%", "#00D4FF"),
            ("Detection Rate", "0.00%", "#28a745"),
            ("False Positive Rate", "0.00%", "#dc3545"),
            ("Samples Processed", "0", "#ffc107")
        ]

        for i, (name, value, color) in enumerate(metrics):
            card = self._create_metric_card(metrics_frame, name, value, color)
            card.grid(row=0, column=i, padx=10, pady=10, sticky="ew")
            self.metric_cards[name] = card

        # Detection log
        log_frame = ctk.CTkFrame(tab)
        log_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        log_label = ctk.CTkLabel(
            log_frame,
            text="Detection Log",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        log_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.detection_log = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word"
        )
        self.detection_log.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # System status
        status_frame = ctk.CTkFrame(tab)
        status_frame.grid(row=1, column=2, sticky="nsew", padx=10, pady=10)
        status_frame.grid_columnconfigure(0, weight=1)

        status_label = ctk.CTkLabel(
            status_frame,
            text="System Status",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        status_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.status_indicators = {}
        indicators = [
            ("Model Status", "Not Loaded"),
            ("DRL Agent", "Inactive"),
            ("Few-Shot Module", "Inactive"),
            ("Detection Mode", "Hybrid"),
            ("Adaptations", "0")
        ]

        for i, (name, value) in enumerate(indicators):
            indicator = self._create_status_indicator(status_frame, name, value)
            indicator.grid(row=i+1, column=0, padx=15, pady=5, sticky="ew")
            self.status_indicators[name] = indicator

    def _build_training_tab(self):
        """Build the training tab."""
        tab = self.tabview.tab("Training")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=2)
        tab.grid_rowconfigure(1, weight=1)

        # Training configuration
        config_frame = ctk.CTkFrame(tab)
        config_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10, pady=10)

        config_label = ctk.CTkLabel(
            config_frame,
            text="Training Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        config_label.pack(padx=15, pady=15, anchor="w")

        # Episodes
        episodes_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        episodes_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(episodes_frame, text="Training Episodes:").pack(anchor="w")
        self.episodes_entry = ctk.CTkEntry(episodes_frame, placeholder_text="500")
        self.episodes_entry.pack(fill="x", pady=5)
        self.episodes_entry.insert(0, str(self.config['training']['num_episodes']))

        # Learning rate
        lr_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        lr_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(lr_frame, text="Learning Rate:").pack(anchor="w")
        self.lr_entry = ctk.CTkEntry(lr_frame, placeholder_text="0.001")
        self.lr_entry.pack(fill="x", pady=5)
        self.lr_entry.insert(0, str(self.config['drl']['learning_rate']))

        # DRL Algorithm
        algo_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        algo_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(algo_frame, text="DRL Algorithm:").pack(anchor="w")
        self.algo_var = ctk.StringVar(value="DQN")
        self.algo_menu = ctk.CTkOptionMenu(
            algo_frame,
            variable=self.algo_var,
            values=["DQN", "PPO"]
        )
        self.algo_menu.pack(fill="x", pady=5)

        # Few-shot config
        fs_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        fs_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(fs_frame, text="K-Shot Samples:").pack(anchor="w")
        self.kshot_entry = ctk.CTkEntry(fs_frame, placeholder_text="5")
        self.kshot_entry.pack(fill="x", pady=5)
        self.kshot_entry.insert(0, str(self.config['few_shot']['k_shot']))

        # Buttons
        btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=20)

        self.train_btn = ctk.CTkButton(
            btn_frame,
            text="Start Training",
            command=self._start_training,
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.train_btn.pack(fill="x", pady=5)

        self.stop_train_btn = ctk.CTkButton(
            btn_frame,
            text="Stop Training",
            command=self._stop_training,
            fg_color="#dc3545",
            hover_color="#c82333",
            state="disabled"
        )
        self.stop_train_btn.pack(fill="x", pady=5)

        # Training progress
        progress_frame = ctk.CTkFrame(tab)
        progress_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        progress_frame.grid_columnconfigure(0, weight=1)

        progress_label = ctk.CTkLabel(
            progress_frame,
            text="Training Progress",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        progress_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="Ready to train",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.grid(row=2, column=0, padx=15, pady=5, sticky="w")

        # Training metrics display
        metrics_text_frame = ctk.CTkFrame(progress_frame)
        metrics_text_frame.grid(row=3, column=0, padx=15, pady=10, sticky="ew")

        self.train_metrics_labels = {}
        for i, metric in enumerate(["Episode", "Reward", "Loss", "Accuracy", "Epsilon"]):
            lbl = ctk.CTkLabel(metrics_text_frame, text=f"{metric}: --")
            lbl.grid(row=0, column=i, padx=10, pady=10)
            self.train_metrics_labels[metric] = lbl

        # Training log
        log_frame = ctk.CTkFrame(tab)
        log_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        log_label = ctk.CTkLabel(
            log_frame,
            text="Training Log",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        log_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.training_log = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.training_log.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

    def _build_detection_tab(self):
        """Build the detection tab."""
        tab = self.tabview.tab("Detection")
        tab.grid_columnconfigure((0, 1), weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Detection controls
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        controls_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Detection mode
        mode_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        mode_frame.grid(row=0, column=0, padx=15, pady=10)

        ctk.CTkLabel(mode_frame, text="Detection Mode:", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        self.detection_mode_var = ctk.StringVar(value="Hybrid")
        self.detection_mode_menu = ctk.CTkOptionMenu(
            mode_frame,
            variable=self.detection_mode_var,
            values=["Hybrid", "DRL Only", "Few-Shot Only", "Adaptive"],
            command=self._on_mode_change
        )
        self.detection_mode_menu.pack(fill="x", pady=5)

        # Confidence threshold
        thresh_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        thresh_frame.grid(row=0, column=1, padx=15, pady=10)

        ctk.CTkLabel(thresh_frame, text="Confidence Threshold:", font=ctk.CTkFont(weight="bold")).pack(anchor="w")
        self.threshold_slider = ctk.CTkSlider(
            thresh_frame,
            from_=0.1,
            to=0.9,
            number_of_steps=8
        )
        self.threshold_slider.pack(fill="x", pady=5)
        self.threshold_slider.set(0.7)

        # Start/Stop buttons
        btn_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=2, padx=15, pady=10)

        self.start_detect_btn = ctk.CTkButton(
            btn_frame,
            text="Start Real-time Detection",
            command=self._toggle_detection,
            fg_color="#007bff",
            hover_color="#0056b3"
        )
        self.start_detect_btn.pack(fill="x", pady=5)

        # Load sample button
        load_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        load_frame.grid(row=0, column=3, padx=15, pady=10)

        self.load_sample_btn = ctk.CTkButton(
            load_frame,
            text="Test Single Sample",
            command=self._test_single_sample,
            fg_color="#6c757d",
            hover_color="#545b62"
        )
        self.load_sample_btn.pack(fill="x", pady=5)

        # Detection results
        results_frame = ctk.CTkFrame(tab)
        results_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(1, weight=1)

        results_label = ctk.CTkLabel(
            results_frame,
            text="Detection Results",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        results_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        self.results_text = ctk.CTkTextbox(
            results_frame,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.results_text.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Real-time visualization
        viz_frame = ctk.CTkFrame(tab)
        viz_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        viz_frame.grid_columnconfigure(0, weight=1)
        viz_frame.grid_rowconfigure(1, weight=1)

        viz_label = ctk.CTkLabel(
            viz_frame,
            text="Detection Visualization",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        viz_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        # Detection indicators
        indicator_frame = ctk.CTkFrame(viz_frame)
        indicator_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        indicator_frame.grid_columnconfigure((0, 1), weight=1)
        indicator_frame.grid_rowconfigure((0, 1, 2), weight=1)

        # Large status indicator
        self.detection_status_label = ctk.CTkLabel(
            indicator_frame,
            text="MONITORING",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#00D4FF"
        )
        self.detection_status_label.grid(row=0, column=0, columnspan=2, pady=30)

        # Confidence gauge
        ctk.CTkLabel(indicator_frame, text="Confidence:", font=ctk.CTkFont(size=14)).grid(row=1, column=0, padx=10)
        self.confidence_bar = ctk.CTkProgressBar(indicator_frame, width=200)
        self.confidence_bar.grid(row=1, column=1, padx=10, pady=10)
        self.confidence_bar.set(0)

        # Novelty gauge
        ctk.CTkLabel(indicator_frame, text="Novelty Score:", font=ctk.CTkFont(size=14)).grid(row=2, column=0, padx=10)
        self.novelty_bar = ctk.CTkProgressBar(indicator_frame, width=200, progress_color="#ffc107")
        self.novelty_bar.grid(row=2, column=1, padx=10, pady=10)
        self.novelty_bar.set(0)

    def _build_analysis_tab(self):
        """Build the analysis tab."""
        tab = self.tabview.tab("Analysis")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)

        # Controls
        controls_frame = ctk.CTkFrame(tab)
        controls_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkButton(
            controls_frame,
            text="Generate Report",
            command=self._generate_report
        ).pack(side="left", padx=10, pady=10)

        ctk.CTkButton(
            controls_frame,
            text="Export Metrics",
            command=self._export_metrics
        ).pack(side="left", padx=10, pady=10)

        ctk.CTkButton(
            controls_frame,
            text="Compare with Baselines",
            command=self._compare_baselines
        ).pack(side="left", padx=10, pady=10)

        # Analysis output
        output_frame = ctk.CTkFrame(tab)
        output_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)

        self.analysis_text = ctk.CTkTextbox(
            output_frame,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.analysis_text.pack(fill="both", expand=True, padx=15, pady=15)

    def _build_settings_tab(self):
        """Build the settings tab."""
        tab = self.tabview.tab("Settings")
        tab.grid_columnconfigure(0, weight=1)

        # Appearance
        appearance_frame = ctk.CTkFrame(tab)
        appearance_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            appearance_frame,
            text="Appearance",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=15, pady=10, anchor="w")

        theme_frame = ctk.CTkFrame(appearance_frame, fg_color="transparent")
        theme_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(theme_frame, text="Theme:").pack(side="left", padx=10)
        self.theme_menu = ctk.CTkOptionMenu(
            theme_frame,
            values=["Dark", "Light", "System"],
            command=self._change_theme
        )
        self.theme_menu.pack(side="left", padx=10)
        self.theme_menu.set("Dark")

        # Model paths
        paths_frame = ctk.CTkFrame(tab)
        paths_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(
            paths_frame,
            text="Model Paths",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=15, pady=10, anchor="w")

        checkpoint_frame = ctk.CTkFrame(paths_frame, fg_color="transparent")
        checkpoint_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(checkpoint_frame, text="Checkpoint Path:").pack(anchor="w")
        self.checkpoint_entry = ctk.CTkEntry(checkpoint_frame)
        self.checkpoint_entry.pack(fill="x", pady=5)
        self.checkpoint_entry.insert(0, os.path.join(self.output_dir, 'checkpoints'))

        ctk.CTkButton(
            checkpoint_frame,
            text="Browse",
            width=100,
            command=self._browse_checkpoint
        ).pack(pady=5)

        # Load/Save buttons
        btn_frame = ctk.CTkFrame(tab)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="Load Model",
            command=self._load_model
        ).pack(side="left", padx=10, pady=10)

        ctk.CTkButton(
            btn_frame,
            text="Save Model",
            command=self._save_model
        ).pack(side="left", padx=10, pady=10)

    def _create_status_bar(self):
        """Create the status bar."""
        status_frame = ctk.CTkFrame(self.root, height=30, corner_radius=0)
        status_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.status_label.pack(side="left", padx=20, pady=5)

        self.connection_label = ctk.CTkLabel(
            status_frame,
            text="Model: Not Loaded",
            font=ctk.CTkFont(size=11),
            text_color="#888888"
        )
        self.connection_label.pack(side="right", padx=20, pady=5)

    def _create_metric_card(self, parent, title, value, color):
        """Create a metric display card."""
        card = ctk.CTkFrame(parent, corner_radius=10)

        title_label = ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=12),
            text_color="#888888"
        )
        title_label.pack(pady=(15, 5))

        value_label = ctk.CTkLabel(
            card,
            text=value,
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=color
        )
        value_label.pack(pady=(5, 15))

        card.value_label = value_label
        return card

    def _create_status_indicator(self, parent, name, value):
        """Create a status indicator."""
        frame = ctk.CTkFrame(parent, fg_color="transparent")

        name_label = ctk.CTkLabel(frame, text=name + ":", font=ctk.CTkFont(size=12))
        name_label.pack(side="left")

        value_label = ctk.CTkLabel(
            frame,
            text=value,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00D4FF"
        )
        value_label.pack(side="right")

        frame.value_label = value_label
        return frame

    def _process_messages(self):
        """Process messages from worker threads."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass

        self.root.after(100, self._process_messages)

    def _handle_message(self, msg):
        """Handle a message from a worker thread."""
        msg_type = msg.get('type', '')

        if msg_type == 'log':
            self._add_log(msg.get('text', ''), msg.get('target', 'training'))
        elif msg_type == 'progress':
            self.progress_bar.set(msg.get('value', 0))
            self.progress_label.configure(text=msg.get('text', ''))
        elif msg_type == 'metrics':
            self._update_training_metrics(msg.get('metrics', {}))
        elif msg_type == 'status':
            self.status_label.configure(text=msg.get('text', ''))
        elif msg_type == 'detection':
            self._update_detection_display(msg)
        elif msg_type == 'complete':
            self._on_training_complete(msg)

    def _add_log(self, text, target='training'):
        """Add text to log."""
        timestamp = time.strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {text}\n"

        if target == 'training':
            self.training_log.insert("end", log_text)
            self.training_log.see("end")
        elif target == 'detection':
            self.detection_log.insert("end", log_text)
            self.detection_log.see("end")
        elif target == 'results':
            self.results_text.insert("end", log_text)
            self.results_text.see("end")

    def _update_training_metrics(self, metrics):
        """Update training metrics display."""
        for name, value in metrics.items():
            if name in self.train_metrics_labels:
                if isinstance(value, float):
                    self.train_metrics_labels[name].configure(text=f"{name}: {value:.4f}")
                else:
                    self.train_metrics_labels[name].configure(text=f"{name}: {value}")

    def _update_detection_display(self, result):
        """Update detection visualization."""
        prediction = result.get('prediction', 0)
        confidence = result.get('confidence', 0)
        novelty = result.get('novelty_score', 0)

        # Update status
        if prediction == 1:
            self.detection_status_label.configure(
                text="THREAT DETECTED",
                text_color="#dc3545"
            )
        else:
            self.detection_status_label.configure(
                text="NORMAL",
                text_color="#28a745"
            )

        # Update gauges
        self.confidence_bar.set(confidence)
        self.novelty_bar.set(novelty)

    def _quick_train(self):
        """Quick train with default settings."""
        self._start_training()

    def _start_training(self):
        """Start the training process."""
        if self.is_training:
            return

        self.is_training = True
        self.train_btn.configure(state="disabled")
        self.stop_train_btn.configure(state="normal")

        # Get training parameters
        try:
            num_episodes = int(self.episodes_entry.get())
        except ValueError:
            num_episodes = 500

        algorithm = self.algo_var.get().lower()

        # Start training in background thread
        def training_worker():
            try:
                self.message_queue.put({'type': 'log', 'text': 'Initializing training...', 'target': 'training'})
                self.message_queue.put({'type': 'status', 'text': 'Training in progress...'})

                # Import here to avoid circular imports
                from utils.config_loader import get_device
                from preprocessing.data_loader import DataLoader
                from hybrid_model.hybrid_detector import HybridDetector
                from hybrid_model.trainer import HybridTrainer

                device = get_device(self.config)
                self.data_loader = DataLoader(self.config)

                self.message_queue.put({'type': 'log', 'text': 'Loading data...', 'target': 'training'})

                data = self.data_loader.load_synthetic_data()
                feature_dim = data['train'][0].shape[1]

                self.message_queue.put({'type': 'log', 'text': f'Data loaded. Feature dim: {feature_dim}', 'target': 'training'})

                # Initialize detector
                self.detector = HybridDetector(
                    state_dim=feature_dim,
                    action_dim=2,
                    config=self.config,
                    device=device,
                    drl_algorithm=algorithm
                )

                self.message_queue.put({'type': 'log', 'text': f'Detector initialized with {algorithm.upper()}', 'target': 'training'})

                # Training with progress updates
                trainer = HybridTrainer(self.config, device)

                # Override num_episodes
                self.config['training']['num_episodes'] = min(num_episodes, 100)  # Limit for demo

                results = trainer.train(self.detector, self.data_loader, os.path.join(self.output_dir, 'checkpoints'))

                self.message_queue.put({
                    'type': 'complete',
                    'success': True,
                    'results': results
                })

            except Exception as e:
                self.message_queue.put({
                    'type': 'log',
                    'text': f'Error: {str(e)}',
                    'target': 'training'
                })
                self.message_queue.put({
                    'type': 'complete',
                    'success': False,
                    'error': str(e)
                })

        thread = threading.Thread(target=training_worker, daemon=True)
        thread.start()

    def _stop_training(self):
        """Stop the training process."""
        self.is_training = False
        self.train_btn.configure(state="normal")
        self.stop_train_btn.configure(state="disabled")
        self._add_log("Training stopped by user", 'training')

    def _on_training_complete(self, msg):
        """Handle training completion."""
        self.is_training = False
        self.train_btn.configure(state="normal")
        self.stop_train_btn.configure(state="disabled")

        if msg.get('success'):
            self._add_log("Training completed successfully!", 'training')
            self.connection_label.configure(text="Model: Loaded")
            self.status_label.configure(text="Training complete")
            self.progress_bar.set(1.0)
            self.progress_label.configure(text="Training complete!")
        else:
            self._add_log(f"Training failed: {msg.get('error', 'Unknown error')}", 'training')

    def _toggle_detection(self):
        """Toggle detection mode."""
        if self.is_detecting:
            self.is_detecting = False
            self.quick_detect_btn.configure(text="Start Detection")
            self.start_detect_btn.configure(text="Start Real-time Detection")
            self.detection_status_label.configure(text="MONITORING", text_color="#00D4FF")
        else:
            if self.detector is None:
                messagebox.showwarning("Warning", "Please train or load a model first.")
                return

            self.is_detecting = True
            self.quick_detect_btn.configure(text="Stop Detection")
            self.start_detect_btn.configure(text="Stop Real-time Detection")

            # Start detection simulation
            self._simulate_detection()

    def _simulate_detection(self):
        """Simulate real-time detection."""
        if not self.is_detecting or self.detector is None:
            return

        # Get test sample
        if hasattr(self, 'data_loader') and self.data_loader is not None:
            test_features = self.data_loader.test_features
            test_labels = self.data_loader.test_labels

            if test_features is not None and len(test_features) > 0:
                idx = np.random.randint(len(test_features))
                features = test_features[idx]
                true_label = test_labels[idx]

                result = self.detector.detect(features, training=False)
                result['true_label'] = true_label

                self._update_detection_display(result)

                # Log result
                pred_str = "BOTNET" if result['prediction'] == 1 else "NORMAL"
                true_str = "BOTNET" if true_label == 1 else "NORMAL"
                correct = "CORRECT" if result['prediction'] == true_label else "WRONG"

                self._add_log(
                    f"Pred: {pred_str} | True: {true_str} | Conf: {result['confidence']:.3f} | {correct}",
                    'detection'
                )

        # Schedule next detection
        self.root.after(500, self._simulate_detection)

    def _test_single_sample(self):
        """Test detection on a single sample."""
        if self.detector is None:
            messagebox.showwarning("Warning", "Please train or load a model first.")
            return

        if hasattr(self, 'data_loader') and self.data_loader is not None:
            test_features = self.data_loader.test_features
            test_labels = self.data_loader.test_labels

            if test_features is not None and len(test_features) > 0:
                idx = np.random.randint(len(test_features))
                features = test_features[idx]
                true_label = test_labels[idx]

                result = self.detector.detect(features, training=False)

                # Display result
                self._add_log("\n" + "=" * 40, 'results')
                self._add_log("SINGLE SAMPLE DETECTION RESULT", 'results')
                self._add_log("=" * 40, 'results')
                self._add_log(f"Prediction: {'BOTNET' if result['prediction'] == 1 else 'NORMAL'}", 'results')
                self._add_log(f"True Label: {'BOTNET' if true_label == 1 else 'NORMAL'}", 'results')
                self._add_log(f"Confidence: {result['confidence']:.4f}", 'results')
                self._add_log(f"Detection Source: {result['detection_source']}", 'results')
                self._add_log(f"Novelty Score: {result['novelty_score']:.4f}", 'results')
                self._add_log(f"DRL Confidence: {result['drl_confidence']:.4f}", 'results')
                self._add_log(f"Few-Shot Confidence: {result['few_shot_confidence']:.4f}", 'results')

                self._update_detection_display(result)

    def _on_mode_change(self, mode):
        """Handle detection mode change."""
        if self.detector is not None:
            from hybrid_model.hybrid_detector import DetectionMode
            mode_map = {
                "Hybrid": DetectionMode.HYBRID,
                "DRL Only": DetectionMode.DRL_ONLY,
                "Few-Shot Only": DetectionMode.FEW_SHOT_ONLY,
                "Adaptive": DetectionMode.ADAPTIVE
            }
            self.detector.set_mode(mode_map.get(mode, DetectionMode.HYBRID))

    def _generate_report(self):
        """Generate analysis report."""
        self.analysis_text.delete("1.0", "end")

        if self.detector is None:
            self.analysis_text.insert("1.0", "No model loaded. Please train or load a model first.")
            return

        metrics = self.detector.get_metrics()

        report = []
        report.append("=" * 60)
        report.append("ZERODAY-DRL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("DRL Agent Metrics:")
        report.append("-" * 40)
        for key, value in metrics.get('drl', {}).items():
            report.append(f"  {key}: {value}")

        report.append("")
        report.append("Meta-Learner Metrics:")
        report.append("-" * 40)
        for key, value in metrics.get('meta_learner', {}).items():
            report.append(f"  {key}: {value}")

        report.append("")
        report.append("Hybrid System Metrics:")
        report.append("-" * 40)
        report.append(f"  Mode: {metrics.get('mode', 'N/A')}")
        report.append(f"  Zero-day Detections: {metrics.get('zero_day_detections', 0)}")
        report.append(f"  Adaptation Triggers: {metrics.get('adaptation_triggers', 0)}")
        report.append(f"  Total Detections: {metrics.get('total_detections', 0)}")
        report.append(f"  Average Confidence: {metrics.get('avg_confidence', 0):.4f}")

        self.analysis_text.insert("1.0", "\n".join(report))

    def _export_metrics(self):
        """Export metrics to file."""
        if self.detector is None:
            messagebox.showwarning("Warning", "No model loaded.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filepath:
            self._generate_report()
            content = self.analysis_text.get("1.0", "end")
            with open(filepath, 'w') as f:
                f.write(content)
            messagebox.showinfo("Success", f"Metrics exported to {filepath}")

    def _compare_baselines(self):
        """Compare with baseline models."""
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "Baseline comparison requires trained model.\n")
        self.analysis_text.insert("end", "This feature runs comparison with:\n")
        self.analysis_text.insert("end", "  - Random Forest\n")
        self.analysis_text.insert("end", "  - SVM\n")
        self.analysis_text.insert("end", "  - MLP\n")
        self.analysis_text.insert("end", "  - LSTM\n")
        self.analysis_text.insert("end", "\nPlease run 'python main.py --mode compare' for full comparison.")

    def _change_theme(self, theme):
        """Change application theme."""
        theme_map = {
            "Dark": "dark",
            "Light": "light",
            "System": "system"
        }
        ctk.set_appearance_mode(theme_map.get(theme, "dark"))

    def _browse_checkpoint(self):
        """Browse for checkpoint directory."""
        path = filedialog.askdirectory()
        if path:
            self.checkpoint_entry.delete(0, "end")
            self.checkpoint_entry.insert(0, path)

    def _load_model(self):
        """Load a trained model."""
        checkpoint_path = self.checkpoint_entry.get()

        if not os.path.exists(checkpoint_path):
            messagebox.showerror("Error", f"Path not found: {checkpoint_path}")
            return

        try:
            from utils.config_loader import get_device
            from preprocessing.data_loader import DataLoader
            from hybrid_model.hybrid_detector import HybridDetector

            device = get_device(self.config)
            self.data_loader = DataLoader(self.config)
            data = self.data_loader.load_synthetic_data()
            feature_dim = data['train'][0].shape[1]

            self.detector = HybridDetector(
                state_dim=feature_dim,
                action_dim=2,
                config=self.config,
                device=device
            )

            model_path = os.path.join(checkpoint_path, 'final_model')
            if os.path.exists(model_path):
                self.detector.load(model_path)
                self.connection_label.configure(text="Model: Loaded")
                messagebox.showinfo("Success", "Model loaded successfully!")
            else:
                messagebox.showerror("Error", "No model found at specified path.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def _save_model(self):
        """Save current model."""
        if self.detector is None:
            messagebox.showwarning("Warning", "No model to save.")
            return

        checkpoint_path = self.checkpoint_entry.get()
        os.makedirs(checkpoint_path, exist_ok=True)

        try:
            self.detector.save(os.path.join(checkpoint_path, 'saved_model'))
            messagebox.showinfo("Success", f"Model saved to {checkpoint_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def run(self):
        """Run the application."""
        self.root.mainloop()


if __name__ == "__main__":
    import yaml

    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    app = ZeroDayDRLApp(config)
    app.run()
