"""
ZeroDay-DRL Simple GUI
Easy-to-use interface for IoT Botnet Detection.
Works on Windows and Linux.
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
import threading
import os
import sys
import numpy as np
from datetime import datetime

# Matplotlib for graphs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, get_device
from utils.helpers import set_seed
from hybrid_model.hybrid_detector import HybridDetector


class SimpleZeroDayGUI:
    """
    Simple and clean GUI for ZeroDay-DRL.
    Designed to be easy to use for everyone.
    """

    def __init__(self):
        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Create main window
        self.root = ctk.CTk()
        self.root.title("ZeroDay-DRL - IoT Botnet Detection")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Application state
        self.detector = None
        self.test_features = None
        self.test_labels = None
        self.is_running = False
        self.results = []

        # Build the interface
        self._build_interface()

        # Load model automatically after window opens
        self.root.after(1000, self._load_model_auto)

    def _build_interface(self):
        """Build the user interface with clear, simple layout."""

        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        # === TOP BAR ===
        self._create_top_bar()

        # === MAIN CONTENT ===
        self._create_main_content()

        # === BOTTOM BAR ===
        self._create_bottom_bar()

    def _create_top_bar(self):
        """Create the top bar with title and status."""
        top_bar = ctk.CTkFrame(self.root, height=70, corner_radius=0)
        top_bar.grid(row=0, column=0, sticky="ew")
        top_bar.grid_columnconfigure(1, weight=1)

        # Title
        title = ctk.CTkLabel(
            top_bar,
            text="ZeroDay-DRL",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.grid(row=0, column=0, padx=25, pady=20)

        # Subtitle
        subtitle = ctk.CTkLabel(
            top_bar,
            text="IoT Botnet Detection System",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle.grid(row=0, column=1, padx=10, pady=20, sticky="w")

        # Status indicator
        status_frame = ctk.CTkFrame(top_bar, fg_color="transparent")
        status_frame.grid(row=0, column=2, padx=25, pady=20)

        self.status_light = ctk.CTkLabel(
            status_frame,
            text="●",
            font=ctk.CTkFont(size=20),
            text_color="orange"
        )
        self.status_light.pack(side="left", padx=5)

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="Loading...",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.status_label.pack(side="left", padx=5)

    def _create_main_content(self):
        """Create the main content area."""
        main = ctk.CTkFrame(self.root, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew", padx=15, pady=10)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=2)
        main.grid_rowconfigure(0, weight=1)

        # Left panel - Controls
        self._create_left_panel(main)

        # Right panel - Results and Graphs
        self._create_right_panel(main)

    def _create_left_panel(self, parent):
        """Create left panel with controls and statistics."""
        left = ctk.CTkFrame(parent, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_columnconfigure(0, weight=1)

        # === CONTROL BUTTONS ===
        controls_label = ctk.CTkLabel(
            left,
            text="Controls",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        controls_label.pack(padx=20, pady=(20, 10), anchor="w")

        # Start/Stop Button
        self.start_button = ctk.CTkButton(
            left,
            text="START DETECTION",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=60,
            corner_radius=10,
            fg_color="#28a745",
            hover_color="#1e7e34",
            command=self._toggle_detection
        )
        self.start_button.pack(padx=20, pady=10, fill="x")

        # Test One Sample Button
        self.test_button = ctk.CTkButton(
            left,
            text="Test Single Sample",
            font=ctk.CTkFont(size=14),
            height=45,
            corner_radius=8,
            fg_color="#007bff",
            hover_color="#0056b3",
            command=self._test_one_sample
        )
        self.test_button.pack(padx=20, pady=5, fill="x")

        # Reset Button
        self.reset_button = ctk.CTkButton(
            left,
            text="Reset Results",
            font=ctk.CTkFont(size=14),
            height=45,
            corner_radius=8,
            fg_color="#6c757d",
            hover_color="#545b62",
            command=self._reset_results
        )
        self.reset_button.pack(padx=20, pady=5, fill="x")

        # === CURRENT RESULT ===
        result_label = ctk.CTkLabel(
            left,
            text="Current Result",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        result_label.pack(padx=20, pady=(30, 10), anchor="w")

        result_box = ctk.CTkFrame(left, corner_radius=10, height=120)
        result_box.pack(padx=20, pady=5, fill="x")
        result_box.pack_propagate(False)

        self.result_text = ctk.CTkLabel(
            result_box,
            text="WAITING",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#17a2b8"
        )
        self.result_text.pack(pady=15)

        self.confidence_text = ctk.CTkLabel(
            result_box,
            text="Confidence: --",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.confidence_text.pack()

        # === STATISTICS ===
        stats_label = ctk.CTkLabel(
            left,
            text="Statistics",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        stats_label.pack(padx=20, pady=(30, 10), anchor="w")

        stats_box = ctk.CTkFrame(left, corner_radius=10)
        stats_box.pack(padx=20, pady=5, fill="x")

        # Statistics rows
        self.stats = {}
        stats_items = [
            ("Total Samples", "0", "#ffffff"),
            ("Threats Found", "0", "#dc3545"),
            ("Safe Traffic", "0", "#28a745"),
            ("Accuracy", "0.0%", "#ffc107"),
            ("Detection Rate", "0.0%", "#17a2b8"),
        ]

        for name, value, color in stats_items:
            row = ctk.CTkFrame(stats_box, fg_color="transparent")
            row.pack(fill="x", padx=15, pady=6)

            ctk.CTkLabel(
                row,
                text=name + ":",
                font=ctk.CTkFont(size=13),
                text_color="gray"
            ).pack(side="left")

            val_label = ctk.CTkLabel(
                row,
                text=value,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=color
            )
            val_label.pack(side="right")
            self.stats[name] = val_label

        # === SPEED CONTROL ===
        speed_label = ctk.CTkLabel(
            left,
            text="Detection Speed",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        speed_label.pack(padx=20, pady=(20, 5), anchor="w")

        self.speed_slider = ctk.CTkSlider(
            left,
            from_=100,
            to=1000,
            number_of_steps=9,
            command=self._update_speed
        )
        self.speed_slider.set(300)
        self.speed_slider.pack(padx=20, pady=5, fill="x")

        self.speed_display = ctk.CTkLabel(
            left,
            text="Speed: 300ms per sample",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.speed_display.pack(padx=20, pady=(0, 20))

        self.detection_delay = 300

    def _create_right_panel(self, parent):
        """Create right panel with graphs and log."""
        right = ctk.CTkFrame(parent, corner_radius=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=2)
        right.grid_rowconfigure(3, weight=1)

        # === GRAPHS ===
        graphs_label = ctk.CTkLabel(
            right,
            text="Detection Graphs",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        graphs_label.grid(row=0, column=0, padx=20, pady=(15, 5), sticky="w")

        graph_frame = ctk.CTkFrame(right, corner_radius=10)
        graph_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 4), dpi=100, facecolor='#2b2b2b')
        self.fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

        # Three subplots
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor('#2b2b2b')
            for spine in ax.spines.values():
                spine.set_color('#555555')
            ax.tick_params(colors='#aaaaaa')

        self._init_graphs()

        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # === DETECTION LOG ===
        log_label = ctk.CTkLabel(
            right,
            text="Detection Log",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        log_label.grid(row=2, column=0, padx=20, pady=(15, 5), sticky="w")

        self.log_box = ctk.CTkTextbox(
            right,
            font=ctk.CTkFont(family="Consolas", size=11),
            corner_radius=10,
            height=150
        )
        self.log_box.grid(row=3, column=0, sticky="nsew", padx=15, pady=(5, 15))

    def _create_bottom_bar(self):
        """Create bottom status bar."""
        bottom = ctk.CTkFrame(self.root, height=35, corner_radius=0)
        bottom.grid(row=2, column=0, sticky="ew")

        self.bottom_status = ctk.CTkLabel(
            bottom,
            text="Ready - Click 'START DETECTION' to begin",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.bottom_status.pack(side="left", padx=20, pady=8)

        version = ctk.CTkLabel(
            bottom,
            text="ZeroDay-DRL v1.0",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        version.pack(side="right", padx=20, pady=8)

    def _init_graphs(self):
        """Initialize empty graphs."""
        # Graph 1: Pie chart
        self.ax1.clear()
        self.ax1.pie([1], colors=['#444444'])
        self.ax1.set_title('Detection Results', color='white', fontsize=10, pad=10)

        # Graph 2: Confidence line
        self.ax2.clear()
        self.ax2.set_xlim(0, 50)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title('Confidence Trend', color='white', fontsize=10, pad=10)
        self.ax2.set_xlabel('Sample', color='#888888', fontsize=8)

        # Graph 3: Accuracy bars
        self.ax3.clear()
        self.ax3.bar(['Correct', 'Wrong'], [0, 0], color=['#28a745', '#dc3545'])
        self.ax3.set_title('Predictions', color='white', fontsize=10, pad=10)

        self.canvas.draw() if hasattr(self, 'canvas') else None

    def _load_model_auto(self):
        """Automatically load the trained model."""
        self._log("Starting ZeroDay-DRL system...")

        def worker():
            try:
                # Load configuration
                config_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'configs', 'config.yaml'
                )
                config = load_config(config_path)
                device = get_device(config)
                set_seed(42)

                self._update_ui(lambda: self._log("Loading dataset..."))

                # Load data
                data_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'data', 'cleaned_data.csv'
                )

                if os.path.exists(data_path):
                    from preprocessing.iot23_loader import IoT23DataLoader
                    data_loader = IoT23DataLoader(config, data_dir='data')
                    data = data_loader.load_cleaned_data('cleaned_data.csv')
                    feature_dim = data['train'][0].shape[1]
                    self.test_features = data['test'][0]
                    self.test_labels = data['test'][1]
                    self._update_ui(lambda: self._log(f"Dataset loaded: {len(self.test_labels)} test samples"))
                else:
                    # Use synthetic data
                    from preprocessing.data_loader import DataLoader
                    data_loader = DataLoader(config)
                    data = data_loader.load_synthetic_data()
                    feature_dim = data['train'][0].shape[1]
                    self.test_features = data['test'][0]
                    self.test_labels = data['test'][1]
                    self._update_ui(lambda: self._log("Using synthetic dataset"))

                # Initialize detector
                self.detector = HybridDetector(
                    state_dim=feature_dim,
                    action_dim=2,
                    config=config,
                    device=device,
                    drl_algorithm='dqn'
                )

                # Try to load trained model
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'results', 'checkpoints', 'final_model'
                )

                if os.path.exists(model_path):
                    self.detector.load(model_path)
                    self._update_ui(lambda: self._log("Trained model loaded successfully"))
                else:
                    # Initialize with few-shot
                    normal = data['train'][0][data['train'][1] == 0][:20]
                    attack = data['train'][0][data['train'][1] == 1][:20]
                    self.detector.initialize_few_shot(normal, attack)
                    self._update_ui(lambda: self._log("Model initialized (no trained checkpoint found)"))

                self._update_ui(lambda: self._set_status("ready"))
                self._update_ui(lambda: self._log("System ready - click START DETECTION"))
                self._update_ui(lambda: self.bottom_status.configure(
                    text=f"Model loaded | {len(self.test_labels)} test samples available"
                ))

            except Exception as e:
                self._update_ui(lambda: self._log(f"Error: {str(e)}"))
                self._update_ui(lambda: self._set_status("error"))

        threading.Thread(target=worker, daemon=True).start()

    def _update_ui(self, func):
        """Thread-safe UI update."""
        self.root.after(0, func)

    def _set_status(self, status):
        """Update status indicator."""
        colors = {
            "ready": ("#28a745", "Ready"),
            "running": ("#00d4ff", "Running"),
            "error": ("#dc3545", "Error"),
            "loading": ("#ffc107", "Loading...")
        }
        color, text = colors.get(status, ("#ffc107", "Unknown"))
        self.status_light.configure(text_color=color)
        self.status_label.configure(text=text)

    def _log(self, message):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{timestamp}] {message}\n")
        self.log_box.see("end")

    def _toggle_detection(self):
        """Start or stop continuous detection."""
        if self.detector is None:
            messagebox.showwarning("Not Ready", "Please wait for the model to load.")
            return

        if self.is_running:
            # Stop detection
            self.is_running = False
            self.start_button.configure(
                text="START DETECTION",
                fg_color="#28a745",
                hover_color="#1e7e34"
            )
            self._set_status("ready")
            self._log("Detection stopped")
        else:
            # Start detection
            self.is_running = True
            self.start_button.configure(
                text="STOP DETECTION",
                fg_color="#dc3545",
                hover_color="#c82333"
            )
            self._set_status("running")
            self._log("Continuous detection started")
            self._run_detection()

    def _run_detection(self):
        """Run one detection cycle."""
        if not self.is_running or self.detector is None:
            return

        # Pick random test sample
        idx = np.random.randint(len(self.test_features))
        features = self.test_features[idx]
        true_label = self.test_labels[idx]

        # Detect
        result = self.detector.detect(features, training=False)
        prediction = result['prediction']
        confidence = result['confidence']
        correct = prediction == true_label

        # Store result
        self.results.append({
            'prediction': prediction,
            'true_label': true_label,
            'confidence': confidence,
            'correct': correct
        })

        # Keep last 100
        if len(self.results) > 100:
            self.results = self.results[-100:]

        # Update display
        self._update_result_display(prediction, confidence)
        self._update_statistics()
        self._update_graphs_display()

        # Log
        pred_str = "THREAT" if prediction == 1 else "SAFE"
        true_str = "Threat" if true_label == 1 else "Safe"
        status = "Correct" if correct else "Wrong"
        self._log(f"{pred_str} (True: {true_str}) | Conf: {confidence:.2f} | {status}")

        # Schedule next detection
        self.root.after(self.detection_delay, self._run_detection)

    def _test_one_sample(self):
        """Test a single random sample."""
        if self.detector is None:
            messagebox.showwarning("Not Ready", "Please wait for the model to load.")
            return

        idx = np.random.randint(len(self.test_features))
        features = self.test_features[idx]
        true_label = self.test_labels[idx]

        result = self.detector.detect(features, training=False)
        prediction = result['prediction']
        confidence = result['confidence']
        correct = prediction == true_label

        self.results.append({
            'prediction': prediction,
            'true_label': true_label,
            'confidence': confidence,
            'correct': correct
        })

        self._update_result_display(prediction, confidence)
        self._update_statistics()
        self._update_graphs_display()

        pred_str = "THREAT" if prediction == 1 else "SAFE"
        true_str = "Threat" if true_label == 1 else "Safe"
        status = "CORRECT" if correct else "WRONG"
        self._log(f"Single test: {pred_str} | True: {true_str} | {status}")

    def _update_result_display(self, prediction, confidence):
        """Update the current result display."""
        if prediction == 1:
            self.result_text.configure(text="THREAT", text_color="#dc3545")
        else:
            self.result_text.configure(text="SAFE", text_color="#28a745")

        self.confidence_text.configure(text=f"Confidence: {confidence:.1%}")

    def _update_statistics(self):
        """Update statistics display."""
        if not self.results:
            return

        total = len(self.results)
        threats = sum(1 for r in self.results if r['prediction'] == 1)
        safe = total - threats
        correct = sum(1 for r in self.results if r['correct'])
        accuracy = correct / total if total > 0 else 0

        true_positives = sum(1 for r in self.results
                           if r['prediction'] == 1 and r['true_label'] == 1)
        actual_threats = sum(1 for r in self.results if r['true_label'] == 1)
        detection_rate = true_positives / actual_threats if actual_threats > 0 else 0

        self.stats["Total Samples"].configure(text=str(total))
        self.stats["Threats Found"].configure(text=str(threats))
        self.stats["Safe Traffic"].configure(text=str(safe))
        self.stats["Accuracy"].configure(text=f"{accuracy:.1%}")
        self.stats["Detection Rate"].configure(text=f"{detection_rate:.1%}")

    def _update_graphs_display(self):
        """Update graphs with current data."""
        if not self.results:
            return

        threats = sum(1 for r in self.results if r['prediction'] == 1)
        safe = len(self.results) - threats
        confidences = [r['confidence'] for r in self.results[-50:]]
        correct = sum(1 for r in self.results if r['correct'])
        wrong = len(self.results) - correct

        # Pie chart
        self.ax1.clear()
        self.ax1.set_facecolor('#2b2b2b')
        if threats + safe > 0:
            sizes = [safe, threats]
            colors = ['#28a745', '#dc3545']
            labels = [f'Safe ({safe})', f'Threat ({threats})']
            self.ax1.pie(sizes, labels=labels, colors=colors,
                        autopct='%1.0f%%', startangle=90,
                        textprops={'color': 'white', 'fontsize': 9})
        self.ax1.set_title('Detection Results', color='white', fontsize=10, pad=10)

        # Confidence line chart
        self.ax2.clear()
        self.ax2.set_facecolor('#2b2b2b')
        self.ax2.plot(confidences, color='#00d4ff', linewidth=2)
        self.ax2.fill_between(range(len(confidences)), confidences,
                             alpha=0.3, color='#00d4ff')
        self.ax2.set_ylim(0, 1)
        self.ax2.set_title('Confidence Trend', color='white', fontsize=10, pad=10)
        self.ax2.set_xlabel('Sample', color='#888888', fontsize=8)
        self.ax2.tick_params(colors='#aaaaaa')
        for spine in self.ax2.spines.values():
            spine.set_color('#555555')

        # Bar chart
        self.ax3.clear()
        self.ax3.set_facecolor('#2b2b2b')
        bars = self.ax3.bar(['Correct', 'Wrong'], [correct, wrong],
                           color=['#28a745', '#dc3545'])
        self.ax3.set_title('Predictions', color='white', fontsize=10, pad=10)
        self.ax3.tick_params(colors='#aaaaaa')
        for spine in self.ax3.spines.values():
            spine.set_color('#555555')

        # Add numbers on bars
        for bar, val in zip(bars, [correct, wrong]):
            if val > 0:
                self.ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                             str(val), ha='center', color='white', fontsize=9)

        self.canvas.draw()

    def _reset_results(self):
        """Reset all results and graphs."""
        self.results = []
        self._init_graphs()
        self.result_text.configure(text="WAITING", text_color="#17a2b8")
        self.confidence_text.configure(text="Confidence: --")

        for key in self.stats:
            if key == "Accuracy" or key == "Detection Rate":
                self.stats[key].configure(text="0.0%")
            else:
                self.stats[key].configure(text="0")

        self._log("Results cleared")

    def _update_speed(self, value):
        """Update detection speed."""
        self.detection_delay = int(value)
        self.speed_display.configure(text=f"Speed: {int(value)}ms per sample")

    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = SimpleZeroDayGUI()
    app.run()


if __name__ == "__main__":
    main()
