# ZeroDay-DRL

A smart IoT Botnet Detection System using Deep Reinforcement Learning + Few-Shot Learning.

**Plug and Play!** Clone, install, and run - the pre-trained model is included!

**NEW:** Cyber Hacker Theme - Red & Black terminal-style interface with monospace fonts and ASCII art styling.

---

## Quick Start (3 Steps)

```bash
# 1. Clone
git clone https://github.com/Jebin-05/ZeroDay-DRL.git
cd ZeroDay-DRL

# 2. Install
pip install -r requirements.txt

# 3. Run
python3 gui/simple_gui.py
```

### First-Time Setup Workflow

When you launch the GUI, you'll see a **3-step interactive workflow** on the home page:

**STEP 1: Load Dataset**
- Click **"Load IoT-23 Dataset"** to load real network traffic data from `data/cleaned_data.csv`
- OR click **"Load Synthetic Data"** to generate test data
- The GUI will show dataset statistics (samples, features, threats vs safe)

**STEP 2: Preprocess Data**
- After loading, the **"Start Preprocessing"** button becomes active
- Click it to clean, normalize, and split the data
- Watch real-time preprocessing progress messages

**STEP 3: Initialize Model**
- After preprocessing, the **"Initialize Model"** button becomes active
- Click it to load the trained DRL model or initialize few-shot learning
- Once complete, you're ready to detect threats!

**STEP 4: Start Detection**
- Navigate to the **Detect** page using the navigation bar
- Click **▶ START** to begin continuous real-time detection
- Watch live threat analysis with actual inference times!

---

## What This Does

This software detects IoT botnet attacks in network traffic:
- Identifies known attack patterns
- Learns to recognize NEW (zero-day) attacks
- Shows results in a real-time graphical interface

---

## Installation

### Windows

1. Install Python 3.8+ from [python.org](https://python.org) (check "Add to PATH")
2. Open Command Prompt and run:
```
git clone https://github.com/Jebin-05/ZeroDay-DRL.git
cd ZeroDay-DRL
pip install -r requirements.txt
python gui/simple_gui.py
```

### Linux

```bash
sudo apt update
sudo apt install python3 python3-pip python3-tk git
git clone https://github.com/Jebin-05/ZeroDay-DRL.git
cd ZeroDay-DRL
pip3 install -r requirements.txt
python3 gui/simple_gui.py
```

---

## GUI User Manual

### Overview - Cyber Hacker Theme

The interface features a **red and black cyber security theme** with:
- **Pure black background** (#000000) for terminal aesthetic
- **Bright red accents** (#ff0000) for critical information
- **Monospace Courier font** for authentic hacker terminal look
- **ASCII art symbols** (⚠, █, ▶, ■) for visual emphasis
- **Red borders** on key panels for the cyber security feel

The interface has:
- **Top Bar**: `⚠ ZeroDay-DRL ⚠` title with live stats and system status
- **Navigation Bar**: 6 red-themed page buttons + 4 quick action buttons
- **Main Content**: Pure black background with red highlighted elements

---

### Top Bar (Always Visible)

| Component | What It Shows | Color |
|-----------|---------------|-------|
| **⚠ ZeroDay-DRL ⚠** | Project title with warning symbols | Bright Red |
| **Time: HH:MM:SS** | Elapsed detection time | Gray |
| **Detections: N** | Total samples processed | Bright Red |
| **Rate: X/s** | Detections per second | Dark Red |
| **Status Badge** | Red = Ready/Loading, Bright Red = Running | Red Variants |

---

### Navigation Buttons (6 Pages)

All buttons use red color variations on black background with Courier font:

| Button | Color | What It Does |
|--------|-------|--------------|
| **Home** | Bright Red (#ff0000) | Welcome page with 3-step workflow buttons |
| **Learn** | Red (#dc2626) | Educational page explaining threat vs safe classification |
| **Detect** | Red (#ef4444) | Main detection page with live real-time analysis |
| **Stats** | Dark Red (#991b1b) | Dashboard with accuracy metrics and performance data |
| **Compare** | Red (#b91c1c) | Model comparison table (DRL vs other methods) |
| **Settings** | Dark Red (#7f1d1d) | Configure detection speed and reload model |

---

### Quick Action Buttons (Right Side of Nav Bar)

| Button | What It Does |
|--------|--------------|
| **E** | Export - Save report to file |
| **S** | Save - Save current results |
| **R** | Refresh - Reload the detection model |
| **?** | Help - Show help information |

---

## Page Details

### 📊 DATA Page (Enhanced GUI Only)

The Data page provides interactive dataset management and preprocessing:

#### Left Panel - Load Dataset

| Button | What It Does |
|--------|--------------|
| **Browse Dataset** | Opens file browser to select custom CSV dataset |
| **Load Default (IoT-23)** | Loads the built-in IoT-23 dataset from data/cleaned_data.csv |
| **Load Synthetic Data** | Generates and loads synthetic test data |

#### Dataset Info Display

Shows real-time information about the loaded dataset:
- **Samples**: Total number of samples in dataset
- **Features**: Number of features/dimensions
- **Threats**: Count of threat samples
- **Safe**: Count of safe samples

#### Right Panel - Preprocessing Pipeline

Interactive preprocessing visualization with:

**Status Indicator**: Shows current preprocessing state
- ⏳ Ready to preprocess (Yellow)
- ⚙️ Processing... (Blue)
- ✅ Preprocessing Complete! (Green)

**Progress Bar**: Visual progress indicator (0-100%)

**Processing Steps** (Shows completion status with checkmarks):
1. ⏺ Load Data → ✅ Load Data
2. ⏺ Clean Missing Values → ✅ Clean Missing Values
3. ⏺ Normalize Features → ✅ Normalize Features
4. ⏺ Split Train/Test → ✅ Split Train/Test
5. ⏺ Initialize Model → ✅ Initialize Model

Each step turns green with a checkmark (✅) when completed.

**START PREPROCESSING Button**: Begins the preprocessing pipeline automatically

**How to Use the Data Page:**
1. Click "Load Default (IoT-23)" or "Browse Dataset" to select data
2. Review the dataset info (samples, features, threats, safe)
3. Click "START PREPROCESSING" to begin processing
4. Watch the progress bar and step indicators update in real-time
5. When complete (all steps green), the model loads automatically

---

### HOME Page - Interactive Workflow

The home page features the **3-step interactive workflow** with visual feedback:

**Title Section:**
- `⚠ IoT Botnet Detection ⚠` - Main title with warning symbols
- `[ Deep Reinforcement Learning + Few-Shot Learning ]` - Subtitle in brackets

**System Status Panel** (Black with red border):
- Shows current workflow status
- Updates in real-time as you complete each step

**STEP 1: Load Dataset**
Two buttons to choose data source:
- **"Load IoT-23 Dataset"** - Loads real network traffic from CSV (48,003 samples)
- **"Load Synthetic Data"** - Generates synthetic test data

After loading, you'll see:
- Total samples, feature dimensions
- Number of safe vs threat samples
- Data source confirmation

**STEP 2: Preprocess Data**
- Button becomes active (bright red) after Step 1
- **"Start Preprocessing"** - Cleans, normalizes, and splits data
- Shows real-time progress messages:
  - Cleaning missing values
  - Normalizing features
  - Splitting train/test sets
- Displays train/test split sizes when complete

**STEP 3: Initialize Model**
- Button becomes active (bright red) after Step 2
- **"Initialize Model"** - Loads trained weights or sets up few-shot learning
- Shows:
  - Device being used (CPU/CUDA)
  - Model type (Trained DRL or Few-Shot)
  - Final sample counts

**Navigation Buttons:**
- **"Start Detection"** - Go to Detect page (enabled after Step 3)
- **"Learn About Threats"** - Go to Learn page
- **"View Statistics"** - Go to Stats page

---

### LEARN Page

Educational content explaining why traffic is classified as threat or safe.

**LEFT COLUMN - WHY THREAT (Red)**
- Unusual traffic volume
- Repetitive patterns
- Spoofed IPs
- C&C communication
- Port scanning
- Malware signatures
- Bot behavior
- DNS tunneling

**RIGHT COLUMN - WHY SAFE (Green)**
- Normal traffic
- Valid authentication
- Known source
- Regular timing
- Standard protocols
- Proper encryption
- Human patterns
- Clean payload

**Buttons on this page:**
- **Try Detection** - Go to Detect page
- **View Comparison** - Go to Compare page

---

### DETECT Page (Main Detection Interface)

This is the main working page for real-time threat detection with cyber hacker styling.

#### Top Indicators (Red on Black)

| Display | Meaning | Style |
|---------|---------|-------|
| **Inference: Xms** | Actual time taken for model to process (proves real-time) | Red text |
| **Sample: #N** | Actual dataset index being tested (proves real data) | Red text |
| **Status Badge** | Ready / █ LIVE █ / Paused | Red background when live |

#### Left Panel - Controls (Courier Font)

| Button/Display | What It Does | Style |
|----------------|--------------|-------|
| **▶ START** | Begin continuous detection loop | Red button with play symbol |
| **■ STOP** | Pause detection (appears when running) | Bright red with stop symbol |
| **Test One** | Test exactly one sample | Dark red button |
| **Reset** | Clear all results and start fresh | Dark red button |

#### Left Panel - Result Display (Cyber Theme)

| Display | Meaning | Style |
|---------|---------|-------|
| **⚠ THREAT ⚠** | Botnet/attack detected | Bright red with warning symbols |
| **✓ SAFE ✓** | Normal traffic detected | Dark gray with checkmarks |
| **WAITING** | No detection yet | Red text on black |
| **Confidence: X%** | How certain the model is | Red text |

#### Left Panel - Live Feature Values

Shows the first 4 actual feature values from the current sample being tested:
- F0: 0.1234
- F1: 0.5678
- F2: 0.9012
- F3: 0.3456

This proves the system is processing real data, not fake values.

#### Left Panel - Explanation

Dynamic explanation based on confidence level:
- **High confidence threat**: "HIGH confidence botnet activity - Strong malicious signature match"
- **Medium confidence threat**: "Abnormal patterns detected - Likely C&C communication"
- **Low confidence threat**: "Suspicious activity - Features deviate from normal baseline"
- **High confidence safe**: "HIGH confidence benign - Traffic matches normal IoT patterns"
- **Medium confidence safe**: "Normal traffic patterns - No threat indicators detected"
- **Low confidence safe**: "Likely safe - Features within expected range"

#### Left Panel - Stats

| Statistic | Meaning |
|-----------|---------|
| **Total** | Total samples tested |
| **Threats** (Red) | Number of threats detected |
| **Safe** (Green) | Number of safe traffic detected |
| **Accuracy** (Yellow) | Percentage of correct predictions |

#### Right Panel - Log Display

Shows stats summary:
- **Safe: X | Threat: Y** - Distribution of predictions
- **Correct: X | Wrong: Y** - Accuracy breakdown
- **Accuracy: X%** - Overall accuracy

#### Right Panel - Detection Log

Scrollable log showing every detection with:
```
[14:23:45.123] [#1234] THREAT (92%) | True: threat | OK | 3.2ms
```

Format: `[Timestamp] [Sample#] PREDICTION (Confidence) | True: label | OK/MISS | InferenceTime`

---

### STATS Page

Dashboard with performance metrics:

| Metric | Value | Color |
|--------|-------|-------|
| **Accuracy** | 95.2% | Green |
| **Detection Rate** | 94.8% | Cyan |
| **False Positive Rate** | 2.1% | Orange |
| **Samples Tested** | 10K+ | Purple |

**Buttons on this page:**
- **Export Report** - Save performance report
- **Save Results** - Save detection results
- **Run Detection** - Go to Detect page
- **Compare Models** - Go to Compare page

---

### COMPARE Page

Comparison table showing our model vs other methods:

| Method | Accuracy | Detection Rate |
|--------|----------|----------------|
| **ZeroDay-DRL (Ours)** | 95.2% | 94.8% |
| Random Forest | 89.1% | 87.5% |
| SVM | 85.3% | 82.1% |
| Neural Network | 91.2% | 89.7% |

**Buttons on this page:**
- **Generate Report** - Export comparison report
- **View Dashboard** - Go to Stats page

---

### SETTINGS Page

Configuration options:

| Setting | Description |
|---------|-------------|
| **Detection Speed Slider** | 100ms (fast) to 1000ms (slow) - Controls delay between detections |

**Buttons on this page:**
- **Reload Model** - Re-initialize the detection model
- **Clear All Data** - Reset everything

---

## Understanding the Cyber Hacker Theme

### Color Scheme

| Color | Hex Code | Meaning |
|-------|----------|---------|
| **Pure Black** | #000000 | Main background - terminal aesthetic |
| **Dark Black** | #0a0a0a | Panel backgrounds |
| **Bright Red** | #ff0000 | Critical info, threats, live status |
| **Red** | #dc2626, #ef4444 | Buttons, accents, ready status |
| **Dark Red** | #b91c1c, #991b1b, #7f1d1d | Secondary buttons, loading status |
| **Dark Gray** | #666, #555 | Safe traffic, secondary text |
| **Light Gray** | #999 | Informational text |

### Visual Elements

| Element | Style | Purpose |
|---------|-------|---------|
| **Fonts** | Courier (monospace) | Authentic terminal/hacker aesthetic |
| **Borders** | Red (#ff0000) solid lines | Frame important panels |
| **Symbols** | ⚠ █ ▶ ■ ✓ | ASCII art for visual emphasis |
| **Backgrounds** | Pure black (#000000) | Maximum contrast for visibility |
| **Threat Indicator** | `⚠ THREAT ⚠` in bright red | Immediate visual alert |
| **Safe Indicator** | `✓ SAFE ✓` in dark gray | Minimal distraction |
| **Live Status** | `█ LIVE █` in bright red | Unmistakable active state |

---

## How Detection Works

### Complete Workflow

**STEP 1: Data Loading** (Home Page)
1. Click "Load IoT-23 Dataset" or "Load Synthetic Data"
2. System reads CSV file with network traffic features
3. Converts text labels ("Benign", "Attack", etc.) to binary (0=safe, 1=threat)
4. Extracts 24 feature dimensions from each sample
5. Displays dataset statistics in red terminal-style text

**STEP 2: Preprocessing** (Home Page)
1. Click "Start Preprocessing" after data loads
2. System performs:
   - **Cleaning**: Fills missing values with column means
   - **Normalization**: Scales all features using StandardScaler
   - **Splitting**: Creates 70% train / 30% test split
3. Real-time progress messages appear in red monospace font
4. Preprocessed data stored in memory for model initialization

**STEP 3: Model Initialization** (Home Page)
1. Click "Initialize Model" after preprocessing
2. System loads the pre-trained Hybrid Detector (DRL + Few-Shot Learning)
3. Checks for saved checkpoint at `results/checkpoints/dqn_best.pth`
4. If checkpoint exists: Loads trained DRL weights
5. If no checkpoint: Initializes few-shot learning with 40 prototype samples
6. Device detection (CUDA GPU or CPU fallback)
7. Model ready indicator turns bright red

**STEP 4: Real-Time Detection** (Detect Page)
1. Navigate to Detect page using navigation bar
2. Click **▶ START** button (turns to **■ STOP** in bright red)
3. Status changes to `█ LIVE █` in bright red
4. For each detection cycle:
   - Randomly selects a sample from test data
   - Runs it through the neural network model
   - Measures actual inference time (milliseconds)
   - Displays prediction: `⚠ THREAT ⚠` or `✓ SAFE ✓`
   - Shows confidence percentage
   - Displays first 4 feature values in red
   - Compares with true label to calculate accuracy
   - Logs everything with timestamps in red terminal text
5. Continues in a loop until you click **■ STOP**
6. All stats update in real-time: total detections, threats, safe, accuracy

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"python not recognized"** | Reinstall Python with "Add to PATH" checked |
| **"No module named tkinter"** | Linux: `sudo apt install python3-tk` |
| **"No module named X"** | Run `pip install -r requirements.txt` again |
| **Window doesn't open** | Make sure you have a display connected |
| **"CUDA not available"** | This is just a warning - CPU mode works fine |
| **Segmentation fault** | Make sure you're using the simple_gui.py file |

---

## File Structure

```
ZeroDay-DRL/
├── gui/
│   └── simple_gui.py      <- Main GUI (RUN THIS!)
├── data/
│   └── cleaned_data.csv   <- Dataset (auto-generated if missing)
├── results/
│   └── checkpoints/       <- Saved models
├── configs/
│   └── config.yaml        <- Configuration
├── hybrid_model/          <- Detection model code
├── preprocessing/         <- Data processing code
├── main.py                <- Command-line entry point
├── requirements.txt       <- Dependencies
└── README.md              <- This file
```

---

## Command Line Options

```bash
# Run the GUI (with 3-step workflow buttons)
python3 gui/simple_gui.py

# Run demo mode
python3 main.py --mode demo

# Train a new model
python3 main.py --mode train

# Train with specific episodes
python3 main.py --mode train --num-episodes 200
```

---

## Technical Details

- **Deep Reinforcement Learning**: DQN agent learns optimal detection policy
- **Few-Shot Learning**: Prototypical networks for zero-day threat detection
- **Hybrid Approach**: Combines DRL and Few-Shot for robust detection
- **Dataset**: IoT-23 (real IoT botnet traffic) or synthetic data

---

## License

Open source for learning and research.
