# ZeroDay-DRL

A smart system that detects harmful botnet attacks on IoT devices (like smart cameras, routers, etc.) before they cause damage.

## Pre-Trained Model Included!

**Good news!** This repository comes with a **pre-trained model** ready to use. You do NOT need to train anything - just install and run!

The trained models are located in `results/checkpoints/best_model/` and will be loaded automatically when you run the detection.

---

## Quick Start (Just 3 Steps!)

```bash
# 1. Clone the repository
git clone https://github.com/Jebin-05/ZeroDay-DRL.git
cd ZeroDay-DRL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run detection (choose one):
python main.py --mode demo      # Quick demo
python main.py --mode gui       # Graphical interface
python gui/simple_gui.py        # Simple GUI
```

---

## What Does This Project Do?

This software watches network traffic and identifies if something suspicious is happening. It can:

- Detect known types of attacks
- Learn to recognize NEW types of attacks it has never seen before
- Show results in an easy-to-use graphical interface

---

## Works On

- Windows 10/11
- Linux (Ubuntu, Debian, etc.)
- Python 3.8 or newer

---

## How to Install

### Step 1: Install Python

**Windows:**
1. Go to python.org
2. Download Python 3.11 or 3.12
3. Run the installer
4. IMPORTANT: Check the box that says "Add Python to PATH"
5. Click Install

**Linux:**
```
sudo apt update
sudo apt install python3 python3-pip python3-tk
```

### Step 2: Open Command Prompt or Terminal

**Windows:**
- Press Windows key + R
- Type "cmd" and press Enter

**Linux:**
- Press Ctrl + Alt + T

### Step 3: Go to the Project Folder

```
cd path/to/ZeroDay-DRL
```

For example:
```
cd C:\Users\YourName\Desktop\ZeroDay-DRL     (Windows)
cd ~/Desktop/ZeroDay-DRL                      (Linux)
```

### Step 4: Install Required Packages

**Windows:**
```
pip install -r requirements.txt
```

**Linux:**
```
pip3 install -r requirements.txt
```

This will download and install everything the project needs. Wait for it to finish.

---

## How to Use

### Option 1: Use the Graphical Interface (Easiest)

This is the easiest way. Just run:

**Windows:**
```
python gui/simple_gui.py
```

**Linux:**
```
python3 gui/simple_gui.py
```

A window will open with:
- A big green "START DETECTION" button
- Live graphs showing results
- Statistics about what was detected

Just click the button and watch it work!

### Option 2: Train a New Model

If you want to train the system with your own data:

**Windows:**
```
python main.py --mode train
```

**Linux:**
```
python3 main.py --mode train
```

This will take some time. You will see progress as it learns.

### Option 3: Test the Model

To see how well the trained model works:

**Windows:**
```
python main.py --mode demo
```

**Linux:**
```
python3 main.py --mode demo
```

---

## Understanding the Interface

When you open the graphical interface:

**Left Side:**
- START DETECTION - Click to begin scanning
- Test Single Sample - Test just one sample
- Reset Results - Clear and start over
- Statistics - Shows how many threats found

**Right Side:**
- Graphs showing detection results
- Log showing what was detected

**Colors:**
- Green = Safe/Normal traffic
- Red = Threat/Botnet detected

---

## Project Files Explained

```
ZeroDay-DRL/
|
|-- data/                  <- Your data files go here
|-- results/               <- Training results saved here
|-- configs/config.yaml    <- Settings file
|-- gui/simple_gui.py      <- The graphical interface
|-- main.py                <- Main program
|-- requirements.txt       <- List of required packages
|-- README.md              <- This file
```

---

## Common Problems and Solutions

### Problem: "python is not recognized"

**Solution:** Python is not installed correctly. Reinstall Python and make sure to check "Add Python to PATH" during installation.

### Problem: "No module named tkinter"

**Solution (Linux only):**
```
sudo apt install python3-tk
```

### Problem: "No module named customtkinter"

**Solution:** Run the install command again:
```
pip install -r requirements.txt
```

### Problem: Window does not open

**Solution:** Make sure you have a display connected. On Linux, make sure you are not in a terminal-only mode.

### Problem: "CUDA not available" warning

**Solution:** This is just a warning, not an error. The program will use your CPU instead. It will still work fine.

---

## About the Data

The system works **out of the box** with:
- **Demo mode**: Uses synthetic data for demonstration
- **Pre-trained model**: Already trained and ready to detect threats

If you want to train with real IoT-23 dataset:
1. Download IoT-23 dataset from [Stratosphere IPS](https://www.stratosphereips.org/datasets-iot23)
2. Place the CSV file in the `data/` folder as `iot23_combined.csv`
3. Run training: `python main.py --mode train --data-source iot23`

---

## Training Options

You can customize training with these options:

```
python main.py --mode train --num-episodes 200
```

Options:
- `--num-episodes 100` - How many training rounds (more = better but slower)
- `--data-source iot23` - Use real IoT-23 dataset
- `--data-source synthetic` - Use generated test data

---

## What Each Button Does

| Button | What It Does |
|--------|--------------|
| START DETECTION | Begins continuous scanning of samples |
| STOP DETECTION | Stops the scanning |
| Test Single Sample | Tests just one sample at a time |
| Reset Results | Clears all results and graphs |
| Speed Slider | Controls how fast detection runs |

---

## Understanding Results

**Accuracy:** How often the system is correct (higher is better)

**Detection Rate:** How many actual threats were caught (higher is better)

**Threats Found:** Number of samples identified as attacks

**Safe Traffic:** Number of samples identified as normal

---

## Need Help?

If something is not working:

1. Make sure Python is installed correctly
2. Make sure all packages are installed (run pip install again)
3. Try restarting your computer
4. Check that you are in the correct folder

---

## Technical Details (For Advanced Users)

This system uses:
- Deep Reinforcement Learning (DQN) for decision making
- Few-Shot Learning for adapting to new threats
- Hybrid detection combining both approaches

The model was trained on IoT network traffic data and can detect:
- Port scanning attacks
- Command and control communication
- Botnet behavior patterns

---

## License

This project is open source. You can use it for learning and research.
