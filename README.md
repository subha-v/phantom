# Phantom EEG + Prosthetic System

![Phantom EEG + Prosthetic System](https://cdn.discordapp.com/attachments/847347007785926676/1417746208734052412/image.png?ex=68cb9a8c&is=68ca490c&hm=e09dd4f9082881f99618c73cfdf92bbfa60660cab921c1854220186c7fcbab6d)

**Phantom** is a breakthrough system that tackles **phantom limb pain at multiple levels**—from decoding neural signals to delivering real-time haptic, visual, and voice feedback. By combining **EEG brain signal processing, prosthetic sensing, haptic stimulation, and an AI-powered assistant**, Phantom creates a **closed-loop system that interrupts, reduces, and manages phantom pain** in amputees.  

💡 **Why this matters:**  
- Up to **80% of amputees suffer from phantom limb pain**, one of the most persistent challenges in rehabilitation.  
- A recent *Nature* article (Aug 2025) confirms that **haptic feedback is one of the most reliable treatments** for phantom limb pain.  
- Phantom demonstrates how **affordable, open-source hardware and AI tools** can be combined into a scalable solution with **clinical potential**.  

---

## ✨ What Phantom Does  

- **Detects Brain Signals**: Reads EEG activity from the somatosensory cortex & parietal lobe, where pain and touch are represented.  
- **Localizes Pain Perception**: Maps brain activity to phantom pain/touch sensations in real time.  
- **Triggers Feedback Loops**: Converts those signals into synchronized **haptic (vibration), visual (LED), and audio cues**.  
- **Prosthetic Integration**: A servo-driven prosthetic hand triggers haptic vibration whenever it touches a surface, restoring feedback to the residual limb.  
- **AI Assistant**: A voice-enabled chat system helps users interact with their prosthetic, adjust intensity, and monitor feedback.  

> 🔄 In short: Phantom creates a **seamless closed-loop system** that translates thought → detection → feedback → relief.  


## 🚀 Quick Start - Running the Haptic Feedback System

### Step 1: Install Dependencies
```bash
# Install Arduino server dependencies (run from project root)
npm install express cors serialport @serialport/parser-readline
```

### Step 2: Upload Arduino Sketch
```bash
# Upload the haptic feedback sketch to Arduino
arduino-cli upload --fqbn arduino:renesas_uno:unor4wifi --port /dev/tty.usbmodem* hardware/routines1_test
```

### Step 3: Start the System

#### Option A: Use the Startup Script (Easiest)
```bash
# From project root, run:
./start_haptic_system.sh
```
This starts both servers automatically!

#### Option B: Manual Start (Two Terminals)

**Terminal 1 - Start Arduino Server:**
```bash
# Run from project root directory
node arduino-server.js
```
You should see:
```
Arduino server running on port 3001
Found Arduino at /dev/tty.usbmodem...
Serial port opened successfully
```

**Terminal 2 - Start Dashboard:**
```bash
# Navigate to dashboard and start
cd phantom-dashboard
npm run dev
```

### Step 4: Test Haptic Feedback
1. Open browser at http://localhost:3000
2. Go to the chat interface
3. Type: `play haptic feedback`
4. The Arduino will trigger haptic motor!

## Hardware Setup
- **Board**: Arduino UNO R4 WiFi
- **Haptic Driver**: Adafruit DRV2605
- **Connection**: Haptic driver connected via I2C/Qwiic
- **FQBN**: `arduino:renesas_uno:unor4wifi`

## Prerequisites
- Node.js (v18+)
- Arduino CLI (version 1.3.1+)
- VS Code (optional, for development)

## Available VS Code Tasks

You can run these tasks using `Cmd+Shift+P` → "Tasks: Run Task":

1. **Arduino: Compile** - Compiles the sketch without uploading
2. **Arduino: Upload** - Uploads the compiled sketch to the board
3. **Arduino: Compile and Upload** - Compiles and uploads in sequence
4. **Arduino: List Boards** - Shows connected Arduino boards
5. **Arduino: Stop/Interrupt Task** - Interrupts running tasks

## Keyboard Shortcuts

The following keyboard shortcuts are available to interrupt running tasks:

- **`Ctrl+C`** - Terminates the currently running task
- **`Cmd+Shift+X`** - Alternative shortcut to terminate tasks
- **`Escape`** - Terminates task when terminal is focused

These shortcuts work when any Arduino task is running (compile, upload, etc.).

## Project Structure

```
phantom/
├── .vscode/
│   └── tasks.json              # VS Code task definitions
│   └── keybindings.json        # Keyboard shortcuts
├── arduino-cli.yaml            # Arduino CLI configuration
├── arduino-server.js           # Node.js server for Arduino serial communication
├── whisper-server.py           # Python server for voice transcription
├── hardware/
│   ├── routines1_test/
│   │   └── routines1_test.ino  # Haptic feedback Arduino sketch
│   ├── prosthetic/
│   │   └── proximity_buzzer.ino # Proximity sensing with buzzer feedback
│   ├── led_blink/
│   │   └── led_blink.ino       # Basic LED blink sketch
│   └── rainbow_led/
│       └── rainbow_led.ino     # Rainbow breathing effect
├── model_training/             # EEG signal processing & ML models
│   ├── preprocess.py           # EEG data preprocessing pipeline
│   ├── features.py             # Feature extraction utilities
│   ├── train_model.py          # Binary touch detection training
│   ├── train_model_multiclass.py # Multiclass marker detection
│   ├── inference.py            # Real-time prediction engine
│   ├── touch_detection_model.pkl # Trained binary classifier (69.6% accuracy)
│   └── multiclass_model.pkl   # Multiclass marker detector
├── phantom-dashboard/          # Next.js web dashboard
│   ├── components/
│   │   └── coach-chat.tsx      # AI chat with voice & haptic integration
│   ├── hooks/
│   │   └── use-voice-recording.ts # Voice recording hook
│   └── app/
│       └── api/
│           └── transcribe/     # Whisper API integration
├── start_haptic_system.sh      # Startup script for haptic system
├── start_complete_system.sh    # Full system startup (all services)
├── requirements.txt            # Python dependencies
└── README.md
```

## New Sketch Workflow

When creating a new Arduino sketch:

1. **Create directory**: `hardware/your_sketch_name/`
2. **Create sketch file**: `hardware/your_sketch_name/your_sketch_name.ino`
3. **Let Windsurf generate the VS Code tasks automatically**

Each sketch gets its own compile and upload tasks in VS Code for clean separation.

## Usage

### Method 1: Using VS Code Tasks (Recommended)
1. Open VS Code in this workspace
2. Press `Cmd+Shift+P`
3. Type "Tasks: Run Task"
4. Select "Arduino: Compile and Upload"

### Method 2: Using Terminal Commands
```bash
# Compile only
arduino-cli compile --fqbn arduino:renesas_uno:unor4wifi hardware/tests

# Upload (requires compilation first)
arduino-cli upload --fqbn arduino:renesas_uno:unor4wifi --port /dev/cu.usbmodem34B7DA631B182 hardware/tests

# Compile and upload in one command
arduino-cli compile --fqbn arduino:renesas_uno:unor4wifi hardware/tests && arduino-cli upload --fqbn arduino:renesas_uno:unor4wifi --port /dev/cu.usbmodem34B7DA631B182 hardware/tests
```

## Current Test Sketch

The `hardware/tests/tests.ino` file contains a simple LED blink program that:
- Blinks the built-in LED every second
- Outputs status messages to Serial Monitor
- Demonstrates basic Arduino functionality

## Serial Monitor

To view serial output:
```bash
arduino-cli monitor --port /dev/cu.usbmodem34B7DA631B182 --config baudrate=9600
```

## Recent Updates (2025-09-14)

### EEG Model Training Pipeline
- **Touch Detection Model**: Achieved 69.6% test accuracy for binary classification (touch vs no-touch)
- **Multiclass Marker Detection**: Supports marker 1, marker 2, or none classification
- **Real-time Inference**: Optimized prediction pipeline for live EEG data streams
- **Feature Engineering**: Advanced frequency band analysis (delta, theta, alpha, beta, gamma)
- **Channel Differencing**: Implements C3-C4, P3-P4, P7-P8, T7-T8 spatial features

### Voice Integration
- **Whisper API**: Integrated OpenAI Whisper for voice transcription
- **Voice Recording Hook**: Real-time audio capture in React dashboard
- **Voice-to-Command**: Voice input triggers haptic feedback and AI responses

### Hardware Enhancements
- **Proximity Sensing**: Added ultrasonic sensor with buzzer feedback for obstacle detection
- **Multi-modal Feedback**: Combined haptic, audio, and visual feedback systems

### System Architecture
- **Complete Pipeline**: EEG → Feature Extraction → ML Model → Command → Arduino → Haptic/Audio Feedback
- **Unified Startup**: Single script launches all services (EEG processing, web server, Arduino, voice)

## Troubleshooting

### Haptic Feedback Issues
- **"Coach is thinking" but no haptic**:
  - Check Arduino server is running: `node arduino-server.js`
  - Verify Arduino is connected and sketch is uploaded
  - Check server logs for "HAPTIC command sent"

- **Arduino not detected**:
  - Unplug and reconnect Arduino USB
  - Close Arduino IDE Serial Monitor if open
  - Check port with: `ls /dev/tty.usbmodem*`

- **Haptic motor not moving**:
  - Verify DRV2605 is connected to I2C/Qwiic port
  - Check haptic motor is attached to DRV2605
  - Test with curl: `curl -X POST http://localhost:3001/api/arduino/haptic`

### General Issues
- **Board not detected**: Check USB connection and ensure the correct port is specified
- **Compilation errors**: Verify the sketch syntax and required libraries
- **Upload fails**: Ensure the board is not in use by another application

## Adding Libraries

```bash
# Search for libraries
arduino-cli lib search <library_name>

# Install a library
arduino-cli lib install <library_name>

# List installed libraries
arduino-cli lib list
```
