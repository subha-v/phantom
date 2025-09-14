# Phantom - Arduino Haptic Feedback with Chat Interface

A system that integrates Arduino-based haptic feedback with a web chat interface, allowing users to trigger physical haptic responses through chat commands.

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
├── hardware/
│   ├── routines1_test/
│   │   └── routines1_test.ino  # Haptic feedback Arduino sketch
│   ├── led_blink/
│   │   └── led_blink.ino       # Basic LED blink sketch
│   └── rainbow_led/
│       └── rainbow_led.ino     # Rainbow breathing effect
├── phantom-dashboard/          # Next.js web dashboard
│   ├── components/
│   │   └── coach-chat.tsx      # Chat component with haptic trigger
│   └── app/                    # Next.js app directory
├── start_haptic_system.sh      # Startup script for entire system
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
