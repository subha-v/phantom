# Arduino UNO R4 WiFi Development Setup

This project is configured for Arduino UNO R4 WiFi development using Arduino CLI and VS Code.

## Hardware Setup
- **Board**: Arduino UNO R4 WiFi
- **Port**: `/dev/cu.usbmodem34B7DA631B182`
- **FQBN**: `arduino:renesas_uno:unor4wifi`

## Prerequisites
- Arduino CLI (already installed - version 1.3.1)
- VS Code with this workspace open

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
│   └── tasks.json          # VS Code task definitions
│   └── keybindings.json    # Keyboard shortcuts for task interruption
├── arduino-cli.yaml        # Arduino CLI configuration
├── hardware/
│   ├── led_blink/
│   │   └── led_blink.ino   # Basic LED blink sketch
│   ├── rainbow_led/
│   │   └── rainbow_led.ino # Rainbow breathing effect
│   └── stop_sketch/
│       └── stop_sketch.ino # Empty sketch to stop programs
├── add_sketch_tasks.py     # Helper script for adding new sketch tasks
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
