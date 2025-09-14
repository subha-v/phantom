# Haptic Feedback Integration

## Overview
This integration allows users to trigger haptic feedback on the Arduino device through the chat interface in the Phantom Dashboard.

## Components

### 1. Arduino Sketch (`hardware/routines1_test/routines1_test.ino`)
- Modified to listen for serial commands
- Responds to "HAPTIC" command to trigger haptic feedback
- Uses effect 47 (strong click) by default
- Can also accept specific effect numbers with "HAPTIC_[number]" format

### 2. Arduino Communication Server (`arduino-server.js`)
- Node.js Express server running on port 3001
- Handles serial communication with Arduino
- Auto-detects Arduino port
- Provides REST API endpoints:
  - `POST /api/arduino/haptic` - Triggers haptic feedback
  - `GET /api/arduino/status` - Checks Arduino connection status

### 3. Dashboard Chat Component (`phantom-dashboard/components/coach-chat.tsx`)
- Detects "play haptic feedback" in user messages
- Sends request to Arduino server
- Provides feedback on success/failure

## Setup Instructions

### 1. Upload Arduino Sketch
```bash
# Upload the modified sketch to your Arduino
arduino-cli upload --fqbn arduino:renesas_uno:unor4wifi --port /dev/cu.usbmodem* hardware/routines1_test
```

### 2. Install Dependencies
```bash
# Install Arduino server dependencies
npm install express cors serialport @serialport/parser-readline

# Alternatively, use the provided package.json
mv arduino-server-package.json package.json
npm install
```

### 3. Start the System

#### Option A: Use the startup script
```bash
./start_haptic_system.sh
```

#### Option B: Manual startup
```bash
# Terminal 1: Start Arduino server
node arduino-server.js

# Terminal 2: Start dashboard
cd phantom-dashboard
npm run dev
```

## Usage

1. Open the Phantom Dashboard at http://localhost:3000
2. Navigate to the chat interface
3. Type "play haptic feedback" in the chat
4. The Arduino will trigger haptic feedback effect 47 (strong click)

## How It Works

1. User types "play haptic feedback" in chat
2. Chat component detects the command
3. Sends POST request to Arduino server at localhost:3001
4. Arduino server sends "HAPTIC\n" command via serial port
5. Arduino receives command and triggers haptic motor
6. Arduino sends confirmation back via serial
7. User sees success message in chat

## Troubleshooting

### Arduino Not Detected
- Check USB connection
- Verify Arduino shows up in `/dev/cu.usbmodem*` or `/dev/ttyUSB*`
- Try unplugging and reconnecting the Arduino

### Server Won't Start
- Check if port 3001 is already in use
- Ensure serialport npm package is installed correctly
- May need to rebuild serialport bindings: `npm rebuild serialport`

### Haptic Not Triggering
- Open Arduino Serial Monitor to see if commands are received
- Check baud rate is set to 9600
- Verify DRV2605 haptic driver is connected properly
- Check Wire1 connection on Arduino R4 WiFi

## Customization

### Different Haptic Effects
Modify the effect number in the Arduino code:
```cpp
playHapticSingle(47);  // Change 47 to any effect 1-117
```

### Custom Commands
Add new commands in the Arduino loop:
```cpp
if (command == "CUSTOM_COMMAND") {
    // Your custom action
}
```

### Multiple Haptic Patterns
The chat can be extended to recognize patterns like:
- "play soft haptic" → Effect 1-10
- "play strong haptic" → Effect 40-50
- "play pattern [name]" → Custom sequences

## Technical Details

- **Serial Communication**: 9600 baud
- **Default Haptic Effect**: 47 (Strong Click)
- **Arduino Board**: Arduino Uno R4 WiFi
- **Haptic Driver**: Adafruit DRV2605
- **Server Port**: 3001
- **Dashboard Port**: 3000