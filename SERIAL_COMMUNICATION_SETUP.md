# Serial Communication Setup: Arduino UNO R4 WiFi ↔ Xiao Seed ESP32

## Hardware Connections

### Wiring Diagram
```
Arduino UNO R4 WiFi          Xiao Seed ESP32C3
┌─────────────────┐          ┌─────────────────┐
│                 │          │                 │
│  Digital Pin 0  │ ────────→│ D7 (RX)         │
│  (TX)           │          │                 │
│                 │          │                 │
│  Digital Pin 1  │ ←──────── │ D6 (TX)         │
│  (RX)           │          │                 │
│                 │          │                 │
│  GND            │ ────────→│ GND             │
│                 │          │                 │
└─────────────────┘          └─────────────────┘
```

### Physical Connections
1. **Arduino TX (Pin 0)** → **ESP32 RX (D7)**
2. **Arduino RX (Pin 1)** → **ESP32 TX (D6)**  
3. **Arduino GND** → **ESP32 GND**

⚠️ **Important Notes:**
- Both devices should be powered separately (Arduino via USB, ESP32 via USB)
- Do NOT connect VCC/3.3V between devices to avoid power conflicts
- Ensure both devices share a common ground connection

## Software Configuration

### ESP32 Code Configuration
The ESP32 sketch uses:
- **Serial1** for Arduino communication (D7=RX, D6=TX)
- **Serial** (USB) for debugging and command input
- **Baud Rate:** 9600 (matches Arduino)

### Arduino Code Configuration  
The Arduino sketch uses:
- **Serial** (USB/Pins 0,1) for ESP32 communication
- **Baud Rate:** 9600 (matches ESP32)

## Testing the Connection

### Step 1: Upload Code
1. Upload `esp32_serial_bridge.ino` to your Xiao Seed ESP32
2. Upload `master_proximity_feedback.ino` to your Arduino UNO R4 WiFi

### Step 2: Test Communication
1. Connect both devices via USB to your computer
2. Open Serial Monitor for ESP32 (115200 baud)
3. Open Serial Monitor for Arduino (9600 baud) in a separate window

### Step 3: Send Test Commands
In the ESP32 Serial Monitor, type:
```
help
subtle
moderate  
high
rainbow
```

You should see:
- ESP32 Serial Monitor: Command acknowledgments and Arduino responses
- Arduino Serial Monitor: Command execution and feedback messages

## Serial Protocol Format

### Command Structure
```
<command>\n
```

### Supported Commands
| Command | Description | Arduino Response |
|---------|-------------|------------------|
| `subtle` | Gentle pain feedback | "=== SUBTLE PAIN FEEDBACK ===" |
| `moderate` | Medium pain feedback | "=== MODERATE PAIN FEEDBACK ===" |
| `high` | Intense pain feedback | "=== HIGH PAIN FEEDBACK ===" |
| `soft` | Soft visual feedback | "=== SOFT VISUAL FEEDBACK ===" |
| `rainbow` | Rainbow chase pattern | "=== RAINBOW CHASE ===" |
| `proximity` | Toggle proximity monitoring | "Proximity feedback: ENABLED/DISABLED" |
| `help` | Show available commands | Command list |

### Response Format
Arduino sends human-readable status messages back to ESP32:
```
Command received: <command>
<execution messages>
<completion message>
```

## Troubleshooting

### No Communication
1. Check wiring connections (TX↔RX, RX↔TX, GND↔GND)
2. Verify baud rates match (9600)
3. Ensure both devices are powered and programmed correctly

### Garbled Text
1. Check baud rate settings
2. Verify TX/RX pins are not swapped
3. Ensure stable power supply to both devices

### Commands Not Working
1. Verify command spelling (case-insensitive)
2. Check that Arduino is receiving commands (monitor Serial output)
3. Ensure ESP32 is validating commands correctly

## Next Steps: Whisper API Integration

Once serial communication is working, the ESP32 can be enhanced with:
1. WiFi connectivity for Whisper API calls
2. Microphone input for voice capture
3. Audio processing and command recognition
4. Voice-to-command translation

The `sendCommandToArduino()` function in the ESP32 sketch is ready for this integration.
