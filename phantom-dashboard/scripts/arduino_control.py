#!/usr/bin/env python3
"""
Arduino control script for haptic feedback
"""

import serial
import serial.tools.list_ports
import time
import sys
import json

def find_arduino():
    """Find the Arduino port"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "usbmodem" in port.device or "usbserial" in port.device or "Arduino" in port.description:
            return port.device
    return None

def send_command(command):
    """Send a command to Arduino and return response"""
    try:
        arduino_port = find_arduino()

        if not arduino_port:
            return json.dumps({
                "success": False,
                "error": "Arduino not found. Please check connection."
            })

        # Open serial connection
        ser = serial.Serial(arduino_port, 9600, timeout=2)
        time.sleep(2)  # Wait for Arduino to initialize

        # Send command
        ser.write(f"{command}\n".encode())

        # Read response
        time.sleep(0.5)
        response = ""
        while ser.in_waiting:
            response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.1)

        ser.close()

        return json.dumps({
            "success": True,
            "command": command,
            "response": response.strip(),
            "port": arduino_port
        })

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No command provided"}))
        sys.exit(1)

    command = sys.argv[1]
    result = send_command(command)
    print(result)