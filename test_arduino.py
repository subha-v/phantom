#!/usr/bin/env python3
"""
Test script to send haptic commands to Arduino
"""

import serial
import serial.tools.list_ports
import time
import sys

def find_arduino():
    """Find the Arduino port"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Found port: {port.device} - {port.description}")
        if "usbmodem" in port.device or "usbserial" in port.device or "Arduino" in port.description:
            return port.device
    return None

def send_command(port_name, command):
    """Send a command to Arduino and read response"""
    try:
        # Open serial connection
        ser = serial.Serial(port_name, 9600, timeout=2)
        time.sleep(2)  # Wait for Arduino to initialize

        print(f"Sending command: {command}")
        ser.write(f"{command}\n".encode())

        # Read response
        time.sleep(0.5)
        response = ""
        while ser.in_waiting:
            response += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            time.sleep(0.1)

        if response:
            print(f"Arduino response: {response}")
        else:
            print("No response received (command may have executed successfully)")

        ser.close()
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=== Arduino Haptic Test ===")

    # Find Arduino port
    arduino_port = find_arduino()

    if not arduino_port:
        print("Arduino not found! Please check connection.")
        sys.exit(1)

    print(f"\nUsing Arduino on port: {arduino_port}")

    # Test all three commands
    commands = ["subtle", "moderate", "high"]

    for cmd in commands:
        print(f"\n--- Testing {cmd.upper()} haptic ---")
        if send_command(arduino_port, cmd):
            print(f"✓ {cmd} command sent successfully")
        else:
            print(f"✗ Failed to send {cmd} command")

        if cmd != commands[-1]:
            print("Waiting 3 seconds before next command...")
            time.sleep(3)

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()