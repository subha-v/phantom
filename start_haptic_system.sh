#!/bin/bash

echo "🚀 Starting Phantom Haptic Feedback System"
echo "==========================================="

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $ARDUINO_PID 2>/dev/null
    kill $DASHBOARD_PID 2>/dev/null
    exit 0
}

# Set up trap to handle Ctrl+C
trap cleanup INT

# Check if Arduino is connected
echo "🔍 Checking for Arduino connection..."
if ls /dev/cu.usbmodem* 1> /dev/null 2>&1 || ls /dev/ttyUSB* 1> /dev/null 2>&1; then
    echo "✅ Arduino detected"
else
    echo "⚠️  Warning: Arduino not detected. Please connect your Arduino."
    echo "   The system will still start but haptic feedback won't work."
fi

# Install dependencies for Arduino server if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Arduino server dependencies..."
    npm install express cors serialport @serialport/parser-readline
fi

# Start Arduino server
echo ""
echo "🔌 Starting Arduino communication server on port 3001..."
node arduino-server.js &
ARDUINO_PID=$!
sleep 2

# Check if Arduino server started
if ps -p $ARDUINO_PID > /dev/null; then
    echo "✅ Arduino server is running (PID: $ARDUINO_PID)"
else
    echo "❌ Failed to start Arduino server"
    exit 1
fi

# Start dashboard
echo ""
echo "🎨 Starting Phantom Dashboard..."
cd phantom-dashboard
npm run dev &
DASHBOARD_PID=$!

# Wait a moment for the dashboard to start
sleep 3

echo ""
echo "==========================================="
echo "✨ System is ready!"
echo ""
echo "📱 Dashboard: http://localhost:3000"
echo "🔌 Arduino Server: http://localhost:3001"
echo ""
echo "💬 To test haptic feedback:"
echo "   1. Open the dashboard in your browser"
echo "   2. Type 'play haptic feedback' in the chat"
echo ""
echo "Press Ctrl+C to stop all services"
echo "==========================================="

# Keep script running
wait