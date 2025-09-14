#!/bin/bash

echo "🚀 Starting Phantom Complete System"
echo "==========================================="

# Function to handle cleanup
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $WHISPER_PID 2>/dev/null
    kill $ARDUINO_PID 2>/dev/null
    kill $DASHBOARD_PID 2>/dev/null
    exit 0
}

# Set up trap to handle Ctrl+C
trap cleanup INT

# Check Python and pip
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3."
    exit 1
fi

# Install Python dependencies if needed
echo "📦 Checking Python dependencies..."
if ! python3 -c "import whisper" 2>/dev/null; then
    echo "Installing Whisper and dependencies..."
    pip3 install -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "✅ Python dependencies already installed"
fi

# Check if Arduino is connected
echo ""
echo "🔍 Checking for Arduino connection..."
if ls /dev/cu.usbmodem* 1> /dev/null 2>&1 || ls /dev/ttyUSB* 1> /dev/null 2>&1; then
    echo "✅ Arduino detected"
else
    echo "⚠️  Warning: Arduino not detected. Haptic feedback won't work."
fi

# Install Node dependencies for Arduino server if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Arduino server dependencies..."
    npm install express cors serialport @serialport/parser-readline
fi

# Start Whisper server
echo ""
echo "🎤 Starting Whisper voice transcription server on port 5001..."
python3 whisper-server.py &
WHISPER_PID=$!
sleep 3

# Check if Whisper server started
if ps -p $WHISPER_PID > /dev/null; then
    echo "✅ Whisper server is running (PID: $WHISPER_PID)"
else
    echo "❌ Failed to start Whisper server"
    exit 1
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
echo "🎤 Whisper Server: http://localhost:5001"
echo ""
echo "💬 Features available:"
echo "   • Type messages in the chat"
echo "   • Click microphone to use voice input (FREE - no API key!)"
echo "   • Type 'play haptic feedback' to trigger Arduino"
echo ""
echo "Press Ctrl+C to stop all services"
echo "==========================================="

# Keep script running
wait