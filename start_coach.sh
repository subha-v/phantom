#!/bin/bash

# Start script for Phantom Pain Coach with MCP integration

echo "🚀 Starting Phantom Pain Coach System..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $MCP_PID 2>/dev/null
    kill $NEXT_PID 2>/dev/null
    exit
}

# Set up trap for cleanup
trap cleanup INT TERM

# Start MCP Server
echo "📡 Starting MCP Server..."
cd mcp-phantom-coach

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

python server.py &
MCP_PID=$!

# Wait for MCP server to start
sleep 3

# Start Next.js Frontend
echo "🎨 Starting Next.js Frontend..."
cd ../phantom-dashboard

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

npm run dev &
NEXT_PID=$!

echo ""
echo "✅ System is running!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 MCP Server: http://localhost:8000/mcp"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for processes
wait