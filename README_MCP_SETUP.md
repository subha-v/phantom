# Phantom Pain Coach - MCP Integration Setup

This guide will help you set up and run the intelligent coaching agent for phantom pain management.

## Overview

The system consists of:
1. **MCP Server** (Python) - Integrates Claude API and Exa search
2. **Next.js Frontend** - Enhanced chat interface with MCP tool indicators
3. **Poke Integration** - For automations and scheduled coaching

## Prerequisites

- Node.js (v18 or higher)
- Python 3.9+
- npm or yarn
- API Keys (already configured in .env files)

## Installation

### 1. Install MCP Server Dependencies

```bash
cd mcp-phantom-coach

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd phantom-dashboard
npm install
```

## Running the System

You'll need two terminal windows:

### Terminal 1: Start the MCP Server

```bash
cd mcp-phantom-coach
# Activate virtual environment if you created one
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the server
python server.py
```

The MCP server will start on `http://localhost:8000/mcp`

You should see:
```
INFO:     Starting MCP server on http://localhost:8000/mcp
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start the Next.js Application

```bash
cd phantom-dashboard
npm run dev
```

The application will start on `http://localhost:3000`

## Using the Chat

1. Open your browser and go to `http://localhost:3000`
2. Navigate to the chat interface
3. Start chatting with the coach!

### Features

The coach can help with:
- **Pain Tracking**: "My pain is 6/10 today"
- **Exercise Suggestions**: "What exercises can help with burning phantom pain?"
- **Medical Information**: "Why does phantom pain occur?"
- **Coping Strategies**: "How can I cope with sudden phantom pain at work?"

### Visual Indicators

When the coach is using MCP tools, you'll see:
- `Thinking [tool_name]...` - Shows which tool is being called
- Spinning loader icon - Indicates processing
- System messages appear as centered, italic text

## Poke Integration

To connect this MCP server to Poke:

1. Go to https://poke.com/settings/connections/integrations/new
2. Add your MCP server URL: `http://localhost:8000/mcp`
3. If needed, ask Poke support to disable the bouncer for your account
4. Create automations like:
   - Daily pain check-ins
   - Exercise reminders
   - Weekly progress summaries

## Troubleshooting

### MCP Server Won't Start
- Check that Python dependencies are installed: `pip install -r requirements.txt`
- Verify API keys are in `mcp-phantom-coach/.env`
- Check port 8000 is not in use: `lsof -i :8000` (Mac/Linux)

### Chat Not Connecting
- Ensure MCP server is running (Terminal 1)
- Check browser console for errors
- Verify `NEXT_PUBLIC_MCP_SERVER_URL` in `.env.local`

### API Errors
- Verify your Claude API key is valid
- Check Exa API key is correct
- Monitor MCP server logs for specific errors

## API Keys

The following API keys are already configured:
- **Claude API**: For intelligent coaching responses
- **Exa API**: For searching medical information

These are stored in:
- `mcp-phantom-coach/.env` (for MCP server)
- `phantom-dashboard/.env.local` (for frontend URL configuration)

## Development Notes

### MCP Tools Available

The MCP server provides these tools:
1. `get_coaching_response` - Personalized coaching using Claude
2. `search_medical_info` - Search phantom pain resources with Exa
3. `track_pain_level` - Log and analyze pain patterns
4. `suggest_exercises` - Targeted exercise recommendations
5. `get_coping_strategies` - Situation-specific coping techniques

### Customizing Responses

To modify the coaching personality or knowledge base:
1. Edit `mcp-phantom-coach/server.py`
2. Update the system prompt in `get_coaching_response`
3. Restart the MCP server

### Adding New MCP Tools

1. Add a new `@mcp.tool` decorated function in `server.py`
2. Update `lib/mcp-client.ts` to call the new tool
3. Add UI indicators in `components/coach-chat.tsx`

## Security Notes

- **Never commit API keys to git**
- The `.env` files are gitignored by default
- Run the MCP server locally only (not exposed to internet)
- For production, implement proper authentication

## Support

For issues or questions:
- Check the console logs in both terminals
- Review the MCP server output for detailed error messages
- Ensure all dependencies are correctly installed