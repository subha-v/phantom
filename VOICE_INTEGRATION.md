# Voice Integration with Local Whisper (FREE - No API Key!)

## Overview
This feature allows users to interact with the chat using voice input. Speech is recorded from the microphone and transcribed to text using OpenAI's **open-source Whisper model running locally** on your machine. No API keys or cloud services required!

## Features
- 🎤 Click-to-record voice input
- 🔄 Real-time transcription using local Whisper model
- 📝 Automatic text insertion into chat
- 🔴 Visual feedback during recording
- ⚡ Fast and accurate transcription
- 💰 **100% FREE** - No API keys or subscriptions needed!
- 🔒 **Private** - Audio never leaves your machine

## Setup

### 1. Install Python Dependencies
```bash
pip3 install -r requirements.txt
```
This installs:
- `openai-whisper` - The open-source Whisper model
- `flask` - Python web server
- `flask-cors` - CORS support

### 2. Start the Whisper Server
```bash
python3 whisper-server.py
```
The server will:
- Download the Whisper model on first run (~140MB for base model)
- Start listening on port 5001
- Be ready to transcribe audio

### 3. Choose Model Size (Optional)
Edit `whisper-server.py` to change model size:
```python
MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
```
- **tiny** - Fastest, least accurate (~39MB)
- **base** - Good balance (default) (~140MB)
- **small** - Better accuracy (~460MB)
- **medium** - High accuracy (~1.5GB)
- **large** - Best accuracy (~3GB)

## Usage

### Recording Voice
1. Click the **microphone button** (🎤) next to the send button
2. The button turns **red** and shows 🔴 while recording
3. Speak your message clearly
4. Click the button again to **stop recording**
5. The speech is automatically transcribed and appears in the text input
6. Press Send or Enter to send the message

### Visual States
- **🎤 (outline)** - Ready to record
- **🔴 (red)** - Currently recording
- **⏳ (spinner)** - Processing/transcribing speech
- **Error message** - Shown below input if something goes wrong

## How It Works

### Technical Flow
1. **User clicks microphone** → Requests browser microphone permission
2. **MediaRecorder API** → Records audio as WebM format
3. **Stop recording** → Audio blob is created
4. **Send to Next.js API** → POST to `/api/transcribe` endpoint
5. **Forward to Python server** → Sent to local Whisper server on port 5001
6. **Local Whisper model** → Transcribes audio to text (no internet needed!)
7. **Return text** → Populated in chat input field

### Components
- **`use-voice-recording.ts`** - Custom React hook for recording logic
- **`coach-chat.tsx`** - Updated UI with microphone button
- **`/api/transcribe/route.ts`** - API endpoint that forwards to local Whisper
- **`whisper-server.py`** - Python server running local Whisper model

## Browser Compatibility
- ✅ Chrome/Edge (Recommended)
- ✅ Firefox
- ✅ Safari (macOS/iOS)
- ⚠️ Requires HTTPS in production

## Troubleshooting

### Microphone Permission Denied
- Check browser settings for microphone permissions
- Ensure the site has permission to access microphone
- Try refreshing the page

### No Audio Recorded
- Check microphone is connected and working
- Test microphone in system settings
- Ensure browser has correct input device selected

### Transcription Fails
- Check Whisper server is running: `python3 whisper-server.py`
- Verify Python dependencies installed: `pip3 install -r requirements.txt`
- Check server logs for errors
- Ensure port 5001 is not in use by another service

### Poor Transcription Quality
- Speak clearly and at normal pace
- Reduce background noise
- Position microphone closer
- Use a quality microphone/headset

## Advantages of Local Whisper
- **No API costs** - Completely free to use
- **No rate limits** - Process as much audio as you want
- **Privacy** - Audio never leaves your computer
- **No internet required** - Works offline after model download
- **Customizable** - Choose model size based on your needs

## System Requirements
- Python 3.7 or higher
- ~140MB disk space for base model (more for larger models)
- 4GB+ RAM recommended
- CPU with AVX support (most modern CPUs)

## Future Enhancements
- [ ] Auto-send after transcription option
- [ ] Multiple language support
- [ ] Voice activity detection
- [ ] Offline transcription option
- [ ] Custom wake words
- [ ] Continuous conversation mode