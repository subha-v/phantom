#!/usr/bin/env python3
"""
Local Whisper Server for Speech-to-Text
Uses OpenAI's open-source Whisper model locally - no API key needed!
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import logging
from werkzeug.utils import secure_filename
import ssl
import certifi

# Fix SSL certificate issues on macOS
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load Whisper model
# Models: tiny, base, small, medium, large
# Smaller models are faster but less accurate
MODEL_SIZE = "base"  # Good balance of speed and accuracy
logger.info(f"Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)
logger.info("Whisper model loaded successfully")

# Allowed audio extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'webm', 'ogg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file using local Whisper model
    """
    try:
        # Check if audio file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']

        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            # Transcribe with Whisper
            logger.info(f"Transcribing audio file: {temp_path}")
            result = model.transcribe(
                temp_path,
                language='en',  # Force English, remove for auto-detection
                fp16=False  # Set to True if you have a GPU
            )

            # Extract transcribed text
            text = result['text'].strip()
            logger.info(f"Transcription complete: {text[:100]}...")

            return jsonify({
                'success': True,
                'text': text,
                'language': result.get('language', 'en')
            })

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return jsonify({
            'error': 'Transcription failed',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': MODEL_SIZE,
        'ready': True
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Get available Whisper models"""
    return jsonify({
        'current': MODEL_SIZE,
        'available': ['tiny', 'base', 'small', 'medium', 'large'],
        'recommended': 'base'
    })

if __name__ == '__main__':
    port = 5001
    logger.info(f"Starting Whisper server on port {port}")
    logger.info(f"Model: {MODEL_SIZE}")
    logger.info("Server ready for transcription requests")

    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Set to True for development
        threaded=True
    )