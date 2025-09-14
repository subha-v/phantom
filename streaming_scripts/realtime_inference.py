#!/usr/bin/env python3
"""
Real-time EEG inference for marker 1 detection
Receives UDP data from OpenBCI GUI, performs inference, and sends results via WebSocket
"""

import socket
import json
import numpy as np
import sys
import os
import asyncio
import websockets
import joblib
import time
from collections import deque
from typing import Tuple, Optional, List
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# Add model_training utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_training', 'utils'))

from features import FeatureExtractor

# Configuration
UDP_IP = "127.0.0.1"
UDP_PORT = 12345
WEBSOCKET_PORT = 8765
MODEL_PATH = "../model_training/models/new_model/new_model.pkl"

# Sampling configuration
INCOMING_SAMPLE_RATE = 500  # Hz from OpenBCI GUI
TARGET_SAMPLE_RATE = 250    # Hz for model
DOWNSAMPLE_FACTOR = 2       # Take every 2nd sample
WINDOW_SIZE = 250           # Samples at 250Hz (1 second)
OVERLAP = 0.5               # 50% overlap for sliding window
UPDATE_INTERVAL = int(WINDOW_SIZE * (1 - OVERLAP))  # 125 samples

# Channel configuration
# Setup v2 to Setup v1 mapping
CHANNEL_REORDER = [2, 3, 4, 5, 6, 7, 0, 1]  # Reorder to match training data

class EEGBuffer:
    """Circular buffer for EEG data with downsampling"""

    def __init__(self, n_channels: int = 8, buffer_size: int = WINDOW_SIZE):
        self.n_channels = n_channels
        self.buffer_size = buffer_size
        self.buffers = [deque(maxlen=buffer_size) for _ in range(n_channels)]
        self.sample_counter = 0

    def add_samples(self, samples: List[List[float]]):
        """Add samples with downsampling from 500Hz to 250Hz"""
        for channel_idx, channel_data in enumerate(samples):
            for sample_idx, sample in enumerate(channel_data):
                # Downsample: take every 2nd sample
                if self.sample_counter % DOWNSAMPLE_FACTOR == 0:
                    self.buffers[channel_idx].append(sample)
                self.sample_counter += 1

    def get_window(self) -> Optional[np.ndarray]:
        """Get current window if buffer is full"""
        if all(len(buf) >= self.buffer_size for buf in self.buffers):
            # Reorder channels to match training data
            reordered = np.zeros((self.n_channels, self.buffer_size))
            for new_idx, old_idx in enumerate(CHANNEL_REORDER):
                reordered[new_idx] = list(self.buffers[old_idx])
            return reordered
        return None

    def is_ready(self) -> bool:
        """Check if buffer has enough samples for inference"""
        return all(len(buf) >= self.buffer_size for buf in self.buffers)

class TouchDetectionInference:
    """Real-time inference for marker 1 detection"""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = None
        self.scaler = None
        self.extractor = FeatureExtractor(sample_rate=TARGET_SAMPLE_RATE, window_size=WINDOW_SIZE)

        # Adjust path if running from streaming_scripts directory
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load the trained model and scaler"""
        print(f"Loading model from {model_path}...")
        model_data = joblib.load(model_path)

        if isinstance(model_data, dict):
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            print(f"Model loaded successfully (accuracy: {model_data.get('accuracy', 'N/A'):.1%})")
        else:
            raise ValueError("Invalid model file format")

    def compute_channel_differences(self, data: np.ndarray) -> np.ndarray:
        """Compute channel differences as used in training"""
        # Channel pairs: C3-C4, P3-P4, P7-P8, T7-T8
        # After reordering: indices [0-1, 4-5, 2-3, 6-7]
        differences = np.zeros((4, data.shape[1]))
        differences[0] = data[0] - data[1]  # C3-C4
        differences[1] = data[4] - data[5]  # P3-P4
        differences[2] = data[2] - data[3]  # P7-P8
        differences[3] = data[6] - data[7]  # T7-T8
        return differences

    def extract_features(self, window: np.ndarray) -> np.ndarray:
        """Extract features from EEG window"""
        # Compute channel differences
        differences = self.compute_channel_differences(window)

        # Channel pair names matching training
        channel_pairs = ['C3-C4', 'P3-P4', 'P7-P8', 'T7-T8']

        # Build feature dictionary matching training format
        all_features_dict = {}
        for idx, (pair_name, channel_diff) in enumerate(zip(channel_pairs, differences)):
            # Extract features for this channel pair
            features = self.extractor.extract_all_features_for_window(channel_diff)

            # Add channel pair prefix to match training format
            for key, value in features.items():
                all_features_dict[f'{pair_name}_{key}'] = value

        # Convert to ordered array matching the model's expected feature order
        # The model expects features in the same order as they were during training
        # Ensure we have exactly 160 features (40 per channel pair)
        feature_values = list(all_features_dict.values())

        if len(feature_values) != 160:
            print(f"Warning: Expected 160 features, got {len(feature_values)}")

        return np.array(feature_values, dtype=np.float32)

    def predict(self, window: np.ndarray) -> Tuple[int, float]:
        """Perform inference on EEG window"""
        # Extract features
        features = self.extract_features(window)

        # Scale features
        if self.scaler:
            features_scaled = self.scaler.transform([features])
        else:
            features_scaled = [features]

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Return prediction and confidence
        # Always return the confidence for the predicted class
        confidence = probabilities[prediction]

        return prediction, confidence

class RealtimeEEGProcessor:
    """Main processor combining UDP receiver, inference, and WebSocket server"""

    def __init__(self):
        self.buffer = EEGBuffer()
        self.inference = TouchDetectionInference()
        self.websocket_clients = set()
        self.status_queue = queue.Queue()
        self.sample_count = 0
        self.last_update = 0

    def process_udp_data(self):
        """Receive and process UDP data from OpenBCI GUI"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        print(f"Listening for UDP data on {UDP_IP}:{UDP_PORT}")

        while True:
            try:
                data, addr = sock.recvfrom(4096)  # Larger buffer for full packets
                json_string = data.decode('utf-8')
                parsed_json = json.loads(json_string)

                # Extract raw EEG data
                if parsed_json.get('type') == 'timeSeriesRaw' and 'data' in parsed_json:
                    raw_eeg_data = parsed_json["data"]

                    # Add to buffer (with downsampling)
                    self.buffer.add_samples(raw_eeg_data)
                    self.sample_count += len(raw_eeg_data[0]) if raw_eeg_data else 0

                    # Check if we should perform inference
                    if self.buffer.is_ready() and (self.sample_count - self.last_update) >= UPDATE_INTERVAL * DOWNSAMPLE_FACTOR:
                        window = self.buffer.get_window()
                        if window is not None:
                            self.perform_inference(window)
                            self.last_update = self.sample_count

            except json.JSONDecodeError:
                print(f"Error decoding JSON")
            except Exception as e:
                print(f"UDP processing error: {e}")

    def perform_inference(self, window: np.ndarray):
        """Perform inference and queue result for WebSocket broadcast"""
        try:
            prediction, confidence = self.inference.predict(window)

            # Determine status
            status = "touch" if prediction == 1 and confidence > 0.6 else "normal"

            # Create status message
            status_msg = {
                "status": status,
                "confidence": float(confidence),
                "timestamp": time.time(),
                "raw_prediction": int(prediction)
            }

            # Queue for WebSocket broadcast
            self.status_queue.put(status_msg)

            # Print to console
            print(f"Status: {status} | Confidence: {confidence:.2f} | Prediction: {prediction}")

        except Exception as e:
            print(f"Inference error: {e}")

    async def websocket_handler(self, websocket):
        """Handle WebSocket connections"""
        self.websocket_clients.add(websocket)
        print(f"WebSocket client connected. Total clients: {len(self.websocket_clients)}")

        try:
            # Send initial status
            await websocket.send(json.dumps({
                "status": "normal",
                "confidence": 0.0,
                "timestamp": time.time(),
                "message": "Connected to EEG inference server"
            }))

            # Keep connection alive
            await websocket.wait_closed()
        finally:
            self.websocket_clients.remove(websocket)
            print(f"WebSocket client disconnected. Total clients: {len(self.websocket_clients)}")

    async def broadcast_status(self):
        """Broadcast status updates to all WebSocket clients"""
        while True:
            try:
                # Check for new status updates
                if not self.status_queue.empty():
                    status_msg = self.status_queue.get()

                    # Broadcast to all connected clients
                    if self.websocket_clients:
                        message = json.dumps(status_msg)
                        await asyncio.gather(
                            *[client.send(message) for client in self.websocket_clients],
                            return_exceptions=True
                        )

                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                print(f"Broadcast error: {e}")
                await asyncio.sleep(0.1)

    async def start_websocket_server(self):
        """Start WebSocket server"""
        print(f"Starting WebSocket server on port {WEBSOCKET_PORT}")
        async with websockets.serve(self.websocket_handler, "localhost", WEBSOCKET_PORT):
            await self.broadcast_status()

    def run(self):
        """Main run method"""
        print("="*60)
        print("Real-time EEG Inference System")
        print("="*60)
        print(f"Model: new_model (binary classification)")
        print(f"Input: 500Hz, 8 channels from OpenBCI")
        print(f"Processing: Downsampling to 250Hz, window size 250 samples")
        print(f"Output: WebSocket on port {WEBSOCKET_PORT}")
        print("="*60)

        # Start UDP receiver in separate thread
        udp_thread = threading.Thread(target=self.process_udp_data, daemon=True)
        udp_thread.start()

        # Run WebSocket server in main thread
        try:
            asyncio.run(self.start_websocket_server())
        except KeyboardInterrupt:
            print("\nShutting down...")

if __name__ == "__main__":
    processor = RealtimeEEGProcessor()
    processor.run()